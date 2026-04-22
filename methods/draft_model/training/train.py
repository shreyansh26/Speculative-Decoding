from __future__ import annotations

"""
Draft-model training builds a compact Qwen-like LM that shares the target
tokenizer, then saves only the draft model weights and config for speculation.
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file, save_file

from common.qwen3 import Qwen3Config, Qwen3ForCausalLM
from common.tokenizer import load_prompts, load_tokenizer, render_prompt


DEFAULT_TARGET_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_INIT_MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DISTILLATION_DATA = "data/ultrachat_300_trunc512_qwen7b_greedy16_ids.jsonl"
DEFAULT_OUTPUT_DIR = "checkpoints/draft_model_qwen25_05b_2ep"
DEFAULT_SEQ_LEN = 528
DEFAULT_EPOCHS = 2
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 1
DEFAULT_LR = 5e-5


@dataclass(slots=True)
class MiniQwenConfig:
    hidden_size: int = 768
    num_layers: int = 8
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    intermediate_size: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    def to_qwen3_config(self, vocab_size: int, max_position_embeddings: int) -> Qwen3Config:
        return Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=self.rope_theta,
            rms_norm_eps=self.rms_norm_eps,
        )


@dataclass(slots=True)
class TrainingExample:
    token_ids: torch.Tensor
    loss_mask: torch.Tensor


def build_draft_model(
    vocab_size: int,
    max_position_embeddings: int,
    config: MiniQwenConfig | None = None,
) -> Qwen3ForCausalLM:
    draft_config = (config or MiniQwenConfig()).to_qwen3_config(
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
    )
    return Qwen3ForCausalLM(draft_config)


def parse_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def build_training_sequences(
    data_path: str | Path,
    tokenizer,
    seq_len: int,
) -> list[torch.Tensor]:
    rows = load_prompts(data_path, tokenizer=tokenizer)
    sequences: list[torch.Tensor] = []
    chunk_len = seq_len + 1
    for row in rows:
        token_ids = tokenizer.encode(row.prompt, add_special_tokens=False)
        if len(token_ids) < 2:
            continue
        if len(token_ids) <= chunk_len:
            sequences.append(torch.tensor(token_ids, dtype=torch.long))
            continue
        for start in range(0, len(token_ids) - 1, seq_len):
            chunk = token_ids[start : start + chunk_len]
            if len(chunk) >= 2:
                sequences.append(torch.tensor(chunk, dtype=torch.long))
    if not sequences:
        raise ValueError("dataset did not produce any language-modeling sequences")
    return sequences


def dataset_uses_completion(data_path: str | Path) -> bool:
    with Path(data_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            return "completion" in sample
    return False


def build_distillation_examples(
    data_path: str | Path,
    tokenizer,
    seq_len: int,
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    chunk_len = seq_len + 1
    with Path(data_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if "completion" not in sample:
                raise ValueError(f"{data_path}:{line_number} is missing required 'completion' field")
            if "prompt_ids" in sample:
                prompt_ids = [int(token_id) for token_id in sample["prompt_ids"]]
            else:
                rendered = render_prompt(tokenizer, sample)
                prompt_ids = tokenizer.encode(rendered.prompt, add_special_tokens=False)
            if "completion_ids" in sample:
                completion_ids = [int(token_id) for token_id in sample["completion_ids"]]
            else:
                completion_ids = tokenizer.encode(str(sample["completion"]), add_special_tokens=False)
            if not completion_ids:
                continue
            token_ids = prompt_ids + completion_ids
            loss_mask = ([0] * len(prompt_ids)) + ([1] * len(completion_ids))
            if len(token_ids) > chunk_len:
                token_ids = token_ids[-chunk_len:]
                loss_mask = loss_mask[-chunk_len:]
            if len(token_ids) < 2:
                continue
            if sum(loss_mask[1:]) <= 0:
                continue
            examples.append(
                TrainingExample(
                    token_ids=torch.tensor(token_ids, dtype=torch.long),
                    loss_mask=torch.tensor(loss_mask, dtype=torch.float32),
                )
            )
    if not examples:
        raise ValueError("dataset did not produce any distillation examples")
    return examples


def make_batch(sequences: Sequence[torch.Tensor], batch_size: int, step: int) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    start = (step * batch_size) % len(sequences)
    selected = [sequences[(start + offset) % len(sequences)] for offset in range(batch_size)]
    max_len = max(sequence.numel() for sequence in selected)
    batch = torch.zeros((batch_size, max_len), dtype=torch.long)
    for row, sequence in enumerate(selected):
        batch[row, : sequence.numel()] = sequence
    return batch


def make_distillation_batch(
    examples: Sequence[TrainingExample],
    batch_size: int,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    start = (step * batch_size) % len(examples)
    selected = [examples[(start + offset) % len(examples)] for offset in range(batch_size)]
    max_len = max(example.token_ids.numel() for example in selected)
    batch = torch.zeros((batch_size, max_len), dtype=torch.long)
    loss_mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    for row, example in enumerate(selected):
        batch[row, : example.token_ids.numel()] = example.token_ids
        loss_mask[row, : example.loss_mask.numel()] = example.loss_mask
    return batch, loss_mask


def language_modeling_loss(model: Qwen3ForCausalLM, batch: torch.Tensor) -> torch.Tensor:
    inputs = batch[:, :-1]
    labels = batch[:, 1:]
    logits = model(inputs).logits
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
    )


def masked_language_modeling_loss(
    model: Qwen3ForCausalLM,
    batch: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    inputs = batch[:, :-1]
    labels = batch[:, 1:]
    label_mask = loss_mask[:, 1:]
    logits = model(inputs).logits
    token_losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    )
    flat_mask = label_mask.reshape(-1)
    total_weight = flat_mask.sum()
    if float(total_weight.item()) <= 0.0:
        raise ValueError("masked batch does not contain any supervised completion tokens")
    return (token_losses * flat_mask).sum() / total_weight


def train_steps(
    model: Qwen3ForCausalLM,
    sequences: Sequence[torch.Tensor],
    *,
    steps: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    device: str | torch.device,
    dtype: torch.dtype,
) -> list[float]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if grad_accum <= 0:
        raise ValueError("grad_accum must be positive")

    model = model.to(device=device)
    if device != "cpu":
        model = model.to(dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []

    optimizer.zero_grad(set_to_none=True)
    for step in range(steps):
        batch = make_batch(sequences, batch_size=batch_size, step=step).to(device=device)
        loss = language_modeling_loss(model, batch)
        (loss / grad_accum).backward()
        losses.append(float(loss.item()))
        if (step + 1) % grad_accum == 0 or step == steps - 1:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return losses


def train_distillation_steps(
    model: Qwen3ForCausalLM,
    examples: Sequence[TrainingExample],
    *,
    steps: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    device: str | torch.device,
    dtype: torch.dtype,
) -> list[float]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if grad_accum <= 0:
        raise ValueError("grad_accum must be positive")

    model = model.to(device=device)
    if device != "cpu":
        model = model.to(dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []

    optimizer.zero_grad(set_to_none=True)
    for step in range(steps):
        batch, loss_mask = make_distillation_batch(examples, batch_size=batch_size, step=step)
        batch = batch.to(device=device)
        loss_mask = loss_mask.to(device=device)
        loss = masked_language_modeling_loss(model, batch, loss_mask)
        (loss / grad_accum).backward()
        losses.append(float(loss.item()))
        if (step + 1) % grad_accum == 0 or step == steps - 1:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return losses


def save_draft_checkpoint(
    model: Qwen3ForCausalLM,
    output_dir: str | Path,
    mini_config: MiniQwenConfig | None,
    tokenizer_name_or_path: str,
    target_model_path: str,
    seq_len: int,
) -> None:
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        str(checkpoint_dir / "model.safetensors"),
    )
    metadata = {
        "mini_qwen_config": asdict(mini_config) if mini_config is not None else None,
        "qwen3_config": asdict(model.config),
        "vocab_size": model.config.vocab_size,
        "max_position_embeddings": model.config.max_position_embeddings,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "target_model_path": target_model_path,
        "seq_len": seq_len,
    }
    (checkpoint_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_draft_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> Qwen3ForCausalLM:
    checkpoint_path = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_path / "config.json").read_text(encoding="utf-8"))
    if metadata.get("qwen3_config") is not None:
        model = Qwen3ForCausalLM(Qwen3Config(**metadata["qwen3_config"]))
    else:
        mini_config = MiniQwenConfig(**metadata["mini_qwen_config"])
        model = build_draft_model(
            vocab_size=int(metadata["vocab_size"]),
            max_position_embeddings=int(metadata["max_position_embeddings"]),
            config=mini_config,
        )
    state_dict = load_file(str(checkpoint_path / "model.safetensors"))
    model.load_state_dict(state_dict, strict=True)
    if dtype is not None and str(device) != "cpu":
        model = model.to(dtype=dtype)
    model = model.to(device=device)
    model.set_fast_single_token_gqa(True)
    return model


def resolve_total_steps(
    dataset_size: int,
    *,
    batch_size: int,
    steps: int,
    epochs: int,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if epochs > 0:
        return math.ceil(dataset_size / batch_size) * epochs
    if steps <= 0:
        raise ValueError("steps must be positive when epochs is not set")
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact draft model.")
    parser.add_argument("--target-model-path", default=DEFAULT_TARGET_MODEL_PATH)
    parser.add_argument("--data", default=DEFAULT_DISTILLATION_DATA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--init-model-path", default=DEFAULT_INIT_MODEL_PATH)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.target_model_path)
    using_distillation = dataset_uses_completion(args.data)
    mini_config: MiniQwenConfig | None = None
    if args.init_model_path:
        model = Qwen3ForCausalLM.from_pretrained(args.init_model_path, device="cpu", dtype=dtype)
    else:
        mini_config = MiniQwenConfig()
        model = build_draft_model(
            vocab_size=len(tokenizer),
            max_position_embeddings=max(args.seq_len, 128),
            config=mini_config,
        )

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    if using_distillation:
        examples = build_distillation_examples(args.data, tokenizer=tokenizer, seq_len=args.seq_len)
        total_steps = resolve_total_steps(
            len(examples),
            batch_size=args.batch_size,
            steps=args.steps,
            epochs=args.epochs,
        )
        losses = train_distillation_steps(
            model=model,
            examples=examples,
            steps=total_steps,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            device=args.device,
            dtype=dtype,
        )
    else:
        sequences = build_training_sequences(args.data, tokenizer=tokenizer, seq_len=args.seq_len)
        total_steps = resolve_total_steps(
            len(sequences),
            batch_size=args.batch_size,
            steps=args.steps,
            epochs=args.epochs,
        )
        losses = train_steps(
            model=model,
            sequences=sequences,
            steps=total_steps,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            device=args.device,
            dtype=dtype,
        )
    save_draft_checkpoint(
        model=model,
        output_dir=args.output,
        mini_config=mini_config,
        tokenizer_name_or_path=args.target_model_path,
        target_model_path=args.target_model_path,
        seq_len=args.seq_len,
    )
    (Path(args.output) / "training_summary.json").write_text(
        json.dumps(
            {
                "steps": total_steps,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "using_distillation": using_distillation,
                "init_model_path": args.init_model_path,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

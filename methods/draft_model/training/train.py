from __future__ import annotations

"""
Draft-model training builds a compact Qwen-like LM that shares the target
tokenizer, then saves only the draft model weights and config for speculation.
"""

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from common.qwen3 import Qwen3Config, Qwen3ForCausalLM
from common.tokenizer import load_prompts, load_tokenizer, render_prompt


DEFAULT_TARGET_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_INIT_MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DISTILLATION_DATA = "data/ultrachat_3000_trunc1024_qwen25_7b_greedy128_ids.jsonl"
DEFAULT_EVAL_DATA = "data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl"
DEFAULT_OUTPUT_DIR = "checkpoints/draft_model_qwen25_05b_ultrachat3000"
DEFAULT_SEQ_LEN = 1152
DEFAULT_EPOCHS = 2
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 1
DEFAULT_LR = 5e-5
DEFAULT_EVAL_DRAFT_LEN = 2


@dataclass(slots=True)
class MiniQwenConfig:
    hidden_size: int = 256
    num_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    intermediate_size: int = 768
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True

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
            tie_word_embeddings=self.tie_word_embeddings,
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


def make_batch(
    sequences: Sequence[torch.Tensor],
    batch_size: int,
    step: int,
    *,
    pad_token_id: int = 0,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    start = (step * batch_size) % len(sequences)
    selected = [sequences[(start + offset) % len(sequences)] for offset in range(batch_size)]
    max_len = max(sequence.numel() for sequence in selected)
    batch = torch.full((batch_size, max_len), int(pad_token_id), dtype=torch.long)
    for row, sequence in enumerate(selected):
        batch[row, : sequence.numel()] = sequence
    return batch


def make_distillation_batch(
    examples: Sequence[TrainingExample],
    batch_size: int,
    step: int,
    *,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    start = (step * batch_size) % len(examples)
    selected = [examples[(start + offset) % len(examples)] for offset in range(batch_size)]
    max_len = max(example.token_ids.numel() for example in selected)
    batch = torch.full((batch_size, max_len), int(pad_token_id), dtype=torch.long)
    loss_mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    for row, example in enumerate(selected):
        batch[row, : example.token_ids.numel()] = example.token_ids
        loss_mask[row, : example.loss_mask.numel()] = example.loss_mask
    return batch, loss_mask


def collate_distillation_examples(
    examples: Sequence[TrainingExample],
    *,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("cannot collate an empty example list")
    max_len = max(example.token_ids.numel() for example in examples)
    batch = torch.full((len(examples), max_len), int(pad_token_id), dtype=torch.long)
    loss_mask = torch.zeros((len(examples), max_len), dtype=torch.float32)
    for row, example in enumerate(examples):
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
    pad_token_id: int = 0,
    weight_decay: float = 0.0,
    max_grad_norm: float = 0.0,
    seed: int = 0,
) -> list[float]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if grad_accum <= 0:
        raise ValueError("grad_accum must be positive")

    model = model.to(device=device)
    if device != "cpu":
        model = model.to(dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    order = list(range(len(examples)))
    rng = random.Random(seed)
    rng.shuffle(order)

    optimizer.zero_grad(set_to_none=True)
    for step in range(steps):
        start = (step * batch_size) % len(order)
        if start == 0 and step > 0:
            rng.shuffle(order)
        selected_indices = [order[(start + offset) % len(order)] for offset in range(batch_size)]
        selected = [examples[index] for index in selected_indices]
        batch, loss_mask = collate_distillation_examples(selected, pad_token_id=pad_token_id)
        batch = batch.to(device=device)
        loss_mask = loss_mask.to(device=device)
        loss = masked_language_modeling_loss(model, batch, loss_mask)
        (loss / grad_accum).backward()
        losses.append(float(loss.item()))
        if (step + 1) % grad_accum == 0 or step == steps - 1:
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return losses


@torch.no_grad()
def evaluate_distillation_fit(
    model: Qwen3ForCausalLM,
    examples: Sequence[TrainingExample],
    *,
    batch_size: int,
    device: str | torch.device,
    pad_token_id: int,
    draft_len: int = DEFAULT_EVAL_DRAFT_LEN,
) -> dict[str, float]:
    if not examples:
        return {
            "eval_loss": 0.0,
            "eval_top1_match": 0.0,
            "eval_acceptance_proxy": 0.0,
            "eval_mean_accepted_tokens_per_step_proxy": 0.0,
        }
    if draft_len <= 0:
        raise ValueError("draft_len must be positive")

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0.0
    total_matches = 0.0
    total_proposed = 0.0
    total_accepted = 0.0
    total_steps = 0.0

    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        batch, loss_mask = collate_distillation_examples(
            batch_examples,
            pad_token_id=pad_token_id,
        )
        batch = batch.to(device=device)
        loss_mask = loss_mask.to(device=device)
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
        total_loss += float((token_losses * flat_mask).sum().item())
        total_tokens += float(flat_mask.sum().item())
        predictions = torch.argmax(logits, dim=-1)
        total_matches += float(((predictions == labels) * label_mask.bool()).sum().item())

        matches = ((predictions == labels) * label_mask.bool()).detach().cpu()
        masks = label_mask.bool().detach().cpu()
        for row in range(matches.shape[0]):
            supervised_positions = torch.nonzero(masks[row], as_tuple=False).flatten().tolist()
            if not supervised_positions:
                continue
            for offset in range(0, len(supervised_positions), draft_len):
                positions = supervised_positions[offset : offset + draft_len]
                if not positions:
                    continue
                total_steps += 1.0
                total_proposed += float(len(positions))
                for position in positions:
                    if not bool(matches[row, position].item()):
                        break
                    total_accepted += 1.0

    if was_training:
        model.train()
    return {
        "eval_loss": (total_loss / total_tokens) if total_tokens else 0.0,
        "eval_top1_match": (total_matches / total_tokens) if total_tokens else 0.0,
        "eval_acceptance_proxy": (total_accepted / total_proposed) if total_proposed else 0.0,
        "eval_mean_accepted_tokens_per_step_proxy": (
            total_accepted / total_steps if total_steps else 0.0
        ),
    }


def save_draft_checkpoint(
    model: Qwen3ForCausalLM,
    output_dir: str | Path,
    mini_config: MiniQwenConfig | None,
    tokenizer_name_or_path: str,
    target_model_path: str,
    seq_len: int,
    *,
    draft_base_model_path: str | None = None,
    training_summary: dict[str, object] | None = None,
) -> None:
    model = getattr(model, "_orig_mod", model)
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target_vocab_size = int(model.config.vocab_size)
    if draft_base_model_path:
        target_hf_config = AutoConfig.from_pretrained(
            target_model_path,
            trust_remote_code=True,
        )
        target_vocab_size = int(target_hf_config.vocab_size)

    state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().clone()
        if name in {"model.embed_tokens.weight", "lm_head.weight"}:
            rows = value.shape[0]
            if rows < target_vocab_size:
                filler = value.mean(dim=0, keepdim=True).expand(target_vocab_size - rows, -1)
                value = torch.cat([value, filler.to(dtype=value.dtype)], dim=0)
            elif rows > target_vocab_size:
                value = value[:target_vocab_size].contiguous()
        state_dict[name] = value

    qwen3_config = asdict(model.config) | {"vocab_size": target_vocab_size}
    save_file(
        state_dict,
        str(checkpoint_dir / "model.safetensors"),
    )
    metadata = {
        "mini_qwen_config": asdict(mini_config) if mini_config is not None else None,
        "qwen3_config": qwen3_config,
        "vocab_size": target_vocab_size,
        "max_position_embeddings": model.config.max_position_embeddings,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "target_model_path": target_model_path,
        "draft_base_model_path": draft_base_model_path,
        "seq_len": seq_len,
    }
    (checkpoint_dir / "draft_model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    if draft_base_model_path:
        hf_config = copy.deepcopy(
            AutoConfig.from_pretrained(draft_base_model_path, trust_remote_code=True)
        )
        hf_config.vocab_size = target_vocab_size
        hf_config.max_position_embeddings = int(model.config.max_position_embeddings)
        hf_config.save_pretrained(checkpoint_dir)
        AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=True,
        ).save_pretrained(checkpoint_dir)
        try:
            GenerationConfig.from_pretrained(
                tokenizer_name_or_path,
                trust_remote_code=True,
            ).save_pretrained(checkpoint_dir)
        except Exception:
            pass
    else:
        (checkpoint_dir / "config.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

    if training_summary is not None:
        (checkpoint_dir / "training_summary.json").write_text(
            json.dumps(training_summary, indent=2) + "\n",
            encoding="utf-8",
        )


def load_draft_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> Qwen3ForCausalLM:
    checkpoint_path = Path(checkpoint_dir)
    metadata_path = checkpoint_path / "draft_model_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = json.loads((checkpoint_path / "config.json").read_text(encoding="utf-8"))
    if metadata.get("qwen3_config") is not None:
        model = Qwen3ForCausalLM(Qwen3Config(**metadata["qwen3_config"]))
    elif metadata.get("mini_qwen_config") is not None:
        mini_config = MiniQwenConfig(**metadata["mini_qwen_config"])
        model = build_draft_model(
            vocab_size=int(metadata["vocab_size"]),
            max_position_embeddings=int(metadata["max_position_embeddings"]),
            config=mini_config,
        )
    else:
        model = Qwen3ForCausalLM(Qwen3Config.from_pretrained(str(checkpoint_path)))
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
    parser.add_argument("--eval-data", default=DEFAULT_EVAL_DATA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--init-model-path", default=DEFAULT_INIT_MODEL_PATH)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--mini-hidden-size", type=int, default=MiniQwenConfig.hidden_size)
    parser.add_argument("--mini-layers", type=int, default=MiniQwenConfig.num_layers)
    parser.add_argument("--mini-attention-heads", type=int, default=MiniQwenConfig.num_attention_heads)
    parser.add_argument("--mini-kv-heads", type=int, default=MiniQwenConfig.num_key_value_heads)
    parser.add_argument("--mini-intermediate-size", type=int, default=MiniQwenConfig.intermediate_size)
    parser.add_argument("--untie-mini-embeddings", action="store_true")
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-draft-len", type=int, default=DEFAULT_EVAL_DRAFT_LEN)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.target_model_path)
    pad_token_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    using_distillation = dataset_uses_completion(args.data)
    mini_config: MiniQwenConfig | None = None

    if args.resume_from:
        model = load_draft_checkpoint(args.resume_from, device="cpu")
    elif args.init_model_path:
        model = Qwen3ForCausalLM.from_pretrained(args.init_model_path, device="cpu", dtype=dtype)
    else:
        mini_config = MiniQwenConfig(
            hidden_size=args.mini_hidden_size,
            num_layers=args.mini_layers,
            num_attention_heads=args.mini_attention_heads,
            num_key_value_heads=args.mini_kv_heads,
            intermediate_size=args.mini_intermediate_size,
            tie_word_embeddings=not args.untie_mini_embeddings,
        )
        model = build_draft_model(
            vocab_size=len(tokenizer),
            max_position_embeddings=max(args.seq_len, 128),
            config=mini_config,
        )

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    if using_distillation:
        examples = build_distillation_examples(args.data, tokenizer=tokenizer, seq_len=args.seq_len)
        if args.limit_samples > 0:
            examples = examples[: args.limit_samples]
        eval_examples = (
            build_distillation_examples(args.eval_data, tokenizer=tokenizer, seq_len=args.seq_len)
            if args.eval_data
            else []
        )
        if args.eval_limit > 0:
            eval_examples = eval_examples[: args.eval_limit]
        total_steps = resolve_total_steps(
            len(examples),
            batch_size=args.batch_size,
            steps=args.steps,
            epochs=args.epochs,
        )

        model = model.to(device=args.device)
        if args.device != "cpu":
            model = model.to(dtype=dtype)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        losses: list[float] = []
        best_eval_score = -1.0
        best_eval_metrics: dict[str, float] = {}
        order = list(range(len(examples)))
        rng = random.Random(args.seed)
        rng.shuffle(order)

        optimizer.zero_grad(set_to_none=True)
        for step in range(total_steps):
            start = (step * args.batch_size) % len(order)
            if start == 0 and step > 0:
                rng.shuffle(order)
            selected_indices = [
                order[(start + offset) % len(order)] for offset in range(args.batch_size)
            ]
            selected = [examples[index] for index in selected_indices]
            batch, loss_mask = collate_distillation_examples(
                selected,
                pad_token_id=pad_token_id,
            )
            batch = batch.to(device=args.device)
            loss_mask = loss_mask.to(device=args.device)
            loss = masked_language_modeling_loss(model, batch, loss_mask)
            (loss / args.grad_accum).backward()
            losses.append(float(loss.item()))
            if (step + 1) % args.grad_accum == 0 or step == total_steps - 1:
                if args.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
                print(
                    json.dumps(
                        {
                            "step": step + 1,
                            "steps": total_steps,
                            "loss": losses[-1],
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )

            should_eval = (
                bool(eval_examples)
                and args.eval_interval > 0
                and ((step + 1) % args.eval_interval == 0 or step == total_steps - 1)
            )
            if should_eval:
                eval_metrics = evaluate_distillation_fit(
                    model,
                    eval_examples,
                    batch_size=max(1, args.eval_batch_size),
                    device=args.device,
                    pad_token_id=pad_token_id,
                    draft_len=args.eval_draft_len,
                )
                print(
                    json.dumps({"step": step + 1} | eval_metrics, ensure_ascii=True),
                    flush=True,
                )
                eval_score = float(eval_metrics["eval_acceptance_proxy"])
                if eval_score >= best_eval_score:
                    best_eval_score = eval_score
                    best_eval_metrics = eval_metrics
                    save_draft_checkpoint(
                        model=model,
                        output_dir=args.output,
                        mini_config=mini_config,
                        tokenizer_name_or_path=args.target_model_path,
                        target_model_path=args.target_model_path,
                        seq_len=args.seq_len,
                        draft_base_model_path=args.init_model_path or None,
                        training_summary={
                            "steps": total_steps,
                            "saved_at_step": step + 1,
                            "epochs": args.epochs,
                            "batch_size": args.batch_size,
                            "grad_accum": args.grad_accum,
                            "lr": args.lr,
                            "weight_decay": args.weight_decay,
                            "max_grad_norm": args.max_grad_norm,
                            "initial_loss": losses[0],
                            "latest_loss": losses[-1],
                            "using_distillation": True,
                            "init_model_path": args.init_model_path,
                            "resume_from": args.resume_from,
                            "data": args.data,
                            "eval_data": args.eval_data,
                            "limit_samples": args.limit_samples,
                            "eval_limit": args.eval_limit,
                        }
                        | eval_metrics,
                    )

        if not eval_examples or best_eval_score < 0.0:
            save_draft_checkpoint(
                model=model,
                output_dir=args.output,
                mini_config=mini_config,
                tokenizer_name_or_path=args.target_model_path,
                target_model_path=args.target_model_path,
                seq_len=args.seq_len,
                draft_base_model_path=args.init_model_path or None,
                training_summary={
                    "steps": total_steps,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "max_grad_norm": args.max_grad_norm,
                    "initial_loss": losses[0],
                    "final_loss": losses[-1],
                    "using_distillation": True,
                    "init_model_path": args.init_model_path,
                    "resume_from": args.resume_from,
                    "data": args.data,
                    "limit_samples": args.limit_samples,
                },
            )
        elif best_eval_metrics:
            summary_path = Path(args.output) / "training_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["best_eval_metrics"] = best_eval_metrics
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    else:
        sequences = build_training_sequences(args.data, tokenizer=tokenizer, seq_len=args.seq_len)
        if args.limit_samples > 0:
            sequences = sequences[: args.limit_samples]
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
            draft_base_model_path=args.init_model_path or None,
            training_summary={
                "steps": total_steps,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "using_distillation": using_distillation,
                "init_model_path": args.init_model_path,
                "data": args.data,
                "limit_samples": args.limit_samples,
            },
        )


if __name__ == "__main__":
    main()

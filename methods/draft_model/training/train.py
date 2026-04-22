from __future__ import annotations

"""
Draft-model training builds a compact Qwen-like LM that shares the target
tokenizer, then saves only the draft model weights and config for speculation.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file, save_file

from common.qwen3 import Qwen3Config, Qwen3ForCausalLM
from common.tokenizer import load_prompts, load_tokenizer


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


def make_batch(sequences: Sequence[torch.Tensor], batch_size: int, step: int) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    selected = [sequences[(step + offset) % len(sequences)] for offset in range(batch_size)]
    max_len = max(sequence.numel() for sequence in selected)
    batch = torch.zeros((batch_size, max_len), dtype=torch.long)
    for row, sequence in enumerate(selected):
        batch[row, : sequence.numel()] = sequence
    return batch


def language_modeling_loss(model: Qwen3ForCausalLM, batch: torch.Tensor) -> torch.Tensor:
    inputs = batch[:, :-1]
    labels = batch[:, 1:]
    logits = model(inputs).logits
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
    )


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


def save_draft_checkpoint(
    model: Qwen3ForCausalLM,
    output_dir: str | Path,
    mini_config: MiniQwenConfig,
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
        "mini_qwen_config": asdict(mini_config),
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
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact draft model.")
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.target_model_path)
    sequences = build_training_sequences(args.data, tokenizer=tokenizer, seq_len=args.seq_len)
    mini_config = MiniQwenConfig()
    model = build_draft_model(
        vocab_size=len(tokenizer),
        max_position_embeddings=max(args.seq_len, 128),
        config=mini_config,
    )

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    losses = train_steps(
        model=model,
        sequences=sequences,
        steps=args.steps,
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
                "steps": args.steps,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

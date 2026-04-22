from __future__ import annotations

"""
PARD adapts a compact draft model to predict K future tokens in one forward by
appending K learned draft-mask tokens after the prefix.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file, save_file

from common.qwen3 import Qwen3ForCausalLM
from common.tokenizer import load_tokenizer
from methods.draft_model.training.train import (
    MiniQwenConfig,
    build_draft_model,
    build_training_sequences,
    parse_dtype,
)


@dataclass(slots=True)
class PARDConfig:
    draft_len: int = 4


def build_pard_model(
    vocab_size: int,
    max_position_embeddings: int,
    mini_config: MiniQwenConfig | None = None,
) -> tuple[Qwen3ForCausalLM, int]:
    mask_token_id = vocab_size
    model = build_draft_model(
        vocab_size=vocab_size + 1,
        max_position_embeddings=max_position_embeddings,
        config=mini_config,
    )
    return model, mask_token_id


def build_pard_training_example(
    sequence: torch.Tensor,
    draft_len: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sequence.numel() <= draft_len:
        raise ValueError("sequence is too short for the requested draft_len")
    prefix = sequence[:-draft_len]
    labels = sequence[-draft_len:]
    input_ids = torch.cat(
        [prefix, torch.full((draft_len,), mask_token_id, dtype=torch.long)],
        dim=0,
    )
    return input_ids, labels


def pard_forward_logits(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    draft_len: int,
) -> torch.Tensor:
    logits = model(input_ids).logits
    return logits[:, -draft_len:, :]


def train_pard(
    model: Qwen3ForCausalLM,
    sequences: Sequence[torch.Tensor],
    *,
    draft_len: int,
    mask_token_id: int,
    steps: int,
    batch_size: int,
    lr: float,
    device: str | torch.device,
) -> list[float]:
    model = model.to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []

    valid_sequences = [sequence for sequence in sequences if sequence.numel() > draft_len]
    if not valid_sequences:
        raise ValueError("no sequences are long enough for PARD training")

    for step in range(steps):
        batch_inputs: list[torch.Tensor] = []
        batch_labels: list[torch.Tensor] = []
        for offset in range(batch_size):
            input_ids, labels = build_pard_training_example(
                valid_sequences[(step + offset) % len(valid_sequences)],
                draft_len=draft_len,
                mask_token_id=mask_token_id,
            )
            batch_inputs.append(input_ids)
            batch_labels.append(labels)
        max_len = max(item.numel() for item in batch_inputs)
        inputs = torch.zeros((batch_size, max_len), dtype=torch.long)
        for row, item in enumerate(batch_inputs):
            inputs[row, : item.numel()] = item
        labels = torch.stack(batch_labels)
        logits = pard_forward_logits(model, inputs.to(device=device), draft_len=draft_len)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.to(device=device).reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses


def save_pard_checkpoint(
    model: Qwen3ForCausalLM,
    checkpoint_dir: str | Path,
    mini_config: MiniQwenConfig,
    target_model_path: str,
    draft_len: int,
    mask_token_id: int,
) -> None:
    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        str(output_dir / "model.safetensors"),
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "mini_qwen_config": asdict(mini_config),
                "target_model_path": target_model_path,
                "draft_len": draft_len,
                "mask_token_id": mask_token_id,
                "vocab_size": model.config.vocab_size,
                "max_position_embeddings": model.config.max_position_embeddings,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_pard_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[Qwen3ForCausalLM, dict[str, object]]:
    checkpoint_path = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_path / "config.json").read_text(encoding="utf-8"))
    mini_config = MiniQwenConfig(**metadata["mini_qwen_config"])
    model = build_draft_model(
        vocab_size=int(metadata["vocab_size"]),
        max_position_embeddings=int(metadata["max_position_embeddings"]),
        config=mini_config,
    )
    model.load_state_dict(load_file(str(checkpoint_path / "model.safetensors")), strict=True)
    model = model.to(device=device)
    return model, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PARD draft model.")
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--draft-len", type=int, default=4)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.target_model_path)
    sequences = build_training_sequences(args.data, tokenizer=tokenizer, seq_len=args.seq_len)
    mini_config = MiniQwenConfig()
    model, mask_token_id = build_pard_model(
        vocab_size=len(tokenizer),
        max_position_embeddings=max(args.seq_len + args.draft_len, 128),
        mini_config=mini_config,
    )
    losses = train_pard(
        model=model,
        sequences=sequences,
        draft_len=args.draft_len,
        mask_token_id=mask_token_id,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    save_pard_checkpoint(
        model=model,
        checkpoint_dir=args.output,
        mini_config=mini_config,
        target_model_path=args.target_model_path,
        draft_len=args.draft_len,
        mask_token_id=mask_token_id,
    )
    (Path(args.output) / "training_summary.json").write_text(
        json.dumps({"initial_loss": losses[0], "final_loss": losses[-1], "steps": args.steps}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

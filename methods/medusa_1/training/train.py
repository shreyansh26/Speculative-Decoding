from __future__ import annotations

"""
Medusa-1 freezes the base model and trains only lightweight future-token heads.
Each head predicts a different future offset from the same target hidden states.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from common.qwen3 import Qwen3ForCausalLM
from common.tokenizer import load_prompts, load_tokenizer
from methods.draft_model.training.train import build_training_sequences, make_batch, parse_dtype


class MedusaHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            residual = nn.Linear(hidden_size, hidden_size)
            nn.init.zeros_(residual.weight)
            nn.init.zeros_(residual.bias)
            layers.append(nn.Sequential(residual, nn.SiLU()))
        self.residual_layers = nn.ModuleList(layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = hidden_states
        for layer in self.residual_layers:
            output = output + layer(output)
        return self.lm_head(output)


class MedusaHeads(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 4,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList(
            [MedusaHead(hidden_size, vocab_size, num_layers=num_layers) for _ in range(num_heads)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(hidden_states) for head in self.heads], dim=0)


@dataclass(slots=True)
class MedusaConfig:
    num_heads: int = 4
    medusa_num_layers: int = 1
    loss_weights: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4)


def freeze_base_model(model: Qwen3ForCausalLM) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def compute_medusa_loss(
    medusa_heads: MedusaHeads,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    loss_weights: Sequence[float],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    logits = medusa_heads(hidden_states)
    losses: list[torch.Tensor] = []
    total_loss = hidden_states.new_tensor(0.0)
    for head_idx in range(logits.shape[0]):
        offset = head_idx + 1
        if input_ids.shape[1] <= offset:
            continue
        head_logits = logits[head_idx, :, :-offset, :]
        labels = input_ids[:, offset:]
        loss = torch.nn.functional.cross_entropy(
            head_logits.reshape(-1, head_logits.shape[-1]),
            labels.reshape(-1),
        )
        weight = loss_weights[head_idx] if head_idx < len(loss_weights) else 1.0
        total_loss = total_loss + (weight * loss)
        losses.append(loss.detach())
    if not losses:
        raise ValueError("input sequence is too short for Medusa offsets")
    return total_loss, losses


def train_medusa_heads(
    target_model: Qwen3ForCausalLM,
    medusa_heads: MedusaHeads,
    sequences: Sequence[torch.Tensor],
    *,
    steps: int,
    batch_size: int,
    lr: float,
    device: str | torch.device,
    loss_weights: Sequence[float],
) -> list[float]:
    target_model = target_model.to(device=device)
    medusa_heads = medusa_heads.to(device=device)
    optimizer = torch.optim.AdamW(medusa_heads.parameters(), lr=lr)
    losses: list[float] = []

    for step in range(steps):
        batch = make_batch(sequences, batch_size=batch_size, step=step).to(device=device)
        with torch.no_grad():
            outputs = target_model(
                batch[:, :-1],
                output_hidden_states=True,
                hidden_state_indices=[target_model.config.num_hidden_layers],
            )
            hidden_states = outputs.hidden_states[target_model.config.num_hidden_layers]
        total_loss, _ = compute_medusa_loss(
            medusa_heads=medusa_heads,
            hidden_states=hidden_states,
            input_ids=batch[:, :-1],
            loss_weights=loss_weights,
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        losses.append(float(total_loss.item()))

    return losses


def save_medusa_checkpoint(
    medusa_heads: MedusaHeads,
    checkpoint_dir: str | Path,
    config: MedusaConfig,
    target_model_path: str,
    hidden_size: int,
    vocab_size: int,
) -> None:
    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.detach().cpu() for name, tensor in medusa_heads.state_dict().items()},
        str(output_dir / "medusa_heads.safetensors"),
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "medusa_config": {
                    "num_heads": config.num_heads,
                    "medusa_num_layers": config.medusa_num_layers,
                    "loss_weights": list(config.loss_weights),
                },
                "target_model_path": target_model_path,
                "hidden_size": hidden_size,
                "vocab_size": vocab_size,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_medusa_checkpoint(checkpoint_dir: str | Path) -> tuple[MedusaHeads, dict[str, object]]:
    checkpoint_path = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_path / "config.json").read_text(encoding="utf-8"))
    medusa_config = MedusaConfig(
        num_heads=int(metadata["medusa_config"]["num_heads"]),
        medusa_num_layers=int(metadata["medusa_config"]["medusa_num_layers"]),
        loss_weights=tuple(metadata["medusa_config"]["loss_weights"]),
    )
    heads = MedusaHeads(
        hidden_size=int(metadata["hidden_size"]),
        vocab_size=int(metadata["vocab_size"]),
        num_heads=medusa_config.num_heads,
        num_layers=medusa_config.medusa_num_layers,
    )
    heads.load_state_dict(load_file(str(checkpoint_path / "medusa_heads.safetensors")), strict=True)
    return heads, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Medusa-1 heads.")
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-medusa-heads", type=int, default=4)
    parser.add_argument("--medusa-num-layers", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.target_model_path)
    sequences = build_training_sequences(args.data, tokenizer=tokenizer, seq_len=args.seq_len)
    target_model = Qwen3ForCausalLM.from_pretrained(args.target_model_path, device=args.device)
    freeze_base_model(target_model)

    medusa_config = MedusaConfig(
        num_heads=args.num_medusa_heads,
        medusa_num_layers=args.medusa_num_layers,
        loss_weights=(1.0, 0.8, 0.6, 0.4)[: args.num_medusa_heads],
    )
    medusa_heads = MedusaHeads(
        hidden_size=target_model.config.hidden_size,
        vocab_size=target_model.config.vocab_size,
        num_heads=medusa_config.num_heads,
        num_layers=medusa_config.medusa_num_layers,
    )

    if args.compile:
        medusa_heads = torch.compile(medusa_heads, mode="reduce-overhead")

    losses = train_medusa_heads(
        target_model=target_model,
        medusa_heads=medusa_heads,
        sequences=sequences,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        loss_weights=medusa_config.loss_weights,
    )
    save_medusa_checkpoint(
        medusa_heads=medusa_heads,
        checkpoint_dir=args.output,
        config=medusa_config,
        target_model_path=args.target_model_path,
        hidden_size=target_model.config.hidden_size,
        vocab_size=target_model.config.vocab_size,
    )
    (Path(args.output) / "training_summary.json").write_text(
        json.dumps(
            {
                "steps": args.steps,
                "batch_size": args.batch_size,
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

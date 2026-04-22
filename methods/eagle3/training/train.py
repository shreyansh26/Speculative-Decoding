from __future__ import annotations

"""
Minimal EAGLE-3 trainer: fuse low/mid/high target hidden states, then predict a
short draft sequence with a lightweight recurrent drafter.
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
from common.tokenizer import load_tokenizer
from methods.draft_model.training.train import build_training_sequences, parse_dtype


@dataclass(slots=True)
class Eagle3Config:
    hidden_size: int
    vocab_size: int
    selected_layers: tuple[int, int, int]
    draft_len: int


class Eagle3Drafter(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.feature_fuser = nn.Linear(3 * hidden_size, hidden_size)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRUCell(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def init_state(self, fused_features: torch.Tensor) -> torch.Tensor:
        return fused_features

    def forward_step(
        self,
        fused_features: torch.Tensor,
        prev_token_ids: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.token_embedding(prev_token_ids)
        state = self.rnn(embedded + fused_features, state)
        logits = self.lm_head(state)
        return logits, state


def freeze_base_model(model: Qwen3ForCausalLM) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def fuse_hidden_states(hidden_state_map: dict[int, torch.Tensor], selected_layers: Sequence[int]) -> torch.Tensor:
    tensors = [hidden_state_map[layer_idx] for layer_idx in selected_layers]
    if len(tensors) != 3:
        raise ValueError("selected_layers must contain exactly three layer indices")
    return torch.cat(tensors, dim=-1)


def run_drafter_training_step(
    drafter: Eagle3Drafter,
    fused_features: torch.Tensor,
    context_last_token: torch.Tensor,
    labels: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    fused = drafter.feature_fuser(fused_features)
    state = drafter.init_state(fused)
    prev_tokens = context_last_token
    for step in range(labels.shape[1]):
        logits, state = drafter.forward_step(fused, prev_tokens, state)
        losses.append(torch.nn.functional.cross_entropy(logits, labels[:, step]))
        if mode == "teacher_forcing":
            prev_tokens = labels[:, step]
        elif mode == "training_time_test":
            prev_tokens = torch.argmax(logits, dim=-1)
        else:
            raise ValueError(f"unsupported mode: {mode}")
    return torch.stack(losses).sum()


def train_eagle3(
    target_model: Qwen3ForCausalLM,
    drafter: Eagle3Drafter,
    sequences: Sequence[torch.Tensor],
    *,
    selected_layers: Sequence[int],
    draft_len: int,
    steps: int,
    batch_size: int,
    lr: float,
    device: str | torch.device,
    mode: str,
) -> list[float]:
    target_model = target_model.to(device=device)
    drafter = drafter.to(device=device)
    optimizer = torch.optim.AdamW(drafter.parameters(), lr=lr)
    valid_sequences = [sequence for sequence in sequences if sequence.numel() > draft_len]
    if not valid_sequences:
        raise ValueError("no sequences are long enough for EAGLE-3 training")
    losses: list[float] = []

    for step in range(steps):
        batch_items = [valid_sequences[(step + idx) % len(valid_sequences)] for idx in range(batch_size)]
        prefixes = [item[:-draft_len] for item in batch_items]
        labels = torch.stack([item[-draft_len:] for item in batch_items]).to(device=device)
        max_len = max(prefix.numel() for prefix in prefixes)
        batch = torch.zeros((batch_size, max_len), dtype=torch.long)
        for row, prefix in enumerate(prefixes):
            batch[row, : prefix.numel()] = prefix
        batch = batch.to(device=device)
        with torch.no_grad():
            outputs = target_model(
                batch,
                output_hidden_states=True,
                hidden_state_indices=selected_layers,
            )
        fused = fuse_hidden_states(outputs.hidden_states, selected_layers)
        fused_last = fused[:, -1, :]
        context_last_token = batch[:, -1]
        loss = run_drafter_training_step(drafter, fused_last, context_last_token, labels, mode)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses


def save_eagle3_checkpoint(
    drafter: Eagle3Drafter,
    checkpoint_dir: str | Path,
    config: Eagle3Config,
    target_model_path: str,
) -> None:
    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.detach().cpu() for name, tensor in drafter.state_dict().items()},
        str(output_dir / "drafter.safetensors"),
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "eagle3_config": asdict(config),
                "target_model_path": target_model_path,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_eagle3_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[Eagle3Drafter, dict[str, object]]:
    checkpoint_path = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_path / "config.json").read_text(encoding="utf-8"))
    config = Eagle3Config(**metadata["eagle3_config"])
    drafter = Eagle3Drafter(hidden_size=config.hidden_size, vocab_size=config.vocab_size)
    drafter.load_state_dict(load_file(str(checkpoint_path / "drafter.safetensors")), strict=True)
    drafter = drafter.to(device=device)
    return drafter, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal EAGLE-3 drafter.")
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
    parser.add_argument("--teacher-forcing-only", action="store_true")
    parser.add_argument("--training-time-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.target_model_path)
    sequences = build_training_sequences(args.data, tokenizer=tokenizer, seq_len=args.seq_len)
    target_model = Qwen3ForCausalLM.from_pretrained(args.target_model_path, device=args.device)
    freeze_base_model(target_model)
    num_layers = target_model.config.num_hidden_layers
    selected_layers = (max(0, num_layers // 4), max(0, num_layers // 2), num_layers)
    drafter = Eagle3Drafter(hidden_size=target_model.config.hidden_size, vocab_size=target_model.config.vocab_size)
    mode = "training_time_test" if args.training_time_test else "teacher_forcing"
    losses = train_eagle3(
        target_model=target_model,
        drafter=drafter,
        sequences=sequences,
        selected_layers=selected_layers,
        draft_len=args.draft_len,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        mode=mode,
    )
    save_eagle3_checkpoint(
        drafter=drafter,
        checkpoint_dir=args.output,
        config=Eagle3Config(
            hidden_size=target_model.config.hidden_size,
            vocab_size=target_model.config.vocab_size,
            selected_layers=selected_layers,
            draft_len=args.draft_len,
        ),
        target_model_path=args.target_model_path,
    )
    (Path(args.output) / "training_summary.json").write_text(
        json.dumps({"initial_loss": losses[0], "final_loss": losses[-1], "mode": mode}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

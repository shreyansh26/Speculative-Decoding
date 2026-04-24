from __future__ import annotations

"""
Medusa-1 trains lightweight future-token heads on top of a frozen target LM.

The k-th Medusa head predicts the token after the target LM's next token:
head 0 predicts offset 2, head 1 predicts offset 3, and so on.  This file is
kept self-contained for the method implementation; shared repo utilities are
only used for tokenizer/model loading and dtype parsing.
"""

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

from common.qwen3 import Qwen3ForCausalLM
from common.tokenizer import load_tokenizer, render_prompt
from methods.draft_model.training.train import parse_dtype


DEFAULT_TARGET_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATA_PATH = "data/ultrachat_3000_trunc1024_qwen25_7b_greedy128_ids.jsonl"
DEFAULT_EVAL_PATH = "data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl"
DEFAULT_OUTPUT_DIR = "checkpoints/medusa_1_qwen25_7b_ultrachat3000"
DEFAULT_SEQ_LEN = 1152
DEFAULT_STEPS = 3000
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 1
DEFAULT_LR = 1e-3
DEFAULT_NUM_HEADS = 4


class MedusaHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int = 1) -> None:
        super().__init__()
        self.residual_layers = nn.ModuleList()
        for _ in range(num_layers):
            residual = nn.Linear(hidden_size, hidden_size)
            nn.init.zeros_(residual.weight)
            nn.init.zeros_(residual.bias)
            self.residual_layers.append(residual)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = hidden_states
        for residual in self.residual_layers:
            output = output + F.silu(residual(output))
        return self.lm_head(output)


class MedusaHeads(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = DEFAULT_NUM_HEADS,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.heads = nn.ModuleList(
            [MedusaHead(hidden_size, vocab_size, num_layers=num_layers) for _ in range(num_heads)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(hidden_states) for head in self.heads], dim=0)

    def initialize_from_lm_head(self, lm_head: nn.Linear) -> None:
        with torch.no_grad():
            source = lm_head.weight.detach()
            for head in self.heads:
                if head.lm_head.weight.shape != source.shape:
                    raise ValueError(
                        f"lm_head shape {tuple(source.shape)} does not match "
                        f"Medusa head shape {tuple(head.lm_head.weight.shape)}"
                    )
                head.lm_head.weight.copy_(source)


@dataclass(slots=True)
class MedusaConfig:
    num_heads: int = DEFAULT_NUM_HEADS
    medusa_num_layers: int = 1
    loss_decay: float = 0.8
    loss_weights: tuple[float, ...] = (1.0, 0.8, 0.64, 0.512)
    train_on_completions_only: bool = True


@dataclass(slots=True)
class MedusaTrainingExample:
    prompt_id: str
    token_ids: torch.Tensor
    loss_mask: torch.Tensor
    prompt_tokens: int
    completion_tokens: int


def freeze_base_model(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def resolve_loss_weights(num_heads: int, loss_decay: float) -> tuple[float, ...]:
    return tuple(float(loss_decay) ** index for index in range(num_heads))


def _ids_from_json_field(sample: dict[str, object], key: str) -> list[int] | None:
    value = sample.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list of token ids")
    return [int(token_id) for token_id in value]


def build_medusa_examples(
    data_path: str | Path,
    tokenizer,
    seq_len: int,
    *,
    completions_only: bool = True,
    limit_samples: int = 0,
) -> list[MedusaTrainingExample]:
    if seq_len < 3:
        raise ValueError("seq_len must be at least 3 for Medusa offset targets")

    examples: list[MedusaTrainingExample] = []
    chunk_len = seq_len
    with Path(data_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            prompt_id = str(sample.get("prompt_id", sample.get("id", f"prompt_{line_number:04d}")))
            prompt_ids = _ids_from_json_field(sample, "prompt_ids")
            if prompt_ids is None:
                rendered = render_prompt(tokenizer, sample)
                prompt_ids = tokenizer.encode(rendered.prompt, add_special_tokens=False)

            completion_ids = _ids_from_json_field(sample, "completion_ids")
            if completion_ids is None and "completion" in sample:
                completion_ids = tokenizer.encode(str(sample["completion"]), add_special_tokens=False)
            if completion_ids is None:
                if completions_only:
                    raise ValueError(
                        f"{data_path}:{line_number} does not include completion targets"
                    )
                completion_ids = []

            token_ids = prompt_ids + completion_ids
            if len(token_ids) < 3:
                continue

            if completions_only:
                loss_mask_values = ([0.0] * len(prompt_ids)) + ([1.0] * len(completion_ids))
            else:
                loss_mask_values = [1.0] * len(token_ids)

            if len(token_ids) > chunk_len:
                if completion_ids:
                    token_ids = token_ids[-chunk_len:]
                    loss_mask_values = loss_mask_values[-chunk_len:]
                else:
                    token_ids = token_ids[:chunk_len]
                    loss_mask_values = loss_mask_values[:chunk_len]

            if len(token_ids) < 3 or sum(loss_mask_values[2:]) <= 0.0:
                continue

            prompt_tokens = min(len(prompt_ids), len(token_ids))
            completion_tokens = max(0, len(token_ids) - prompt_tokens)
            examples.append(
                MedusaTrainingExample(
                    prompt_id=prompt_id,
                    token_ids=torch.tensor(token_ids, dtype=torch.long),
                    loss_mask=torch.tensor(loss_mask_values, dtype=torch.float32),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
            if limit_samples > 0 and len(examples) >= limit_samples:
                break

    if not examples:
        raise ValueError("dataset did not produce any Medusa training examples")
    return examples


def collate_medusa_examples(
    examples: Sequence[MedusaTrainingExample],
    *,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("cannot collate an empty example list")
    max_len = max(example.token_ids.numel() for example in examples)
    batch = torch.full((len(examples), max_len), int(pad_token_id), dtype=torch.long)
    loss_mask = torch.zeros((len(examples), max_len), dtype=torch.float32)
    for row, example in enumerate(examples):
        length = example.token_ids.numel()
        batch[row, :length] = example.token_ids
        loss_mask[row, :length] = example.loss_mask
    return batch, loss_mask


def make_medusa_batch(
    examples: Sequence[MedusaTrainingExample],
    batch_size: int,
    step: int,
    order: Sequence[int] | None = None,
    *,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source_order = list(range(len(examples))) if order is None else list(order)
    start = (step * batch_size) % len(source_order)
    selected = [examples[source_order[(start + offset) % len(source_order)]] for offset in range(batch_size)]
    return collate_medusa_examples(selected, pad_token_id=pad_token_id)


def compute_medusa_loss(
    medusa_heads: MedusaHeads,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    loss_weights: Sequence[float],
    loss_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Compute offset-2+ Medusa loss.

    `hidden_states[:, t]` is trained to predict `input_ids[:, t + head + 2]`.
    If `loss_mask` is provided, a label contributes only when the target token
    position is non-zero in the mask.
    """

    logits = medusa_heads(hidden_states)
    losses: list[torch.Tensor] = []
    total_loss = hidden_states.new_tensor(0.0)

    for head_idx in range(logits.shape[0]):
        offset = head_idx + 2
        if input_ids.shape[1] <= offset:
            continue
        head_logits = logits[head_idx, :, :-offset, :]
        labels = input_ids[:, offset:]
        if loss_mask is None:
            loss = F.cross_entropy(
                head_logits.reshape(-1, head_logits.shape[-1]),
                labels.reshape(-1),
            )
        else:
            label_mask = loss_mask[:, offset:].to(device=head_logits.device, dtype=head_logits.dtype)
            token_losses = F.cross_entropy(
                head_logits.reshape(-1, head_logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).view_as(labels)
            total_weight = label_mask.sum()
            if float(total_weight.item()) <= 0.0:
                continue
            loss = (token_losses * label_mask).sum() / total_weight

        weight = loss_weights[head_idx] if head_idx < len(loss_weights) else 1.0
        total_loss = total_loss + (float(weight) * loss)
        losses.append(loss.detach())

    if not losses:
        raise ValueError("input sequence is too short or fully masked for Medusa offsets")
    return total_loss, losses


def evaluate_medusa_heads(
    target_model: Qwen3ForCausalLM,
    medusa_heads: MedusaHeads,
    examples: Sequence[MedusaTrainingExample],
    *,
    batch_size: int,
    device: str | torch.device,
    loss_weights: Sequence[float],
    pad_token_id: int,
    max_batches: int = 8,
) -> dict[str, float]:
    target_model.eval()
    medusa_heads.eval()
    losses: list[float] = []
    correct = [0 for _ in range(medusa_heads.num_heads)]
    totals = [0 for _ in range(medusa_heads.num_heads)]
    with torch.inference_mode():
        batches = min(max_batches, max(1, (len(examples) + batch_size - 1) // batch_size))
        for step in range(batches):
            batch, mask = make_medusa_batch(
                examples,
                batch_size=batch_size,
                step=step,
                pad_token_id=pad_token_id,
            )
            batch = batch.to(device=device)
            mask = mask.to(device=device)
            outputs = target_model(
                batch,
                output_hidden_states=True,
                hidden_state_indices=[target_model.config.num_hidden_layers],
            )
            hidden = outputs.hidden_states[target_model.config.num_hidden_layers]
            total_loss, _ = compute_medusa_loss(
                medusa_heads,
                hidden,
                batch,
                loss_weights,
                loss_mask=mask,
            )
            losses.append(float(total_loss.item()))
            logits = medusa_heads(hidden)
            for head_idx in range(logits.shape[0]):
                offset = head_idx + 2
                if batch.shape[1] <= offset:
                    continue
                valid = mask[:, offset:] > 0
                if not bool(valid.any().item()):
                    continue
                predictions = torch.argmax(logits[head_idx, :, :-offset, :], dim=-1)
                labels = batch[:, offset:]
                correct[head_idx] += int(((predictions == labels) & valid).sum().item())
                totals[head_idx] += int(valid.sum().item())
    medusa_heads.train()
    metrics = {"loss": sum(losses) / len(losses) if losses else 0.0}
    for head_idx, total in enumerate(totals):
        metrics[f"head{head_idx}_top1"] = (correct[head_idx] / total) if total else 0.0
    return metrics


def train_medusa_heads(
    target_model: Qwen3ForCausalLM,
    medusa_heads: MedusaHeads,
    examples: Sequence[MedusaTrainingExample],
    *,
    steps: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    device: str | torch.device,
    dtype: torch.dtype,
    loss_weights: Sequence[float],
    pad_token_id: int,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    seed: int = 0,
) -> list[float]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if grad_accum <= 0:
        raise ValueError("grad_accum must be positive")

    target_model = target_model.to(device=device)
    medusa_heads = medusa_heads.to(device=device)
    if torch.device(device).type != "cpu":
        target_model = target_model.to(dtype=dtype)
        medusa_heads = medusa_heads.to(dtype=dtype)
    target_model.eval()
    medusa_heads.train()

    optimizer = torch.optim.AdamW(medusa_heads.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    order = list(range(len(examples)))
    rng = random.Random(seed)
    rng.shuffle(order)

    optimizer.zero_grad(set_to_none=True)
    for step in range(steps):
        start = (step * batch_size) % len(order)
        if start == 0 and step > 0:
            rng.shuffle(order)
        batch, mask = make_medusa_batch(
            examples,
            batch_size=batch_size,
            step=step,
            order=order,
            pad_token_id=pad_token_id,
        )
        batch = batch.to(device=device)
        mask = mask.to(device=device)

        with torch.no_grad():
            outputs = target_model(
                batch,
                output_hidden_states=True,
                hidden_state_indices=[target_model.config.num_hidden_layers],
            )
            hidden_states = outputs.hidden_states[target_model.config.num_hidden_layers]

        total_loss, _ = compute_medusa_loss(
            medusa_heads=medusa_heads,
            hidden_states=hidden_states,
            input_ids=batch,
            loss_weights=loss_weights,
            loss_mask=mask,
        )
        (total_loss / grad_accum).backward()
        losses.append(float(total_loss.item()))
        if (step + 1) % grad_accum == 0 or step == steps - 1:
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(medusa_heads.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return losses


def save_medusa_checkpoint(
    medusa_heads: MedusaHeads,
    checkpoint_dir: str | Path,
    config: MedusaConfig,
    target_model_path: str,
    hidden_size: int,
    vocab_size: int,
    training_summary: dict[str, object] | None = None,
) -> None:
    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.detach().cpu() for name, tensor in medusa_heads.state_dict().items()},
        str(output_dir / "medusa_heads.safetensors"),
    )
    metadata = {
        "medusa_config": asdict(config),
        "target_model_path": target_model_path,
        "hidden_size": int(hidden_size),
        "vocab_size": int(vocab_size),
    }
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if training_summary is not None:
        (output_dir / "training_summary.json").write_text(
            json.dumps(training_summary, indent=2) + "\n",
            encoding="utf-8",
        )


def load_medusa_checkpoint(checkpoint_dir: str | Path) -> tuple[MedusaHeads, dict[str, object]]:
    checkpoint_path = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_path / "config.json").read_text(encoding="utf-8"))
    raw_config = metadata["medusa_config"]
    medusa_config = MedusaConfig(
        num_heads=int(raw_config["num_heads"]),
        medusa_num_layers=int(raw_config["medusa_num_layers"]),
        loss_decay=float(raw_config.get("loss_decay", 0.8)),
        loss_weights=tuple(float(value) for value in raw_config["loss_weights"]),
        train_on_completions_only=bool(raw_config.get("train_on_completions_only", True)),
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
    parser.add_argument("--target-model-path", default=DEFAULT_TARGET_MODEL_PATH)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--eval-data", default=DEFAULT_EVAL_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-medusa-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--medusa-num-layers", type=int, default=1)
    parser.add_argument("--loss-decay", type=float, default=0.8)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-on-full-sequence", action="store_true")
    parser.add_argument("--no-lm-head-init", action="store_true")
    parser.add_argument("--compile-heads", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.target_model_path)
    pad_token_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0)
    completions_only = not args.train_on_full_sequence
    examples = build_medusa_examples(
        args.data,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        completions_only=completions_only,
        limit_samples=args.limit_samples,
    )
    eval_examples: list[MedusaTrainingExample] = []
    if args.eval_data:
        eval_examples = build_medusa_examples(
            args.eval_data,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            completions_only=completions_only,
            limit_samples=0,
        )

    target_model = Qwen3ForCausalLM.from_pretrained(args.target_model_path, device=args.device, dtype=dtype)
    freeze_base_model(target_model)

    medusa_config = MedusaConfig(
        num_heads=args.num_medusa_heads,
        medusa_num_layers=args.medusa_num_layers,
        loss_decay=args.loss_decay,
        loss_weights=resolve_loss_weights(args.num_medusa_heads, args.loss_decay),
        train_on_completions_only=completions_only,
    )
    medusa_heads = MedusaHeads(
        hidden_size=target_model.config.hidden_size,
        vocab_size=target_model.config.vocab_size,
        num_heads=medusa_config.num_heads,
        num_layers=medusa_config.medusa_num_layers,
    )
    if not args.no_lm_head_init:
        medusa_heads.initialize_from_lm_head(target_model.lm_head)
    if args.compile_heads:
        medusa_heads = torch.compile(medusa_heads, mode="reduce-overhead")

    losses = train_medusa_heads(
        target_model=target_model,
        medusa_heads=medusa_heads,
        examples=examples,
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        device=args.device,
        dtype=dtype,
        loss_weights=medusa_config.loss_weights,
        pad_token_id=pad_token_id,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )
    eval_metrics = (
        evaluate_medusa_heads(
            target_model=target_model,
            medusa_heads=medusa_heads,
            examples=eval_examples,
            batch_size=args.batch_size,
            device=args.device,
            loss_weights=medusa_config.loss_weights,
            pad_token_id=pad_token_id,
            max_batches=args.eval_batches,
        )
        if eval_examples
        else {}
    )
    save_medusa_checkpoint(
        medusa_heads=medusa_heads,
        checkpoint_dir=args.output,
        config=medusa_config,
        target_model_path=args.target_model_path,
        hidden_size=target_model.config.hidden_size,
        vocab_size=target_model.config.vocab_size,
        training_summary={
            "steps": args.steps,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "data": args.data,
            "eval_data": args.eval_data,
            "limit_samples": args.limit_samples,
            "seq_len": args.seq_len,
            "pad_token_id": pad_token_id,
            "lm_head_init": not args.no_lm_head_init,
            "eval_metrics": eval_metrics,
        },
    )


if __name__ == "__main__":
    main()

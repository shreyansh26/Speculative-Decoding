from __future__ import annotations

"""
Train a PARD parallel draft model.

This file intentionally keeps the full implementation local.  It adapts a normal
autoregressive draft model into a mask-token parallel drafter using the PARD
training layout from the paper/reference implementation, then exports a standard
Hugging Face checkpoint that vLLM can load with parallel_drafting=True.
"""

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from common.tokenizer import load_tokenizer, render_prompt
from methods.draft_model.training.train import parse_dtype


DEFAULT_TARGET_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DRAFT_BASE_MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TRAIN_DATA = "data/ultrachat_3000_trunc1024_qwen25_7b_greedy128_ids.jsonl"
DEFAULT_EVAL_DATA = "data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl"
DEFAULT_OUTPUT = "checkpoints/parallel_draft_models_qwen25_05b_ultrachat3000"
DEFAULT_PARD_TOKEN_ID = 151665
DEFAULT_DRAFT_LEN = 8
DEFAULT_SEQ_LEN = 512
IGNORE_INDEX = -100


@dataclass(slots=True)
class PARDExample:
    prompt_id: str
    prompt_ids: list[int]
    completion_ids: list[int]

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_ids + self.completion_ids

    @property
    def loss_mask(self) -> list[int]:
        return ([0] * len(self.prompt_ids)) + ([1] * len(self.completion_ids))


@dataclass(slots=True)
class PARDBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor


def load_pard_examples(
    data_path: str | Path,
    tokenizer,
    *,
    seq_len: int,
    limit: int = 0,
) -> list[PARDExample]:
    examples: list[PARDExample] = []
    with Path(data_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            prompt_id = str(sample.get("prompt_id", sample.get("id", f"prompt_{line_number:04d}")))
            if "prompt_ids" in sample:
                prompt_ids = [int(token_id) for token_id in sample["prompt_ids"]]
            else:
                rendered = render_prompt(tokenizer, sample)
                prompt_ids = tokenizer.encode(rendered.prompt, add_special_tokens=False)

            if "completion_ids" in sample:
                completion_ids = [int(token_id) for token_id in sample["completion_ids"]]
            elif "completion" in sample:
                completion_ids = tokenizer.encode(str(sample["completion"]), add_special_tokens=False)
            else:
                raise ValueError(f"{data_path}:{line_number} is missing completion_ids/completion")

            if not prompt_ids or not completion_ids:
                continue

            token_ids = prompt_ids + completion_ids
            loss_mask = ([0] * len(prompt_ids)) + ([1] * len(completion_ids))
            if len(token_ids) > seq_len:
                token_ids = token_ids[-seq_len:]
                loss_mask = loss_mask[-seq_len:]
                prompt_count = 0
                trimmed_prompt_ids: list[int] = []
                trimmed_completion_ids: list[int] = []
                original_prompt_len = len(prompt_ids)
                original_start = len(prompt_ids) + len(completion_ids) - len(token_ids)
                for offset, token_id in enumerate(token_ids):
                    original_index = original_start + offset
                    if original_index < original_prompt_len:
                        prompt_count += 1
                        trimmed_prompt_ids.append(token_id)
                    else:
                        trimmed_completion_ids.append(token_id)
                prompt_ids = trimmed_prompt_ids
                completion_ids = trimmed_completion_ids
            if not completion_ids:
                continue
            examples.append(
                PARDExample(
                    prompt_id=prompt_id,
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                )
            )
            if limit > 0 and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"{data_path} did not produce any PARD examples")
    return examples


def _collate_base_examples(
    examples: Sequence[PARDExample],
    *,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    max_len = max(len(example.token_ids) for example in examples)
    input_ids = torch.full((len(examples), max_len), int(pad_token_id), dtype=torch.long)
    labels = torch.full((len(examples), max_len), IGNORE_INDEX, dtype=torch.long)
    for row, example in enumerate(examples):
        token_ids = example.token_ids
        loss_mask = example.loss_mask
        input_ids[row, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
        row_labels = torch.tensor(
            [
                int(token_id) if int(mask) else IGNORE_INDEX
                for token_id, mask in zip(token_ids, loss_mask, strict=True)
            ],
            dtype=torch.long,
        )
        labels[row, : len(token_ids)] = row_labels
    return input_ids, labels


def build_pard_attention_mask(seq_len: int, draft_len: int) -> torch.Tensor:
    """Return the 4D PARD training mask used by the reference implementation."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if draft_len <= 0:
        raise ValueError("draft_len must be positive")

    total_len = seq_len * draft_len
    min_value = torch.finfo(torch.float32).min
    mask = torch.full((total_len, total_len), min_value, dtype=torch.float32)
    indices = torch.arange(total_len)
    allow = indices == indices.view(total_len, 1)
    for draft_index in range(draft_len):
        allow |= indices == (indices - seq_len * draft_index - draft_index).view(total_len, 1)
        allow |= (
            (indices < (indices - draft_index * seq_len - (draft_index - 1)).view(total_len, 1))
            & (indices < (draft_index + 1) * seq_len).view(-1, 1)
        )
    mask.masked_fill_(allow, 0.0)
    return mask[None, None, :, :]


def build_pard_batch(
    examples: Sequence[PARDExample],
    *,
    draft_len: int,
    pard_token_id: int,
    pad_token_id: int,
    cod_ratio: float = 1.0,
    cod_min_ratio: float = 0.0,
    generator: torch.Generator | None = None,
) -> PARDBatch:
    base_input_ids, base_labels = _collate_base_examples(examples, pad_token_id=pad_token_id)
    batch_size, seq_len = base_input_ids.shape
    if draft_len <= 0:
        raise ValueError("draft_len must be positive")

    mask_blocks = [
        torch.full_like(base_input_ids, int(pard_token_id))
        for _ in range(max(draft_len - 1, 0))
    ]
    input_ids = torch.cat([base_input_ids] + mask_blocks, dim=1) if mask_blocks else base_input_ids
    position_ids = torch.arange(seq_len, dtype=torch.long).repeat(draft_len).unsqueeze(0).repeat(batch_size, 1)

    label_blocks: list[torch.Tensor] = []
    for draft_index in range(draft_len):
        if draft_index == 0:
            label_blocks.append(base_labels)
        else:
            prefix = torch.full((batch_size, draft_index), IGNORE_INDEX, dtype=torch.long)
            label_blocks.append(torch.cat([prefix, base_labels[:, draft_index:]], dim=1))
    labels = torch.cat(label_blocks, dim=1)
    attention_mask = build_pard_attention_mask(seq_len, draft_len).repeat(batch_size, 1, 1, 1)

    if cod_ratio != 1.0 and draft_len > 1:
        index_mask = torch.zeros((draft_len, seq_len), dtype=torch.bool)
        index_mask[0, :] = True
        prev_indices = torch.arange(seq_len)
        for draft_index in range(1, draft_len):
            keep_ratio = max(cod_ratio**draft_index, cod_min_ratio)
            keep_count = min(len(prev_indices), max(0, int(seq_len * keep_ratio)))
            if keep_count <= 0:
                break
            order = torch.randperm(len(prev_indices), generator=generator)
            selected = prev_indices[order[:keep_count]]
            index_mask[draft_index, selected] = True
            prev_indices = (selected + 1) % seq_len

        indices = index_mask.reshape(-1).nonzero(as_tuple=True)[0]
        input_ids = input_ids[:, indices].contiguous()
        position_ids = position_ids[:, indices].contiguous()
        labels = torch.roll(torch.roll(labels, shifts=-1, dims=1)[:, indices], shifts=1, dims=1).contiguous()
        attention_mask = attention_mask[:, :, indices, :][:, :, :, indices].contiguous()

    return PARDBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels,
    )


def build_pard_training_example(
    sequence: torch.Tensor,
    draft_len: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility helper used by the toy tests."""
    if sequence.numel() <= draft_len:
        raise ValueError("sequence is too short for the requested draft_len")
    prefix = sequence[:-draft_len]
    labels = sequence[-draft_len:]
    input_ids = torch.cat(
        [prefix, torch.full((draft_len,), int(mask_token_id), dtype=torch.long)],
        dim=0,
    )
    return input_ids, labels


def _select_batch(
    examples: Sequence[PARDExample],
    order: list[int],
    *,
    step: int,
    batch_size: int,
    rng: random.Random,
) -> list[PARDExample]:
    start = (step * batch_size) % len(order)
    if start == 0 and step > 0:
        rng.shuffle(order)
    return [examples[order[(start + offset) % len(order)]] for offset in range(batch_size)]


@torch.no_grad()
def evaluate_pard_acceptance(
    model: torch.nn.Module,
    examples: Sequence[PARDExample],
    *,
    draft_len: int,
    pard_token_id: int,
    device: str | torch.device,
    limit: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    selected = list(examples[:limit]) if limit > 0 else list(examples)
    total_proposed = 0
    total_accepted = 0
    total_steps = 0
    total_first_token_matches = 0
    total_first_token_steps = 0

    for example in selected:
        generated_prefix = list(example.prompt_ids)
        completion = list(example.completion_ids)
        for offset in range(0, len(completion), draft_len):
            gold = completion[offset : offset + draft_len]
            requested = len(gold)
            input_ids = generated_prefix + ([pard_token_id] * max(requested - 1, 0))
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            logits = model(input_ids=input_tensor, use_cache=False, return_dict=True).logits[0, -requested:]
            predictions = torch.argmax(logits, dim=-1).tolist()
            accepted = 0
            for predicted, gold_token in zip(predictions, gold, strict=True):
                if int(predicted) != int(gold_token):
                    break
                accepted += 1
            total_steps += 1
            total_proposed += requested
            total_accepted += accepted
            total_first_token_steps += 1
            total_first_token_matches += int(int(predictions[0]) == int(gold[0]))
            generated_prefix.extend(gold)

    if was_training:
        model.train()
    return {
        "eval_acceptance_proxy": (total_accepted / total_proposed) if total_proposed else 0.0,
        "eval_first_token_match": (
            total_first_token_matches / total_first_token_steps if total_first_token_steps else 0.0
        ),
        "eval_mean_accepted_tokens_per_step_proxy": (
            total_accepted / total_steps if total_steps else 0.0
        ),
        "eval_steps": float(total_steps),
        "eval_proposed_tokens": float(total_proposed),
    }


def save_pard_checkpoint(
    model: torch.nn.Module,
    output_dir: str | Path,
    *,
    tokenizer_name_or_path: str,
    target_model_path: str,
    draft_base_model_path: str,
    draft_len: int,
    pard_token_id: int,
    training_summary: dict[str, object] | None = None,
) -> None:
    model = getattr(model, "_orig_mod", model)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.config.pard_token = int(pard_token_id)
    model.config.spd_type = "pard"
    model.config.save_pretrained(output_path)
    model.save_pretrained(output_path, safe_serialization=True)
    AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True).save_pretrained(output_path)
    try:
        GenerationConfig.from_pretrained(target_model_path, trust_remote_code=True).save_pretrained(output_path)
    except Exception:
        pass

    metadata = {
        "method": "parallel_draft_models",
        "target_model_path": target_model_path,
        "draft_base_model_path": draft_base_model_path,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "draft_len": draft_len,
        "pard_token_id": int(pard_token_id),
        "vocab_size": int(model.config.vocab_size),
    }
    (output_path / "pard_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    if training_summary is not None:
        (output_path / "training_summary.json").write_text(
            json.dumps(training_summary, indent=2) + "\n",
            encoding="utf-8",
        )


def load_pard_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[torch.nn.Module, dict[str, object]]:
    checkpoint_path = Path(checkpoint_dir)
    if (checkpoint_path / "config.json").exists() and "model_type" in json.loads(
        (checkpoint_path / "config.json").read_text(encoding="utf-8")
    ):
        load_kwargs: dict[str, object] = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if dtype is not None:
            load_kwargs["dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            **load_kwargs,
        ).to(device=device)
        metadata_path = checkpoint_path / "pard_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        return model, metadata

    # Backward compatibility for the previous local mini-Qwen checkpoint format.
    from methods.draft_model.training.train import MiniQwenConfig, build_draft_model

    metadata = json.loads((checkpoint_path / "config.json").read_text(encoding="utf-8"))
    mini_config = MiniQwenConfig(**metadata["mini_qwen_config"])
    model = build_draft_model(
        vocab_size=int(metadata["vocab_size"]),
        max_position_embeddings=int(metadata["max_position_embeddings"]),
        config=mini_config,
    )
    model.load_state_dict(load_file(str(checkpoint_path / "model.safetensors")), strict=True)
    if dtype is not None and str(device) != "cpu":
        model = model.to(dtype=dtype)
    return model.to(device=device), metadata


def train(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dtype = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.target_model_path)
    pad_token_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

    train_examples = load_pard_examples(args.data, tokenizer, seq_len=args.seq_len, limit=args.train_limit)
    eval_examples = load_pard_examples(args.eval_data, tokenizer, seq_len=args.seq_len, limit=args.eval_limit)

    target_config = AutoConfig.from_pretrained(args.target_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.draft_base_model_path,
        trust_remote_code=True,
        dtype=dtype if str(args.device) != "cpu" else torch.float32,
        attn_implementation="eager",
    )
    model.resize_token_embeddings(int(target_config.vocab_size))
    model.config.vocab_size = int(target_config.vocab_size)
    model.config.pad_token_id = pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pard_token = int(args.pard_token_id)
    model.config.spd_type = "pard"
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device=args.device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    order = list(range(len(train_examples)))
    rng = random.Random(args.seed)
    rng.shuffle(order)
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(args.seed)

    best_eval_score = -math.inf
    best_eval_metrics: dict[str, float] = {}
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)

    for step in range(args.steps):
        selected = _select_batch(
            train_examples,
            order,
            step=step,
            batch_size=args.batch_size,
            rng=rng,
        )
        batch = build_pard_batch(
            selected,
            draft_len=args.draft_len,
            pard_token_id=args.pard_token_id,
            pad_token_id=pad_token_id,
            cod_ratio=args.cod_ratio,
            cod_min_ratio=args.cod_min_ratio,
            generator=torch_generator,
        )
        batch = PARDBatch(
            input_ids=batch.input_ids.to(device=args.device),
            attention_mask=batch.attention_mask.to(device=args.device),
            position_ids=batch.position_ids.to(device=args.device),
            labels=batch.labels.to(device=args.device),
        )
        output = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
            labels=batch.labels,
            use_cache=False,
            return_dict=True,
        )
        loss = output.loss
        (loss / args.grad_accum).backward()
        losses.append(float(loss.detach().cpu().item()))

        if (step + 1) % args.grad_accum == 0 or step == args.steps - 1:
            if args.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        should_log = (step + 1) % args.log_interval == 0 or step == 0 or step == args.steps - 1
        should_eval = (step + 1) % args.eval_interval == 0 or step == args.steps - 1
        if should_log:
            print(json.dumps({"step": step + 1, "loss": losses[-1]}, ensure_ascii=True), flush=True)
        if should_eval:
            eval_metrics = evaluate_pard_acceptance(
                model,
                eval_examples,
                draft_len=args.draft_len,
                pard_token_id=args.pard_token_id,
                device=args.device,
                limit=args.eval_limit,
            )
            print(json.dumps({"step": step + 1} | eval_metrics, ensure_ascii=True), flush=True)
            eval_score = float(eval_metrics["eval_acceptance_proxy"])
            if eval_score >= best_eval_score:
                best_eval_score = eval_score
                best_eval_metrics = eval_metrics
                save_pard_checkpoint(
                    model,
                    args.output,
                    tokenizer_name_or_path=args.target_model_path,
                    target_model_path=args.target_model_path,
                    draft_base_model_path=args.draft_base_model_path,
                    draft_len=args.draft_len,
                    pard_token_id=args.pard_token_id,
                    training_summary={
                        "step": step + 1,
                        "loss": losses[-1],
                        "best_eval_metrics": best_eval_metrics,
                        "train_examples": len(train_examples),
                        "eval_examples": len(eval_examples),
                        "seq_len": args.seq_len,
                        "draft_len": args.draft_len,
                        "cod_ratio": args.cod_ratio,
                        "cod_min_ratio": args.cod_min_ratio,
                    },
                )

    summary = {
        "steps": args.steps,
        "initial_loss": losses[0] if losses else 0.0,
        "final_loss": losses[-1] if losses else 0.0,
        "best_eval_metrics": best_eval_metrics,
        "output": args.output,
    }
    summary_path = Path(args.output) / "training_summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = existing | summary
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PARD parallel draft model.")
    parser.add_argument("--target-model-path", default=DEFAULT_TARGET_MODEL_PATH)
    parser.add_argument("--draft-base-model-path", default=DEFAULT_DRAFT_BASE_MODEL_PATH)
    parser.add_argument("--data", default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--eval-data", default=DEFAULT_EVAL_DATA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--draft-len", type=int, default=DEFAULT_DRAFT_LEN)
    parser.add_argument("--pard-token-id", type=int, default=DEFAULT_PARD_TOKEN_ID)
    parser.add_argument("--cod-ratio", type=float, default=0.7)
    parser.add_argument("--cod-min-ratio", type=float, default=0.2)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train(args)
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

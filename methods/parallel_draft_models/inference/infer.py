"""
PARD speculative decoding.

The cached path follows the PARD reference loop: the draft model receives the
unprocessed real tokens plus K-1 mask tokens in one forward and returns K draft
predictions; the target model verifies those K tokens in one forward.
"""

import argparse
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, StaticCache

from common.metrics import SpecDecodeStats, summarize_jsonl, write_jsonl_record
from common.sampling import autoregressive_generate
from common.tokenizer import load_tokenizer, render_prompt
from common.verification import greedy_verify
from methods.draft_model.training.train import parse_dtype
from methods.parallel_draft_models.training.train import (
    DEFAULT_OUTPUT as DEFAULT_CHECKPOINT_PATH,
    DEFAULT_PARD_TOKEN_ID,
    DEFAULT_TARGET_MODEL_PATH,
    load_pard_checkpoint,
)


DEFAULT_PROMPTS_PATH = "data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_INFERENCE_DRAFT_LEN = 5
DEFAULT_WARMUP_PROMPTS = 1


@dataclass(slots=True)
class PromptRecord:
    prompt_id: str
    prompt_ids: list[int]


def load_prompt_records(prompts_path: str | Path, tokenizer) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with Path(prompts_path).open("r", encoding="utf-8") as handle:
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
            records.append(PromptRecord(prompt_id=prompt_id, prompt_ids=prompt_ids))
    if not records:
        raise ValueError(f"{prompts_path} did not contain any prompts")
    return records


def _model_device(model: torch.nn.Module) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def _model_dtype(model: torch.nn.Module) -> torch.dtype:
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        return dtype
    return next(model.parameters()).dtype


def _supports_hf_cache(model: torch.nn.Module) -> bool:
    if not hasattr(model, "config") or not hasattr(model, "forward"):
        return False
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False
    return "past_key_values" in parameters and "cache_position" in parameters


def propose_parallel_draft_tokens(
    draft_model: torch.nn.Module,
    prefix_ids: Sequence[int],
    draft_len: int,
    mask_token_id: int,
) -> list[int]:
    """Slow no-cache helper used by tests and fallback execution."""
    if draft_len <= 0:
        return []
    try:
        device = next(draft_model.parameters()).device
    except StopIteration:
        device = torch.device(getattr(draft_model, "device", "cpu"))
    input_ids = torch.tensor(
        [list(prefix_ids) + ([int(mask_token_id)] * max(draft_len - 1, 0))],
        dtype=torch.long,
        device=device,
    )
    logits = draft_model(input_ids).logits[0, -draft_len:, :]
    return [int(torch.argmax(logits[index], dim=-1).item()) for index in range(draft_len)]


def _make_static_cache(model: torch.nn.Module, max_cache_len: int) -> StaticCache:
    return StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=_model_device(model),
        dtype=_model_dtype(model),
    )


def _hf_forward_with_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cache: StaticCache,
    cache_length: int,
):
    cache_position = torch.arange(
        cache_length,
        cache_length + input_ids.shape[1],
        dtype=torch.long,
        device=input_ids.device,
    )
    return model(
        input_ids=input_ids,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    )


@torch.inference_mode()
def _hf_greedy_generate(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    eos_token_id: int | None,
) -> list[int]:
    device = _model_device(model)
    max_cache_len = len(prompt_ids) + max_new_tokens + 8
    cache = _make_static_cache(model, max_cache_len=max_cache_len)
    prompt_tensor = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    output = _hf_forward_with_cache(model, prompt_tensor, cache, 0)
    cache_length = prompt_tensor.shape[1]
    next_logits = output.logits[0, -1]
    generated: list[int] = []
    for _ in range(max_new_tokens):
        next_token = int(torch.argmax(next_logits).item())
        generated.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
        token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        output = _hf_forward_with_cache(model, token_tensor, cache, cache_length)
        cache_length += 1
        next_logits = output.logits[0, -1]
    return generated


@torch.inference_mode()
def _run_hf_pard_speculative_decode(
    target_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    draft_len: int,
    mask_token_id: int,
    eos_token_id: int | None,
) -> tuple[list[int], dict[str, int]]:
    if draft_len <= 0:
        raise ValueError("draft_len must be positive")
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")

    target_device = _model_device(target_model)
    draft_device = _model_device(draft_model)
    max_cache_len = len(prompt_ids) + max_new_tokens + draft_len + 16
    target_cache = _make_static_cache(target_model, max_cache_len=max_cache_len)
    draft_cache = _make_static_cache(draft_model, max_cache_len=max_cache_len)
    target_cache_length = 0
    draft_cache_length = 0
    target_pending = torch.tensor([list(prompt_ids)], dtype=torch.long, device=target_device)
    draft_pending = torch.tensor([list(prompt_ids)], dtype=torch.long, device=draft_device)
    generated: list[int] = []
    counters = {
        "speculation_steps": 0,
        "target_forwards": 0,
        "draft_forwards": 0,
        "proposed_draft_tokens": 0,
        "accepted_draft_tokens": 0,
    }

    while len(generated) < max_new_tokens:
        requested = min(draft_len, max_new_tokens - len(generated))
        mask_count = max(requested - 1, 0)
        if mask_count:
            mask_ids = torch.full(
                (1, mask_count),
                int(mask_token_id),
                dtype=torch.long,
                device=draft_device,
            )
            draft_input = torch.cat([draft_pending, mask_ids], dim=1)
        else:
            draft_input = draft_pending

        draft_output = _hf_forward_with_cache(
            draft_model,
            draft_input,
            draft_cache,
            draft_cache_length,
        )
        draft_cache_length += draft_input.shape[1]
        draft_ids = torch.argmax(draft_output.logits[0, -requested:, :], dim=-1).tolist()
        draft_ids = [int(token_id) for token_id in draft_ids]

        draft_tensor = torch.tensor([draft_ids], dtype=torch.long, device=target_device)
        target_input = torch.cat([target_pending, draft_tensor], dim=1)
        target_output = _hf_forward_with_cache(
            target_model,
            target_input,
            target_cache,
            target_cache_length,
        )
        target_cache_length += target_input.shape[1]
        predictions = torch.argmax(target_output.logits[0, -(requested + 1) :, :], dim=-1).tolist()
        predictions = [int(token_id) for token_id in predictions]

        accepted_count = 0
        for index, draft_token in enumerate(draft_ids):
            if predictions[index] != int(draft_token):
                break
            accepted_count += 1
        emitted = draft_ids[:accepted_count] + [predictions[accepted_count]]

        counters["speculation_steps"] += 1
        counters["target_forwards"] += 1
        counters["draft_forwards"] += 1
        counters["proposed_draft_tokens"] += len(draft_ids)
        counters["accepted_draft_tokens"] += accepted_count

        kept: list[int] = []
        stop = False
        for token in emitted:
            if len(generated) >= max_new_tokens:
                stop = True
                break
            token = int(token)
            generated.append(token)
            kept.append(token)
            if eos_token_id is not None and token == eos_token_id:
                stop = True
                break

        if not kept:
            break
        total_token_length = len(prompt_ids) + len(generated)
        target_cache_length = max(total_token_length - 1, 0)
        target_pending = torch.tensor([[kept[-1]]], dtype=torch.long, device=target_device)

        # Drop the mask-token KV entries. The next draft forward writes accepted
        # real tokens over those slots.
        draft_cache_length -= mask_count
        draft_pending = torch.tensor([kept], dtype=torch.long, device=draft_device)
        if stop:
            break

    return generated, counters


@torch.inference_mode()
def _run_fallback_pard_speculative_decode(
    target_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    draft_len: int,
    mask_token_id: int,
    eos_token_id: int | None,
) -> tuple[list[int], dict[str, int]]:
    prefix = list(prompt_ids)
    generated: list[int] = []
    counters = {
        "speculation_steps": 0,
        "target_forwards": 0,
        "draft_forwards": 0,
        "proposed_draft_tokens": 0,
        "accepted_draft_tokens": 0,
    }
    while len(generated) < max_new_tokens:
        requested = min(draft_len, max_new_tokens - len(generated))
        draft_ids = propose_parallel_draft_tokens(
            draft_model=draft_model,
            prefix_ids=prefix,
            draft_len=requested,
            mask_token_id=mask_token_id,
        )
        result = greedy_verify(target_model, prefix_ids=prefix, draft_ids=draft_ids)
        counters["speculation_steps"] += 1
        counters["target_forwards"] += 1
        counters["draft_forwards"] += 1
        counters["proposed_draft_tokens"] += result.proposed_draft_tokens
        counters["accepted_draft_tokens"] += result.accepted_draft_tokens
        for token in result.emitted_ids:
            if len(generated) >= max_new_tokens:
                break
            prefix.append(int(token))
            generated.append(int(token))
            if eos_token_id is not None and int(token) == eos_token_id:
                return generated, counters
    return generated, counters


@torch.inference_mode()
def run_pard_speculative_decode(
    target_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    draft_len: int,
    mask_token_id: int,
    eos_token_id: int | None = None,
) -> tuple[list[int], dict[str, int]]:
    if _supports_hf_cache(target_model) and _supports_hf_cache(draft_model):
        return _run_hf_pard_speculative_decode(
            target_model=target_model,
            draft_model=draft_model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            draft_len=draft_len,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
        )
    return _run_fallback_pard_speculative_decode(
        target_model=target_model,
        draft_model=draft_model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        draft_len=draft_len,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PARD speculative decoding inference.")
    parser.add_argument("--model-path", default=DEFAULT_TARGET_MODEL_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--output", default="")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--draft-len", type=int, default=DEFAULT_INFERENCE_DRAFT_LEN)
    parser.add_argument("--pard-token-id", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup-prompts", type=int, default=DEFAULT_WARMUP_PROMPTS)
    parser.add_argument("--limit-prompts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--allow-divergence", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--require-baseline-match", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.temperature != 0.0 or args.top_p != 1.0:
        raise ValueError("PARD inference currently supports greedy decoding only")

    torch.manual_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    prompts = load_prompt_records(args.prompts, tokenizer=tokenizer)
    if args.limit_prompts > 0:
        prompts = prompts[: args.limit_prompts]

    target_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=dtype if torch.device(args.device).type != "cpu" else torch.float32,
        attn_implementation=args.attn_implementation,
    ).to(device=args.device)
    draft_model, metadata = load_pard_checkpoint(args.checkpoint_path, device=args.device, dtype=dtype)
    pard_token_id = int(
        args.pard_token_id
        or metadata.get("pard_token_id", 0)
        or getattr(getattr(draft_model, "config", object()), "pard_token", 0)
        or DEFAULT_PARD_TOKEN_ID
    )
    target_model.eval()
    draft_model.eval()

    compile_enabled = False
    if args.compile:
        target_model.forward = torch.compile(target_model.forward, mode="reduce-overhead", fullgraph=False)
        draft_model.forward = torch.compile(draft_model.forward, mode="reduce-overhead", fullgraph=False)
        compile_enabled = True

    output_path = (
        Path(args.output)
        if args.output
        else Path(f"runs/parallel_draft_models_len{args.draft_len}_nonvllm.jsonl")
    )
    if output_path.exists():
        output_path.unlink()

    def synchronize() -> None:
        if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(torch.device(args.device))

    warmup_count = min(max(args.warmup_prompts, 0), len(prompts))
    for record in prompts[:warmup_count]:
        if not args.skip_baseline:
            if _supports_hf_cache(target_model):
                _hf_greedy_generate(target_model, record.prompt_ids, args.max_new_tokens, tokenizer.eos_token_id)
            else:
                autoregressive_generate(
                    target_model,
                    record.prompt_ids,
                    args.max_new_tokens,
                    args.temperature,
                    args.top_p,
                    tokenizer.eos_token_id,
                )
        run_pard_speculative_decode(
            target_model=target_model,
            draft_model=draft_model,
            prompt_ids=record.prompt_ids,
            max_new_tokens=args.max_new_tokens,
            draft_len=args.draft_len,
            mask_token_id=pard_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        synchronize()

    divergence_count = 0
    first_diverged_prompt_id = ""
    token_count_mismatches = 0
    for record in prompts:
        baseline_tokens: list[int] = []
        baseline_time_s = 0.0
        if not args.skip_baseline:
            synchronize()
            wall_start = time.perf_counter()
            if _supports_hf_cache(target_model):
                baseline_tokens = _hf_greedy_generate(
                    target_model,
                    record.prompt_ids,
                    args.max_new_tokens,
                    tokenizer.eos_token_id,
                )
            else:
                baseline_tokens = autoregressive_generate(
                    target_model,
                    record.prompt_ids,
                    args.max_new_tokens,
                    args.temperature,
                    args.top_p,
                    tokenizer.eos_token_id,
                )
            synchronize()
            baseline_time_s = time.perf_counter() - wall_start

        synchronize()
        wall_start = time.perf_counter()
        generated_tokens, counters = run_pard_speculative_decode(
            target_model=target_model,
            draft_model=draft_model,
            prompt_ids=record.prompt_ids,
            max_new_tokens=args.max_new_tokens,
            draft_len=args.draft_len,
            mask_token_id=pard_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        synchronize()
        method_time_s = time.perf_counter() - wall_start

        matches_baseline = True
        if not args.skip_baseline and generated_tokens != baseline_tokens:
            matches_baseline = False
            divergence_count += 1
            if len(generated_tokens) != len(baseline_tokens):
                token_count_mismatches += 1
            if not first_diverged_prompt_id:
                first_diverged_prompt_id = record.prompt_id
        if args.require_baseline_match and not matches_baseline:
            raise RuntimeError(f"PARD greedy output diverged for {record.prompt_id}")

        stats = SpecDecodeStats(
            method="parallel_draft_models",
            model=args.model_path,
            prompt_id=record.prompt_id,
            prompt_tokens=len(record.prompt_ids),
            generated_tokens=len(generated_tokens),
            generated_text=tokenizer.decode(generated_tokens, skip_special_tokens=True),
            temperature=args.temperature,
            draft_len=args.draft_len,
            speculation_steps=counters["speculation_steps"],
            target_forwards=counters["target_forwards"],
            draft_forwards=counters["draft_forwards"],
            proposed_draft_tokens=counters["proposed_draft_tokens"],
            accepted_draft_tokens=counters["accepted_draft_tokens"],
            baseline_wall_time_s=baseline_time_s,
            method_wall_time_s=method_time_s,
            torch_compile=compile_enabled,
            cuda_graphs=False,
            cuda_graphs_reason="disabled: Python-level cached PARD decode loop",
            seed=args.seed,
        )
        output_record = stats.to_record()
        output_record["matches_baseline"] = matches_baseline
        output_record["pard_token_id"] = pard_token_id
        write_jsonl_record(output_path, output_record)

    summary = summarize_jsonl(output_path) | {
        "method": "parallel_draft_models",
        "num_prompts": len(prompts),
        "output": str(output_path),
        "matches_baseline": divergence_count == 0,
        "diverged_prompts": divergence_count,
        "first_diverged_prompt_id": first_diverged_prompt_id,
        "token_count_mismatches": token_count_mismatches,
        "pard_token_id": pard_token_id,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

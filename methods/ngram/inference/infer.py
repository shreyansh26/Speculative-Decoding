from __future__ import annotations

"""
Training-free n-gram speculative decoding.

The proposer uses exact suffix matches against the prompt plus accepted
generation history. The non-vLLM verifier keeps the target KV cache in a
pending-token form so each speculative step needs at most one target decode
forward.
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from common.metrics import SpecDecodeStats, summarize_jsonl, write_jsonl_record
from common.qwen3 import Qwen3ForCausalLM
from common.sampling import autoregressive_generate
from common.tokenizer import load_tokenizer, render_prompt
from common.verification import advance_prefix_state, prefill_prefix, run_greedy_speculative_decode


DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_PROMPTS_PATH = "data/wiki_extract_ngram_eval100_qwen25_7b.jsonl"
DEFAULT_OUTPUT_PATH = "runs/ngram_wiki_nonvllm.jsonl"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_DRAFT_LEN = 5
DEFAULT_PROMPT_LOOKUP_MIN = 3
DEFAULT_PROMPT_LOOKUP_MAX = 8
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_WARMUP_PROMPTS = 1


@dataclass(slots=True)
class PromptRecord:
    prompt_id: str
    prompt_ids: list[int]
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NgramTimings:
    prefill_wall_time_s: float
    decode_wall_time_s: float
    total_wall_time_s: float


class NgramDraftIndex:
    def __init__(
        self,
        history_ids: Sequence[int],
        prompt_lookup_min: int,
        prompt_lookup_max: int,
    ) -> None:
        if prompt_lookup_min <= 0:
            raise ValueError("prompt_lookup_min must be positive")
        if prompt_lookup_max < prompt_lookup_min:
            raise ValueError("prompt_lookup_max must be >= prompt_lookup_min")
        self.prompt_lookup_min = int(prompt_lookup_min)
        self.prompt_lookup_max = int(prompt_lookup_max)
        self.history: list[int] = []
        self.positions_by_size: dict[int, dict[tuple[int, ...], list[int]]] = {
            size: defaultdict(list)
            for size in range(self.prompt_lookup_min, self.prompt_lookup_max + 1)
        }
        self.extend(history_ids)

    def extend(self, token_ids: Sequence[int]) -> None:
        for token_id in token_ids:
            self.append(int(token_id))

    def append(self, token_id: int) -> None:
        self.history.append(int(token_id))
        end = len(self.history)
        for ngram_size in range(self.prompt_lookup_min, self.prompt_lookup_max + 1):
            if end < ngram_size:
                continue
            start = end - ngram_size
            key = tuple(self.history[start:end])
            self.positions_by_size[ngram_size][key].append(start)

    def propose(self, draft_len: int) -> list[int]:
        if draft_len <= 0:
            return []
        history_len = len(self.history)
        if history_len < self.prompt_lookup_min:
            return []

        max_window = min(self.prompt_lookup_max, history_len)
        for ngram_size in range(max_window, self.prompt_lookup_min - 1, -1):
            suffix_start = history_len - ngram_size
            suffix = tuple(self.history[suffix_start:])
            positions = self.positions_by_size[ngram_size].get(suffix, ())
            for start in positions:
                if start >= suffix_start:
                    break
                candidate_start = start + ngram_size
                if candidate_start >= history_len:
                    continue
                return self.history[candidate_start : candidate_start + draft_len]
        return []


def find_ngram_draft(
    history_ids: Sequence[int],
    draft_len: int,
    prompt_lookup_min: int,
    prompt_lookup_max: int,
) -> list[int]:
    index = NgramDraftIndex(
        history_ids=history_ids,
        prompt_lookup_min=prompt_lookup_min,
        prompt_lookup_max=prompt_lookup_max,
    )
    return index.propose(draft_len)


def build_ngram_draft_provider(
    prompt_lookup_min: int,
    prompt_lookup_max: int,
    draft_len: int,
):
    def provider(history_ids: list[int], requested: int) -> list[int]:
        return find_ngram_draft(
            history_ids=history_ids,
            draft_len=min(draft_len, requested),
            prompt_lookup_min=prompt_lookup_min,
            prompt_lookup_max=prompt_lookup_max,
        )

    return provider


def _model_device(model: torch.nn.Module) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _supports_cached_decode(model: torch.nn.Module) -> bool:
    return hasattr(model, "prefill") and hasattr(model, "decode_many")


def _as_tensor(token_ids: Sequence[int], device: torch.device) -> torch.Tensor:
    if not token_ids:
        raise ValueError("token_ids must not be empty")
    return torch.tensor([list(token_ids)], dtype=torch.long, device=device)


def _truncate_cache(cache: list[object | None], total_length: int) -> list[object | None]:
    truncated: list[object | None] = []
    for layer_cache in cache:
        if layer_cache is None:
            truncated.append(None)
            continue
        truncated.append(
            type(layer_cache)(
                key=layer_cache.key[:, :, :total_length, :].detach(),
                value=layer_cache.value[:, :, :total_length, :].detach(),
            )
        )
    return truncated


def _synchronize(device: str | torch.device) -> None:
    torch_device = torch.device(device)
    if torch.cuda.is_available() and torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)


def _empty_counters() -> dict[str, int]:
    return {
        "speculation_steps": 0,
        "target_forwards": 0,
        "draft_forwards": 0,
        "proposed_draft_tokens": 0,
        "accepted_draft_tokens": 0,
    }


@torch.inference_mode()
def _run_cached_ngram_speculative_decode(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    draft_len: int,
    prompt_lookup_min: int,
    prompt_lookup_max: int,
    eos_token_id: int | None,
    sync_device: str | torch.device | None = None,
) -> tuple[list[int], dict[str, int], NgramTimings]:
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if draft_len < 0:
        raise ValueError("draft_len must be non-negative")

    device = _model_device(model)
    if sync_device is not None:
        _synchronize(sync_device)
    total_start = time.perf_counter()
    prefill_start = time.perf_counter()
    state = prefill_prefix(model, prompt_ids)
    if sync_device is not None:
        _synchronize(sync_device)
    prefill_time_s = time.perf_counter() - prefill_start

    cache = state.cache
    cached_ids = list(prompt_ids)
    pending_ids: list[int] = []
    last_logits = state.last_logits
    index = NgramDraftIndex(
        prompt_ids,
        prompt_lookup_min=prompt_lookup_min,
        prompt_lookup_max=prompt_lookup_max,
    )
    generated: list[int] = []
    counters = _empty_counters()

    decode_start = time.perf_counter()
    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        requested = min(draft_len, remaining)
        draft_ids = index.propose(requested)
        if len(draft_ids) > requested:
            raise ValueError("ngram proposer returned more tokens than requested")

        counters["speculation_steps"] += 1
        counters["proposed_draft_tokens"] += len(draft_ids)
        counters["draft_forwards"] += 1 if draft_ids else 0

        first_prediction: int | None = None
        output = None
        target_input: list[int] = []
        ran_target_forward = False

        if not pending_ids:
            first_prediction = int(torch.argmax(last_logits).item())
            if not draft_ids or first_prediction != int(draft_ids[0]):
                predictions = [first_prediction]
            else:
                target_input = list(draft_ids)
                output = model.decode_many(_as_tensor(target_input, device), cache=cache)
                ran_target_forward = True
                tail_predictions = torch.argmax(output.logits[0, : len(draft_ids)], dim=-1).tolist()
                predictions = [first_prediction] + [int(token_id) for token_id in tail_predictions]
        else:
            target_input = pending_ids + list(draft_ids)
            output = model.decode_many(_as_tensor(target_input, device), cache=cache)
            ran_target_forward = True
            start = len(pending_ids) - 1
            stop = len(pending_ids) + len(draft_ids)
            predictions = torch.argmax(output.logits[0, start:stop], dim=-1).tolist()
            predictions = [int(token_id) for token_id in predictions]

        if ran_target_forward:
            counters["target_forwards"] += 1

        accepted_count = 0
        for draft_token in draft_ids:
            if predictions[accepted_count] != int(draft_token):
                break
            accepted_count += 1

        final_token = int(predictions[accepted_count])
        emitted = [int(token_id) for token_id in draft_ids[:accepted_count]] + [final_token]
        counters["accepted_draft_tokens"] += accepted_count

        if ran_target_forward:
            assert output is not None
            consumed_count = len(pending_ids) + accepted_count
            if consumed_count > 0:
                cached_ids.extend(target_input[:consumed_count])
                cache = _truncate_cache(getattr(output, "cache", None) or [], len(cached_ids))
                last_logits = output.logits[0, consumed_count - 1].detach()
        pending_ids = [final_token]

        for token_id in emitted:
            if len(generated) >= max_new_tokens:
                break
            token_id = int(token_id)
            generated.append(token_id)
            index.append(token_id)
            if eos_token_id is not None and token_id == eos_token_id:
                if sync_device is not None:
                    _synchronize(sync_device)
                decode_time_s = time.perf_counter() - decode_start
                total_time_s = time.perf_counter() - total_start
                return generated, counters, NgramTimings(
                    prefill_wall_time_s=prefill_time_s,
                    decode_wall_time_s=decode_time_s,
                    total_wall_time_s=total_time_s,
                )

    if sync_device is not None:
        _synchronize(sync_device)
    decode_time_s = time.perf_counter() - decode_start
    total_time_s = time.perf_counter() - total_start
    return generated, counters, NgramTimings(
        prefill_wall_time_s=prefill_time_s,
        decode_wall_time_s=decode_time_s,
        total_wall_time_s=total_time_s,
    )


def run_ngram_speculative_decode(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    draft_len: int,
    prompt_lookup_min: int,
    prompt_lookup_max: int,
    eos_token_id: int | None = None,
) -> tuple[list[int], dict[str, int]]:
    if _supports_cached_decode(model):
        generated, counters, _ = _run_cached_ngram_speculative_decode(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            draft_len=draft_len,
            prompt_lookup_min=prompt_lookup_min,
            prompt_lookup_max=prompt_lookup_max,
            eos_token_id=eos_token_id,
        )
        return generated, counters

    provider = build_ngram_draft_provider(
        prompt_lookup_min=prompt_lookup_min,
        prompt_lookup_max=prompt_lookup_max,
        draft_len=draft_len,
    )
    return run_greedy_speculative_decode(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        draft_provider=provider,
        draft_len=draft_len,
    )


def timed_ngram_speculative_decode(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    draft_len: int,
    prompt_lookup_min: int,
    prompt_lookup_max: int,
    eos_token_id: int | None,
    sync_device: str | torch.device | None,
) -> tuple[list[int], dict[str, int], NgramTimings]:
    if _supports_cached_decode(model):
        return _run_cached_ngram_speculative_decode(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            draft_len=draft_len,
            prompt_lookup_min=prompt_lookup_min,
            prompt_lookup_max=prompt_lookup_max,
            eos_token_id=eos_token_id,
            sync_device=sync_device,
        )

    if sync_device is not None:
        _synchronize(sync_device)
    start = time.perf_counter()
    generated, counters = run_ngram_speculative_decode(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        draft_len=draft_len,
        prompt_lookup_min=prompt_lookup_min,
        prompt_lookup_max=prompt_lookup_max,
        eos_token_id=eos_token_id,
    )
    if sync_device is not None:
        _synchronize(sync_device)
    elapsed = time.perf_counter() - start
    return generated, counters, NgramTimings(0.0, elapsed, elapsed)


@torch.inference_mode()
def timed_cached_greedy_generate(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    eos_token_id: int | None,
    sync_device: str | torch.device | None,
) -> tuple[list[int], NgramTimings]:
    if not _supports_cached_decode(model):
        if sync_device is not None:
            _synchronize(sync_device)
        start = time.perf_counter()
        generated = autoregressive_generate(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=eos_token_id,
        )
        if sync_device is not None:
            _synchronize(sync_device)
        elapsed = time.perf_counter() - start
        return generated, NgramTimings(0.0, elapsed, elapsed)

    if sync_device is not None:
        _synchronize(sync_device)
    total_start = time.perf_counter()
    prefill_start = time.perf_counter()
    state = prefill_prefix(model, prompt_ids)
    if sync_device is not None:
        _synchronize(sync_device)
    prefill_time_s = time.perf_counter() - prefill_start

    generated: list[int] = []
    decode_start = time.perf_counter()
    while len(generated) < max_new_tokens:
        token_id = int(torch.argmax(state.last_logits).item())
        generated.append(token_id)
        if eos_token_id is not None and token_id == eos_token_id:
            break
        if len(generated) >= max_new_tokens:
            break
        state = advance_prefix_state(model, state, token_id)

    if sync_device is not None:
        _synchronize(sync_device)
    decode_time_s = time.perf_counter() - decode_start
    total_time_s = time.perf_counter() - total_start
    return generated, NgramTimings(prefill_time_s, decode_time_s, total_time_s)


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
                prompt = str(sample.get("prompt", ""))
            else:
                rendered = render_prompt(tokenizer, sample)
                prompt_ids = tokenizer.encode(rendered.prompt, add_special_tokens=False)
                prompt = rendered.prompt
            records.append(
                PromptRecord(
                    prompt_id=prompt_id,
                    prompt_ids=prompt_ids,
                    prompt=prompt,
                    metadata={key: value for key, value in sample.items() if key not in {"prompt_ids", "prompt"}},
                )
            )
    if not records:
        raise ValueError(f"{prompts_path} did not contain any prompts")
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="N-gram speculative decoding inference.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--draft-len", type=int, default=DEFAULT_DRAFT_LEN)
    parser.add_argument("--prompt-lookup-min", type=int, default=DEFAULT_PROMPT_LOOKUP_MIN)
    parser.add_argument("--prompt-lookup-max", type=int, default=DEFAULT_PROMPT_LOOKUP_MAX)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--warmup-prompts", type=int, default=DEFAULT_WARMUP_PROMPTS)
    parser.add_argument("--limit-prompts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--require-baseline-match", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.temperature != 0.0 or args.top_p != 1.0:
        raise ValueError("n-gram inference currently supports greedy decoding only")

    torch.manual_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    prompts = load_prompt_records(args.prompts, tokenizer=tokenizer)
    if args.limit_prompts > 0:
        prompts = prompts[: args.limit_prompts]

    model = Qwen3ForCausalLM.from_pretrained(args.model_path, device=args.device, dtype=dtype)
    model.eval()
    compile_enabled = False
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
        compile_enabled = True

    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()

    sync_device = args.device if torch.device(args.device).type == "cuda" else None
    warmup_count = min(max(args.warmup_prompts, 0), len(prompts))
    for record in prompts[:warmup_count]:
        if len(record.prompt_ids) + args.max_new_tokens > args.max_model_len:
            continue
        if not args.skip_baseline:
            timed_cached_greedy_generate(
                model=model,
                prompt_ids=record.prompt_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                sync_device=sync_device,
            )
        timed_ngram_speculative_decode(
            model=model,
            prompt_ids=record.prompt_ids,
            max_new_tokens=args.max_new_tokens,
            draft_len=args.draft_len,
            prompt_lookup_min=args.prompt_lookup_min,
            prompt_lookup_max=args.prompt_lookup_max,
            eos_token_id=tokenizer.eos_token_id,
            sync_device=sync_device,
        )

    divergence_count = 0
    first_diverged_prompt_id = ""
    token_count_mismatches = 0
    for record in prompts:
        if len(record.prompt_ids) + args.max_new_tokens > args.max_model_len:
            raise ValueError(
                f"{record.prompt_id} has {len(record.prompt_ids)} prompt tokens and "
                f"{args.max_new_tokens} requested decode tokens, exceeding "
                f"--max-model-len={args.max_model_len}"
            )

        baseline_tokens: list[int] = []
        baseline_timings = NgramTimings(0.0, 0.0, 0.0)
        if not args.skip_baseline:
            baseline_tokens, baseline_timings = timed_cached_greedy_generate(
                model=model,
                prompt_ids=record.prompt_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                sync_device=sync_device,
            )

        generated_tokens, counters, method_timings = timed_ngram_speculative_decode(
            model=model,
            prompt_ids=record.prompt_ids,
            max_new_tokens=args.max_new_tokens,
            draft_len=args.draft_len,
            prompt_lookup_min=args.prompt_lookup_min,
            prompt_lookup_max=args.prompt_lookup_max,
            eos_token_id=tokenizer.eos_token_id,
            sync_device=sync_device,
        )

        matches_baseline = True
        if baseline_tokens and generated_tokens != baseline_tokens:
            matches_baseline = False
            divergence_count += 1
            if len(generated_tokens) != len(baseline_tokens):
                token_count_mismatches += 1
            if not first_diverged_prompt_id:
                first_diverged_prompt_id = record.prompt_id
        if args.require_baseline_match and not matches_baseline:
            raise RuntimeError(f"n-gram greedy output diverged for {record.prompt_id}")

        stats = SpecDecodeStats(
            method="ngram",
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
            baseline_wall_time_s=baseline_timings.total_wall_time_s,
            method_wall_time_s=method_timings.total_wall_time_s,
            torch_compile=compile_enabled,
            cuda_graphs=False,
            cuda_graphs_reason=(
                "disabled: dynamic n-gram lookup and variable accepted lengths"
                if args.cuda_graphs
                else "disabled"
            ),
            seed=args.seed,
        )
        record_out = stats.to_record()
        record_out |= {
            "matches_baseline": matches_baseline,
            "prompt_lookup_min": args.prompt_lookup_min,
            "prompt_lookup_max": args.prompt_lookup_max,
            "baseline_prefill_wall_time_s": baseline_timings.prefill_wall_time_s,
            "baseline_decode_wall_time_s": baseline_timings.decode_wall_time_s,
            "method_prefill_wall_time_s": method_timings.prefill_wall_time_s,
            "method_decode_wall_time_s": method_timings.decode_wall_time_s,
            "decode_speedup": (
                baseline_timings.decode_wall_time_s / method_timings.decode_wall_time_s
                if method_timings.decode_wall_time_s
                else 0.0
            ),
            "max_model_len": args.max_model_len,
            "article_title": record.metadata.get("article_title", ""),
            "prompt_truncated": bool(record.metadata.get("prompt_truncated", False)),
        }
        write_jsonl_record(output_path, record_out)

    summary = summarize_jsonl(output_path) | {
        "method": "ngram",
        "num_prompts": len(prompts),
        "output": str(output_path),
        "matches_baseline": divergence_count == 0,
        "diverged_prompts": divergence_count,
        "first_diverged_prompt_id": first_diverged_prompt_id,
        "token_count_mismatches": token_count_mismatches,
        "draft_len": args.draft_len,
        "prompt_lookup_min": args.prompt_lookup_min,
        "prompt_lookup_max": args.prompt_lookup_max,
        "max_model_len": args.max_model_len,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

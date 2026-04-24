from __future__ import annotations

"""
Training-free suffix decoding.

The proposer keeps bounded suffix-frequency indexes over the prompt, accepted
generation history, and an optional global cache. The target verifier uses the
same pending-token KV-cache shape as the n-gram implementation so each
speculative step needs at most one target decode forward.
"""

import argparse
import json
import time
from collections import Counter, defaultdict
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
DEFAULT_OUTPUT_PATH = "runs/suffix_wiki_nonvllm.jsonl"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_DRAFT_LEN = 24
DEFAULT_MAX_TREE_DEPTH = 24
DEFAULT_MAX_SPEC_FACTOR = 1.0
DEFAULT_MIN_TOKEN_PROB = 0.1
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_WARMUP_PROMPTS = 1


@dataclass(slots=True)
class PromptRecord:
    prompt_id: str
    prompt_ids: list[int]
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DecodeTimings:
    prefill_wall_time_s: float
    decode_wall_time_s: float
    total_wall_time_s: float


class SuffixIndex:
    def __init__(self, max_tree_depth: int = DEFAULT_MAX_TREE_DEPTH) -> None:
        if max_tree_depth <= 0:
            raise ValueError("max_tree_depth must be positive")
        self.max_tree_depth = int(max_tree_depth)
        self.history: list[int] = []
        self.counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)

    def update(self, tokens: Sequence[int]) -> None:
        self.extend(tokens)

    def extend(self, token_ids: Sequence[int]) -> None:
        for token_id in token_ids:
            self.append(int(token_id))

    def append(self, token_id: int) -> None:
        next_token = int(token_id)
        max_depth = min(self.max_tree_depth, len(self.history))
        for depth in range(1, max_depth + 1):
            key = tuple(self.history[-depth:])
            self.counts[key][next_token] += 1
        self.history.append(next_token)

    def next_token_distribution(self, suffix: Sequence[int]) -> tuple[tuple[int, ...], Counter[int]]:
        values = list(suffix)
        max_depth = min(self.max_tree_depth, len(values))
        for depth in range(max_depth, 0, -1):
            key = tuple(values[-depth:])
            counts = self.counts.get(key)
            if counts:
                return key, counts
        return (), Counter()


def _merge_counts(local: Counter[int], global_counts: Counter[int]) -> Counter[int]:
    if not local:
        return global_counts.copy()
    if not global_counts:
        return local.copy()
    merged = local.copy()
    merged.update(global_counts)
    return merged


def _best_distribution(
    index: SuffixIndex,
    suffix: Sequence[int],
    global_index: SuffixIndex | None = None,
) -> tuple[tuple[int, ...], Counter[int]]:
    values = list(suffix)
    max_available_depth = index.max_tree_depth
    if global_index is not None:
        max_available_depth = max(max_available_depth, global_index.max_tree_depth)
    max_depth = min(max_available_depth, len(values))
    for depth in range(max_depth, 0, -1):
        key = tuple(values[-depth:])
        local_counts = index.counts.get(key, Counter()) if depth <= index.max_tree_depth else Counter()
        global_counts = (
            global_index.counts.get(key, Counter())
            if global_index is not None and depth <= global_index.max_tree_depth
            else Counter()
        )
        if local_counts or global_counts:
            return key, _merge_counts(local_counts, global_counts)
    return (), Counter()


def _most_likely_token(counts: Counter[int]) -> tuple[int, int]:
    return max(counts.items(), key=lambda item: (item[1], -item[0]))


def propose_suffix_tokens(
    index: SuffixIndex,
    history_ids: Sequence[int],
    draft_len: int,
    max_spec_factor: float,
    min_token_prob: float,
    global_index: SuffixIndex | None = None,
) -> list[int]:
    if draft_len <= 0:
        return []
    if max_spec_factor < 0:
        raise ValueError("max_spec_factor must be non-negative")
    if not 0.0 <= min_token_prob <= 1.0:
        raise ValueError("min_token_prob must be in [0, 1]")

    proposals: list[int] = []
    working_history = list(history_ids)
    while len(proposals) < draft_len:
        match, next_counts = _best_distribution(index, working_history, global_index=global_index)
        if not match or not next_counts:
            break
        max_spec = min(
            draft_len,
            max(1, int(max_spec_factor * len(match))),
            index.max_tree_depth - len(match) + 1,
        )
        if len(proposals) >= max_spec:
            break
        total = sum(next_counts.values())
        token, count = _most_likely_token(next_counts)
        if total <= 0 or (count / total) < min_token_prob:
            break
        proposals.append(token)
        working_history.append(token)
    return proposals


def load_global_suffix_index(path: str | Path | None, max_tree_depth: int) -> SuffixIndex:
    index = SuffixIndex(max_tree_depth=max_tree_depth)
    if not path:
        return index
    cache_path = Path(path)
    if not cache_path.exists():
        return index
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            index.update(row["tokens"])
    return index


def append_global_suffix_cache(path: str | Path | None, tokens: Sequence[int]) -> None:
    if not path:
        return
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"tokens": list(tokens)}, ensure_ascii=True) + "\n")


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
def _run_cached_suffix_speculative_decode(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    draft_len: int,
    max_tree_depth: int,
    max_spec_factor: float,
    min_token_prob: float,
    global_index: SuffixIndex | None,
    eos_token_id: int | None,
    sync_device: str | torch.device | None = None,
) -> tuple[list[int], dict[str, int], DecodeTimings]:
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
    index = SuffixIndex(max_tree_depth=max_tree_depth)
    index.update(prompt_ids)
    generated: list[int] = []
    counters = _empty_counters()

    decode_start = time.perf_counter()
    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        requested = min(draft_len, remaining)
        history_ids = index.history
        draft_ids = propose_suffix_tokens(
            index=index,
            history_ids=history_ids,
            draft_len=requested,
            max_spec_factor=max_spec_factor,
            min_token_prob=min_token_prob,
            global_index=global_index,
        )
        if len(draft_ids) > requested:
            raise ValueError("suffix proposer returned more tokens than requested")

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
                return generated, counters, DecodeTimings(prefill_time_s, decode_time_s, total_time_s)

    if sync_device is not None:
        _synchronize(sync_device)
    decode_time_s = time.perf_counter() - decode_start
    total_time_s = time.perf_counter() - total_start
    return generated, counters, DecodeTimings(prefill_time_s, decode_time_s, total_time_s)


def run_suffix_speculative_decode(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    draft_len: int,
    max_tree_depth: int,
    max_spec_factor: float,
    min_token_prob: float,
    global_index: SuffixIndex | None = None,
    eos_token_id: int | None = None,
) -> tuple[list[int], dict[str, int]]:
    if _supports_cached_decode(model):
        generated, counters, _ = _run_cached_suffix_speculative_decode(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            draft_len=draft_len,
            max_tree_depth=max_tree_depth,
            max_spec_factor=max_spec_factor,
            min_token_prob=min_token_prob,
            global_index=global_index,
            eos_token_id=eos_token_id,
        )
        return generated, counters

    def provider(history_ids: list[int], requested: int) -> list[int]:
        index = SuffixIndex(max_tree_depth=max_tree_depth)
        index.update(history_ids)
        return propose_suffix_tokens(
            index=index,
            history_ids=history_ids,
            draft_len=min(draft_len, requested),
            max_spec_factor=max_spec_factor,
            min_token_prob=min_token_prob,
            global_index=global_index,
        )

    return run_greedy_speculative_decode(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        draft_provider=provider,
        draft_len=draft_len,
    )


def timed_suffix_speculative_decode(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    draft_len: int,
    max_tree_depth: int,
    max_spec_factor: float,
    min_token_prob: float,
    global_index: SuffixIndex | None,
    eos_token_id: int | None,
    sync_device: str | torch.device | None,
) -> tuple[list[int], dict[str, int], DecodeTimings]:
    if _supports_cached_decode(model):
        return _run_cached_suffix_speculative_decode(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            draft_len=draft_len,
            max_tree_depth=max_tree_depth,
            max_spec_factor=max_spec_factor,
            min_token_prob=min_token_prob,
            global_index=global_index,
            eos_token_id=eos_token_id,
            sync_device=sync_device,
        )

    if sync_device is not None:
        _synchronize(sync_device)
    start = time.perf_counter()
    generated, counters = run_suffix_speculative_decode(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        draft_len=draft_len,
        max_tree_depth=max_tree_depth,
        max_spec_factor=max_spec_factor,
        min_token_prob=min_token_prob,
        global_index=global_index,
        eos_token_id=eos_token_id,
    )
    if sync_device is not None:
        _synchronize(sync_device)
    elapsed = time.perf_counter() - start
    return generated, counters, DecodeTimings(0.0, elapsed, elapsed)


@torch.inference_mode()
def timed_cached_greedy_generate(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    eos_token_id: int | None,
    sync_device: str | torch.device | None,
) -> tuple[list[int], DecodeTimings]:
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
        return generated, DecodeTimings(0.0, elapsed, elapsed)

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
    return generated, DecodeTimings(prefill_time_s, decode_time_s, total_time_s)


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
    parser = argparse.ArgumentParser(description="Suffix decoding speculative inference.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--draft-len", type=int, default=DEFAULT_DRAFT_LEN)
    parser.add_argument("--max-tree-depth", type=int, default=DEFAULT_MAX_TREE_DEPTH)
    parser.add_argument("--max-spec-factor", type=float, default=DEFAULT_MAX_SPEC_FACTOR)
    parser.add_argument("--min-token-prob", type=float, default=DEFAULT_MIN_TOKEN_PROB)
    parser.add_argument("--global-cache", default="")
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
        raise ValueError("suffix decoding currently supports greedy decoding only")

    torch.manual_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    prompts = load_prompt_records(args.prompts, tokenizer=tokenizer)
    if args.limit_prompts > 0:
        prompts = prompts[: args.limit_prompts]

    model = Qwen3ForCausalLM.from_pretrained(args.model_path, device=args.device, dtype=dtype)
    model.eval()
    global_index = load_global_suffix_index(args.global_cache or None, max_tree_depth=args.max_tree_depth)

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
        timed_suffix_speculative_decode(
            model=model,
            prompt_ids=record.prompt_ids,
            max_new_tokens=args.max_new_tokens,
            draft_len=args.draft_len,
            max_tree_depth=args.max_tree_depth,
            max_spec_factor=args.max_spec_factor,
            min_token_prob=args.min_token_prob,
            global_index=global_index,
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
        baseline_timings = DecodeTimings(0.0, 0.0, 0.0)
        if not args.skip_baseline:
            baseline_tokens, baseline_timings = timed_cached_greedy_generate(
                model=model,
                prompt_ids=record.prompt_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                sync_device=sync_device,
            )

        generated_tokens, counters, method_timings = timed_suffix_speculative_decode(
            model=model,
            prompt_ids=record.prompt_ids,
            max_new_tokens=args.max_new_tokens,
            draft_len=args.draft_len,
            max_tree_depth=args.max_tree_depth,
            max_spec_factor=args.max_spec_factor,
            min_token_prob=args.min_token_prob,
            global_index=global_index,
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
            raise RuntimeError(f"suffix decoding greedy output diverged for {record.prompt_id}")

        stats = SpecDecodeStats(
            method="suffix_decoding",
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
                "disabled: dynamic suffix lookup and variable accepted lengths"
                if args.cuda_graphs
                else "disabled"
            ),
            seed=args.seed,
        )
        record_out = stats.to_record()
        record_out |= {
            "matches_baseline": matches_baseline,
            "max_tree_depth": args.max_tree_depth,
            "max_spec_factor": args.max_spec_factor,
            "min_token_prob": args.min_token_prob,
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
            "global_cache": args.global_cache,
            "article_title": record.metadata.get("article_title", ""),
            "prompt_truncated": bool(record.metadata.get("prompt_truncated", False)),
        }
        write_jsonl_record(output_path, record_out)

        cache_tokens = record.prompt_ids + generated_tokens
        append_global_suffix_cache(args.global_cache or None, cache_tokens)
        if args.global_cache:
            global_index.update(cache_tokens)

    summary = summarize_jsonl(output_path) | {
        "method": "suffix_decoding",
        "num_prompts": len(prompts),
        "output": str(output_path),
        "matches_baseline": divergence_count == 0,
        "diverged_prompts": divergence_count,
        "first_diverged_prompt_id": first_diverged_prompt_id,
        "token_count_mismatches": token_count_mismatches,
        "draft_len": args.draft_len,
        "max_tree_depth": args.max_tree_depth,
        "max_spec_factor": args.max_spec_factor,
        "min_token_prob": args.min_token_prob,
        "max_model_len": args.max_model_len,
        "global_cache": args.global_cache,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

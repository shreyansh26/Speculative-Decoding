from __future__ import annotations

"""
Suffix decoding uses a bounded suffix-frequency index over prompt and generated
history to propose likely continuations without a trained draft model.
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import torch

from common.metrics import SpecDecodeStats, write_jsonl_record
from common.qwen3 import Qwen3ForCausalLM
from common.sampling import autoregressive_generate
from common.tokenizer import load_prompts, load_tokenizer
from common.verification import greedy_verify
from methods.draft_model.training.train import parse_dtype


class SuffixIndex:
    def __init__(self, max_tree_depth: int = 24) -> None:
        self.max_tree_depth = max_tree_depth
        self.counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)

    def update(self, tokens: Sequence[int]) -> None:
        values = list(tokens)
        for start in range(len(values) - 1):
            max_depth = min(self.max_tree_depth, len(values) - start - 1)
            for depth in range(1, max_depth + 1):
                key = tuple(values[start : start + depth])
                next_token = values[start + depth]
                self.counts[key][next_token] += 1

    def next_token_distribution(self, suffix: Sequence[int]) -> tuple[tuple[int, ...], Counter[int]]:
        values = list(suffix)
        max_depth = min(self.max_tree_depth, len(values))
        for depth in range(max_depth, 0, -1):
            key = tuple(values[-depth:])
            if key in self.counts:
                return key, self.counts[key]
        return (), Counter()


def propose_suffix_tokens(
    index: SuffixIndex,
    history_ids: Sequence[int],
    draft_len: int,
    max_spec_factor: float,
    min_token_prob: float,
) -> list[int]:
    proposals: list[int] = []
    working_history = list(history_ids)
    while len(proposals) < draft_len:
        match, next_counts = index.next_token_distribution(working_history)
        if not match or not next_counts:
            break
        match_len = len(match)
        max_spec = min(
            draft_len,
            max(1, int(max_spec_factor * match_len)),
            index.max_tree_depth - match_len + 1,
        )
        if len(proposals) >= max_spec:
            break
        total = sum(next_counts.values())
        token, count = next_counts.most_common(1)[0]
        probability = count / total
        if probability < min_token_prob:
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
        handle.write(json.dumps({"tokens": list(tokens)}) + "\n")


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
    per_request_index = SuffixIndex(max_tree_depth=max_tree_depth)
    if prompt_ids:
        per_request_index.update(prompt_ids)
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
        combined_index = SuffixIndex(max_tree_depth=max_tree_depth)
        combined_index.counts.update(per_request_index.counts)
        if global_index is not None:
            for key, counter in global_index.counts.items():
                combined_index.counts[key].update(counter)
        draft_ids = propose_suffix_tokens(
            combined_index,
            history_ids=prefix,
            draft_len=requested,
            max_spec_factor=max_spec_factor,
            min_token_prob=min_token_prob,
        )
        result = greedy_verify(model, prefix_ids=prefix, draft_ids=draft_ids)
        counters["speculation_steps"] += 1
        counters["target_forwards"] += 1
        counters["draft_forwards"] += 1 if draft_ids else 0
        counters["proposed_draft_tokens"] += result.proposed_draft_tokens
        counters["accepted_draft_tokens"] += result.accepted_draft_tokens
        for token in result.emitted_ids:
            if len(generated) >= max_new_tokens:
                break
            prefix.append(token)
            generated.append(token)
            per_request_index.update(prefix)
            if eos_token_id is not None and token == eos_token_id:
                return generated, counters
    return generated, counters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suffix decoding speculative inference.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--draft-len", type=int, default=32)
    parser.add_argument("--max-tree-depth", type=int, default=24)
    parser.add_argument("--max-spec-factor", type=float, default=1.0)
    parser.add_argument("--min-token-prob", type=float, default=0.1)
    parser.add_argument("--global-cache", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.temperature != 0.0 or args.top_p != 1.0:
        raise ValueError("phase-6 suffix decoding currently supports greedy decoding only")

    torch.manual_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    prompts = load_prompts(args.prompts, tokenizer=tokenizer)
    model = Qwen3ForCausalLM.from_pretrained(args.model_path, device=args.device, dtype=dtype)
    global_index = load_global_suffix_index(args.global_cache or None, max_tree_depth=args.max_tree_depth)

    compile_enabled = False
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
        compile_enabled = True

    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()

    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt.prompt, add_special_tokens=False)
        baseline_tokens: list[int] = []
        baseline_time_s = 0.0
        if not args.skip_baseline:
            wall_start = time.perf_counter()
            baseline_tokens = autoregressive_generate(model, prompt_ids, args.max_new_tokens, 0.0, 1.0, tokenizer.eos_token_id)
            baseline_time_s = time.perf_counter() - wall_start
        wall_start = time.perf_counter()
        generated_tokens, counters = run_suffix_speculative_decode(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            draft_len=args.draft_len,
            max_tree_depth=args.max_tree_depth,
            max_spec_factor=args.max_spec_factor,
            min_token_prob=args.min_token_prob,
            global_index=global_index,
            eos_token_id=tokenizer.eos_token_id,
        )
        method_time_s = time.perf_counter() - wall_start
        if baseline_tokens and generated_tokens != baseline_tokens:
            raise AssertionError("suffix decoding greedy output diverged from baseline")
        stats = SpecDecodeStats(
            method="suffix_decoding",
            model=args.model_path,
            prompt_id=prompt.prompt_id,
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(generated_tokens),
            generated_text=tokenizer.decode(generated_tokens, skip_special_tokens=True),
            temperature=0.0,
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
            cuda_graphs_reason=(
                "disabled: suffix index updates are dynamic"
                if args.cuda_graphs
                else "disabled"
            ),
            seed=args.seed,
        )
        write_jsonl_record(output_path, stats.to_record())
        append_global_suffix_cache(args.global_cache or None, prompt_ids + generated_tokens)


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
N-gram speculative decoding uses exact suffix matching over prompt+history to
propose draft tokens, then relies on the shared verifier for correctness.
"""

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from common.metrics import SpecDecodeStats, write_jsonl_record
from common.qwen3 import Qwen3ForCausalLM
from common.sampling import autoregressive_generate
from common.tokenizer import load_prompts, load_tokenizer
from common.verification import run_greedy_speculative_decode


def find_ngram_draft(
    history_ids: Sequence[int],
    draft_len: int,
    prompt_lookup_min: int,
    prompt_lookup_max: int,
) -> list[int]:
    if draft_len <= 0:
        return []
    if prompt_lookup_min <= 0:
        raise ValueError("prompt_lookup_min must be positive")
    if prompt_lookup_max < prompt_lookup_min:
        raise ValueError("prompt_lookup_max must be >= prompt_lookup_min")

    history = list(history_ids)
    if len(history) < prompt_lookup_min:
        return []

    max_window = min(prompt_lookup_max, len(history))
    for ngram_size in range(max_window, prompt_lookup_min - 1, -1):
        suffix_start = len(history) - ngram_size
        suffix = history[suffix_start:]
        latest_start = len(history) - ngram_size - 1
        for start in range(latest_start, -1, -1):
            if history[start : start + ngram_size] != suffix:
                continue
            candidate_start = start + ngram_size
            candidate = history[candidate_start : candidate_start + draft_len]
            if candidate:
                return candidate
    return []


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


def run_ngram_speculative_decode(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    draft_len: int,
    prompt_lookup_min: int,
    prompt_lookup_max: int,
) -> tuple[list[int], dict[str, int]]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="N-gram speculative decoding inference.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--draft-len", type=int, default=5)
    parser.add_argument("--prompt-lookup-min", type=int, default=2)
    parser.add_argument("--prompt-lookup-max", type=int, default=5)
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
    if args.temperature != 0.0:
        raise ValueError("phase-1 n-gram inference supports only greedy temperature=0.0")
    if args.top_p != 1.0:
        raise ValueError("phase-1 n-gram inference expects top_p=1.0")

    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(args.model_path)
    prompts = load_prompts(args.prompts, tokenizer=tokenizer)

    dtype = parse_dtype(args.dtype)
    model = Qwen3ForCausalLM.from_pretrained(args.model_path, device=args.device, dtype=dtype)
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
            start = torch.cuda.Event(enable_timing=True) if args.device.startswith("cuda") else None
            end = torch.cuda.Event(enable_timing=True) if args.device.startswith("cuda") else None
            if start is not None and end is not None:
                start.record()
            else:
                baseline_start = torch.cuda.default_generators[0].device.index if False else None
            import time

            wall_start = time.perf_counter()
            baseline_tokens = autoregressive_generate(
                model=model,
                prompt_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
            )
            baseline_time_s = time.perf_counter() - wall_start
            if start is not None and end is not None:
                end.record()
                torch.cuda.synchronize()

        import time

        wall_start = time.perf_counter()
        generated_tokens, counters = run_ngram_speculative_decode(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            draft_len=args.draft_len,
            prompt_lookup_min=args.prompt_lookup_min,
            prompt_lookup_max=args.prompt_lookup_max,
        )
        method_time_s = time.perf_counter() - wall_start

        if baseline_tokens and generated_tokens != baseline_tokens:
            raise AssertionError(
                f"greedy speculative output diverged for {prompt.prompt_id}:"
                f" baseline={baseline_tokens} speculative={generated_tokens}"
            )

        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        stats = SpecDecodeStats(
            method="ngram",
            model=args.model_path,
            prompt_id=prompt.prompt_id,
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(generated_tokens),
            generated_text=generated_text,
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
            cuda_graphs_reason=(
                "disabled: dynamic history-length prompt lookup is not graph-safe"
                if args.cuda_graphs
                else "disabled"
            ),
            seed=args.seed,
        )
        write_jsonl_record(output_path, stats.to_record())

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "method": "ngram",
                "num_prompts": len(prompts),
                "output": str(output_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

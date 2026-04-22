from __future__ import annotations

"""
PARD speculative decoding proposes K future tokens in one draft forward using
mask-token slots, then verifies them with the target model.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import torch

from common.metrics import SpecDecodeStats, write_jsonl_record
from common.qwen3 import Qwen3ForCausalLM
from common.sampling import autoregressive_generate
from common.tokenizer import load_prompts, load_tokenizer
from common.verification import greedy_verify
from methods.draft_model.training.train import parse_dtype
from methods.parallel_draft_models.training.train import load_pard_checkpoint


def propose_parallel_draft_tokens(
    draft_model: torch.nn.Module,
    prefix_ids: Sequence[int],
    draft_len: int,
    mask_token_id: int,
) -> list[int]:
    if draft_len <= 0:
        return []
    device = getattr(draft_model, "device", torch.device("cpu"))
    input_ids = torch.tensor(
        [list(prefix_ids) + [mask_token_id] * draft_len],
        dtype=torch.long,
        device=device,
    )
    logits = draft_model(input_ids).logits[0, -draft_len:, :]
    return [int(torch.argmax(logits[idx], dim=-1).item()) for idx in range(draft_len)]


def run_pard_speculative_decode(
    target_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    draft_len: int,
    mask_token_id: int,
    eos_token_id: int | None = None,
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
            prefix.append(token)
            generated.append(token)
            if eos_token_id is not None and token == eos_token_id:
                return generated, counters
    return generated, counters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PARD speculative decoding inference.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--draft-len", type=int, default=4)
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
        raise ValueError("phase-4 PARD inference currently supports greedy decoding only")

    torch.manual_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    prompts = load_prompts(args.prompts, tokenizer=tokenizer)
    target_model = Qwen3ForCausalLM.from_pretrained(args.model_path, device=args.device, dtype=dtype)
    draft_model, metadata = load_pard_checkpoint(args.checkpoint_path, device=args.device)
    mask_token_id = int(metadata["mask_token_id"])

    compile_enabled = False
    if args.compile:
        target_model = torch.compile(target_model, mode="reduce-overhead")
        draft_model = torch.compile(draft_model, mode="reduce-overhead")
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
            baseline_tokens = autoregressive_generate(target_model, prompt_ids, args.max_new_tokens, 0.0, 1.0, tokenizer.eos_token_id)
            baseline_time_s = time.perf_counter() - wall_start
        wall_start = time.perf_counter()
        generated_tokens, counters = run_pard_speculative_decode(
            target_model=target_model,
            draft_model=draft_model,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            draft_len=args.draft_len,
            mask_token_id=mask_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        method_time_s = time.perf_counter() - wall_start
        if baseline_tokens and generated_tokens != baseline_tokens:
            raise AssertionError("PARD greedy output diverged from baseline")
        stats = SpecDecodeStats(
            method="parallel_draft_models",
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
                "disabled: dynamic prefix lengths and mask slots are not graph-safe"
                if args.cuda_graphs
                else "disabled"
            ),
            seed=args.seed,
        )
        write_jsonl_record(output_path, stats.to_record())

    (output_path.with_suffix(".summary.json")).write_text(
        json.dumps({"method": "parallel_draft_models", "num_prompts": len(prompts), "output": str(output_path)}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

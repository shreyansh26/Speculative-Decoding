from __future__ import annotations

"""
Minimal EAGLE-3 inference: fuse low/mid/high target states, sequentially draft
tokens with the lightweight drafter, then verify against the target model.
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
from common.verification import PrefixState, greedy_verify_with_state, prefill_prefix
from methods.draft_model.training.train import parse_dtype
from methods.eagle3.training.train import (
    Eagle3Config,
    Eagle3Drafter,
    fuse_hidden_states,
    load_eagle3_checkpoint,
)


def propose_eagle3_tokens(
    drafter: Eagle3Drafter,
    target_state: PrefixState,
    prev_token_id: int,
    selected_layers: Sequence[int],
    draft_len: int,
) -> list[int]:
    if draft_len <= 0:
        return []
    if target_state.hidden_states is None:
        raise ValueError("target hidden states are required for EAGLE-3 proposals")
    fused = drafter.feature_fuser(fuse_hidden_states(target_state.hidden_states, selected_layers)[:, -1, :])
    state = drafter.init_state(fused)
    prev_token = torch.tensor([prev_token_id], device=fused.device, dtype=torch.long)
    proposals: list[int] = []
    for _ in range(draft_len):
        logits, state = drafter.forward_step(fused, prev_token, state)
        prev_token = torch.argmax(logits, dim=-1)
        proposals.append(int(prev_token.item()))
    return proposals


def run_eagle3_speculative_decode(
    target_model: torch.nn.Module,
    drafter: Eagle3Drafter,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    selected_layers: Sequence[int],
    draft_len: int,
    eos_token_id: int | None = None,
) -> tuple[list[int], dict[str, int]]:
    with torch.inference_mode():
        generated: list[int] = []
        target_state = prefill_prefix(
            target_model,
            prompt_ids,
            hidden_state_indices=selected_layers,
        )
        prev_token_id = int(prompt_ids[-1])
        counters = {
            "speculation_steps": 0,
            "target_forwards": 0,
            "draft_forwards": 0,
            "proposed_draft_tokens": 0,
            "accepted_draft_tokens": 0,
        }
        while len(generated) < max_new_tokens:
            requested = min(draft_len, max_new_tokens - len(generated))
            draft_ids = propose_eagle3_tokens(drafter, target_state, prev_token_id, selected_layers, requested)
            result, target_state = greedy_verify_with_state(
                target_model,
                target_state,
                draft_ids,
                hidden_state_indices=selected_layers,
            )
            counters["speculation_steps"] += 1
            counters["target_forwards"] += len(result.emitted_ids)
            counters["draft_forwards"] += len(draft_ids)
            counters["proposed_draft_tokens"] += result.proposed_draft_tokens
            counters["accepted_draft_tokens"] += result.accepted_draft_tokens
            for token in result.emitted_ids:
                if len(generated) >= max_new_tokens:
                    break
                prev_token_id = int(token)
                generated.append(token)
                if eos_token_id is not None and token == eos_token_id:
                    return generated, counters
    return generated, counters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EAGLE-3 speculative decoding inference.")
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
    parser.add_argument("--export-vllm-config", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.temperature != 0.0 or args.top_p != 1.0:
        raise ValueError("phase-5 EAGLE-3 inference currently supports greedy decoding only")

    torch.manual_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    prompts = load_prompts(args.prompts, tokenizer=tokenizer)
    target_model = Qwen3ForCausalLM.from_pretrained(args.model_path, device=args.device, dtype=dtype)
    drafter, metadata = load_eagle3_checkpoint(args.checkpoint_path, device=args.device)
    if args.device != "cpu":
        drafter = drafter.to(dtype=dtype)
    config = Eagle3Config(**metadata["eagle3_config"])

    compile_enabled = False
    if args.compile:
        target_model = torch.compile(target_model, mode="reduce-overhead")
        drafter = torch.compile(drafter, mode="reduce-overhead")
        compile_enabled = True

    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()

    def synchronize() -> None:
        if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(torch.device(args.device))

    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt.prompt, add_special_tokens=False)
        baseline_tokens: list[int] = []
        baseline_time_s = 0.0
        if not args.skip_baseline:
            synchronize()
            wall_start = time.perf_counter()
            baseline_tokens = autoregressive_generate(target_model, prompt_ids, args.max_new_tokens, 0.0, 1.0, tokenizer.eos_token_id)
            synchronize()
            baseline_time_s = time.perf_counter() - wall_start
        synchronize()
        wall_start = time.perf_counter()
        generated_tokens, counters = run_eagle3_speculative_decode(
            target_model=target_model,
            drafter=drafter,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            selected_layers=config.selected_layers,
            draft_len=args.draft_len,
            eos_token_id=tokenizer.eos_token_id,
        )
        synchronize()
        method_time_s = time.perf_counter() - wall_start
        stats = SpecDecodeStats(
            method="eagle3",
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
                "disabled: target-state fusion is not graph-safe in phase 5"
                if args.cuda_graphs
                else "disabled"
            ),
            seed=args.seed,
        )
        write_jsonl_record(output_path, stats.to_record())

    if args.export_vllm_config:
        Path(args.export_vllm_config).write_text(
            json.dumps(
                {
                    "model": args.checkpoint_path,
                    "num_speculative_tokens": args.draft_len,
                    "method": "eagle3",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

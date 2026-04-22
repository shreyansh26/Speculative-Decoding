from __future__ import annotations

"""
Draft-target speculative decoding uses a compact autoregressive draft model to
propose K tokens, then verifies that chain against the target in one pass.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import torch

from common.metrics import SpecDecodeStats, write_jsonl_record
from common.qwen3 import Qwen3ForCausalLM
from common.sampling import autoregressive_generate, sample_from_logits
from common.tokenizer import load_prompts, load_tokenizer
from common.verification import (
    PrefixState,
    VerificationResult,
    advance_prefix_state,
    greedy_verify_with_state,
    prefill_prefix,
    state_after_decoded_tokens,
)
from methods.draft_model.training.train import load_draft_checkpoint, parse_dtype


def propose_draft_tokens(
    draft_model: torch.nn.Module,
    draft_state: PrefixState,
    draft_len: int,
    temperature: float,
    top_p: float,
    eos_token_id: int | None,
) -> tuple[list[int], list[PrefixState]]:
    if draft_len <= 0:
        return [], []
    working_state = draft_state
    proposals: list[int] = []
    proposal_states: list[PrefixState] = []
    for _ in range(draft_len):
        next_token = sample_from_logits(
            working_state.last_logits,
            temperature=temperature,
            top_p=top_p,
        )
        proposals.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
        if len(proposals) >= draft_len:
            break
        working_state = advance_prefix_state(draft_model, working_state, next_token)
        proposal_states.append(working_state)
    return proposals, proposal_states


def run_draft_model_speculative_decode(
    target_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    draft_len: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> tuple[list[int], dict[str, int]]:
    with torch.inference_mode():
        generated: list[int] = []
        target_state = prefill_prefix(target_model, prompt_ids)
        draft_state = prefill_prefix(draft_model, prompt_ids)
        pending_target_token: int | None = None
        use_deferred_target_state = bool(getattr(target_state, "cache", None)) and hasattr(target_model, "decode_many")
        counters = {
            "speculation_steps": 0,
            "target_forwards": 0,
            "draft_forwards": 0,
            "proposed_draft_tokens": 0,
            "accepted_draft_tokens": 0,
        }

        while len(generated) < max_new_tokens:
            requested = min(draft_len, max_new_tokens - len(generated))
            draft_ids, proposal_states = propose_draft_tokens(
                draft_model=draft_model,
                draft_state=draft_state,
                draft_len=requested,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=eos_token_id,
            )
            if use_deferred_target_state:
                result, target_state, pending_target_token, target_forward_calls = verify_draft_model_chunk(
                    target_model=target_model,
                    target_state=target_state,
                    draft_ids=draft_ids,
                    pending_target_token=pending_target_token,
                )
            else:
                result, target_state = greedy_verify_with_state(target_model, target_state, draft_ids)
                pending_target_token = None
                target_forward_calls = 1 if result.mismatch_index == 0 else (2 if draft_ids else 1)
            counters["speculation_steps"] += 1
            counters["target_forwards"] += target_forward_calls
            counters["proposed_draft_tokens"] += result.proposed_draft_tokens
            counters["accepted_draft_tokens"] += result.accepted_draft_tokens

            accepted_count = result.accepted_draft_tokens
            draft_forward_calls = len(proposal_states)
            if accepted_count == 0:
                current_draft_state = draft_state
            elif accepted_count < len(draft_ids):
                current_draft_state = proposal_states[accepted_count - 1]
            else:
                if proposal_states:
                    current_draft_state = advance_prefix_state(draft_model, proposal_states[-1], draft_ids[-1])
                else:
                    current_draft_state = advance_prefix_state(draft_model, draft_state, draft_ids[-1])
                draft_forward_calls += 1

            for token in result.emitted_ids[accepted_count:]:
                current_draft_state = advance_prefix_state(draft_model, current_draft_state, token)
                draft_forward_calls += 1

            counters["draft_forwards"] += draft_forward_calls
            draft_state = current_draft_state

            for token in result.emitted_ids:
                if len(generated) >= max_new_tokens:
                    break
                generated.append(token)
                if eos_token_id is not None and token == eos_token_id:
                    return generated, counters

    return generated, counters


def verify_draft_model_chunk(
    target_model: torch.nn.Module,
    target_state: PrefixState,
    draft_ids: Sequence[int],
    pending_target_token: int | None,
) -> tuple[VerificationResult, PrefixState, int, int]:
    device = getattr(target_model, "device", torch.device("cpu"))
    accepted_count = 0
    accepted_ids: list[int] = []

    if pending_target_token is None:
        first_prediction = int(torch.argmax(target_state.last_logits).item())
        if draft_ids and first_prediction != int(draft_ids[0]):
            return (
                VerificationResult(
                    accepted_ids=[],
                    emitted_ids=[first_prediction],
                    target_predictions=[first_prediction],
                    proposed_draft_tokens=len(draft_ids),
                    accepted_draft_tokens=0,
                    mismatch_index=0,
                    bonus_token=None,
                ),
                target_state,
                first_prediction,
                0,
            )

        if draft_ids:
            verify_ids = list(draft_ids)
            output = target_model.decode_many(
                torch.tensor([verify_ids], dtype=torch.long, device=device),
                cache=target_state.cache,
            )
            predictions = [first_prediction] + [int(token) for token in torch.argmax(output.logits[0], dim=-1).tolist()]
            target_forward_calls = 1
        else:
            output = None
            predictions = [first_prediction]
            target_forward_calls = 0
        consumed_prefix_ids: list[int] = []
    else:
        verify_ids = [int(pending_target_token)] + [int(token) for token in draft_ids]
        output = target_model.decode_many(
            torch.tensor([verify_ids], dtype=torch.long, device=device),
            cache=target_state.cache,
        )
        predictions = [int(token) for token in torch.argmax(output.logits[0], dim=-1).tolist()]
        target_forward_calls = 1
        consumed_prefix_ids = [int(pending_target_token)]

    for index, draft_token in enumerate(draft_ids):
        if predictions[index] != int(draft_token):
            break
        accepted_count += 1

    accepted_ids = [int(token) for token in draft_ids[:accepted_count]]
    next_token = int(predictions[accepted_count])
    consumed_ids = consumed_prefix_ids + accepted_ids
    next_state = (
        target_state if not consumed_ids else state_after_decoded_tokens(target_state, output, consumed_ids)
    )
    mismatch_index = None if accepted_count == len(draft_ids) else accepted_count
    bonus_token = next_token if mismatch_index is None else None
    return (
        VerificationResult(
            accepted_ids=accepted_ids,
            emitted_ids=accepted_ids + [next_token],
            target_predictions=predictions[: len(draft_ids) + 1],
            proposed_draft_tokens=len(draft_ids),
            accepted_draft_tokens=accepted_count,
            mismatch_index=mismatch_index,
            bonus_token=bonus_token,
        ),
        next_state,
        next_token,
        target_forward_calls,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draft-model speculative decoding inference.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--draft-len", type=int, default=5)
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
        raise ValueError("phase-2 draft-model inference currently supports only greedy decoding")
    if args.top_p != 1.0:
        raise ValueError("phase-2 draft-model inference expects top_p=1.0")

    torch.manual_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    prompts = load_prompts(args.prompts, tokenizer=tokenizer)

    target_model = Qwen3ForCausalLM.from_pretrained(args.model_path, device=args.device, dtype=dtype)
    draft_model = load_draft_checkpoint(args.checkpoint_path, device=args.device, dtype=dtype)

    compile_enabled = False
    if args.compile:
        target_model = torch.compile(target_model, mode="reduce-overhead")
        draft_model = torch.compile(draft_model, mode="reduce-overhead")
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
            baseline_tokens = autoregressive_generate(
                model=target_model,
                prompt_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
            )
            synchronize()
            baseline_time_s = time.perf_counter() - wall_start

        synchronize()
        wall_start = time.perf_counter()
        generated_tokens, counters = run_draft_model_speculative_decode(
            target_model=target_model,
            draft_model=draft_model,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            draft_len=args.draft_len,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
        )
        synchronize()
        method_time_s = time.perf_counter() - wall_start

        stats = SpecDecodeStats(
            method="draft_model",
            model=args.model_path,
            prompt_id=prompt.prompt_id,
            prompt_tokens=len(prompt_ids),
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
            cuda_graphs_reason=(
                "disabled: dual-model dynamic decode loop is not graph-safe"
                if args.cuda_graphs
                else "disabled"
            ),
            seed=args.seed,
        )
        write_jsonl_record(output_path, stats.to_record())

    (output_path.with_suffix(".summary.json")).write_text(
        json.dumps(
            {
                "method": "draft_model",
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

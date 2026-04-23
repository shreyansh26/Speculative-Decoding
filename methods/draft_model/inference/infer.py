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

from common.metrics import SpecDecodeStats, summarize_jsonl, write_jsonl_record
from common.qwen3 import Qwen3ForCausalLM
from common.sampling import autoregressive_generate
from common.tokenizer import load_prompts, load_tokenizer
from common.verification import (
    VerificationResult,
    advance_prefix_state,
    greedy_verify_with_state,
    prefill_prefix,
    state_after_decoded_tokens,
)
from methods.draft_model.training.train import load_draft_checkpoint, parse_dtype


DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CHECKPOINT_PATH = "checkpoints/draft_model_qwen25_05b_2ep"
DEFAULT_PROMPTS_PATH = "data/ultrachat_eval_50_short_trunc512.jsonl"
DEFAULT_MAX_NEW_TOKENS = 16
DEFAULT_DRAFT_LEN = 4
DEFAULT_WARMUP_PROMPTS = 2


def propose_draft_tokens(
    draft_model: torch.nn.Module,
    draft_state,
    requested: int,
    eos_token_id: int | None = None,
) -> tuple[list[int], list[object]]:
    proposals: list[int] = []
    proposal_states: list[object] = []
    working_state = draft_state
    for _ in range(requested):
        token = int(torch.argmax(working_state.last_logits).item())
        proposals.append(token)
        working_state = advance_prefix_state(draft_model, working_state, token)
        proposal_states.append(working_state)
        if eos_token_id is not None and token == eos_token_id:
            break
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
    fast_verify: bool = False,
) -> tuple[list[int], dict[str, int]]:
    def verify_fn(
        model: torch.nn.Module,
        state,
        draft_ids: Sequence[int],
    ) -> tuple[VerificationResult, object]:
        if not fast_verify:
            return greedy_verify_with_state(model, state, draft_ids)
        if hasattr(model, "decode_many") and state.cache:
            first_prediction = int(torch.argmax(state.last_logits).item())
            if draft_ids and first_prediction != int(draft_ids[0]):
                updated_state = advance_prefix_state(model, state, first_prediction)
                return VerificationResult(
                    accepted_ids=[],
                    emitted_ids=[first_prediction],
                    target_predictions=[first_prediction],
                    proposed_draft_tokens=len(draft_ids),
                    accepted_draft_tokens=0,
                    mismatch_index=0,
                    bonus_token=None,
                ), updated_state

            device = getattr(model, "device", torch.device("cpu"))
            if draft_ids:
                draft_tensor = torch.tensor([list(draft_ids)], dtype=torch.long, device=device)
                output = model.decode_many(draft_tensor, cache=state.cache)
                prediction_tail = torch.argmax(output.logits[0], dim=-1).tolist()
            else:
                output = None
                prediction_tail = []

            predictions = [first_prediction] + [int(token) for token in prediction_tail]
            accepted_count = 0
            for index, draft_token in enumerate(draft_ids):
                if predictions[index] != int(draft_token):
                    break
                accepted_count += 1

            accepted_ids = [int(token) for token in draft_ids[:accepted_count]]
            if accepted_count < len(draft_ids):
                accepted_state = (
                    state if accepted_count == 0 else state_after_decoded_tokens(state, output, accepted_ids)
                )
                mismatch_token = int(predictions[accepted_count])
                updated_state = advance_prefix_state(model, accepted_state, mismatch_token)
                return VerificationResult(
                    accepted_ids=accepted_ids,
                    emitted_ids=accepted_ids + [mismatch_token],
                    target_predictions=predictions[: accepted_count + 1],
                    proposed_draft_tokens=len(draft_ids),
                    accepted_draft_tokens=accepted_count,
                    mismatch_index=accepted_count,
                    bonus_token=None,
                ), updated_state

            accepted_state = state if not accepted_ids else state_after_decoded_tokens(state, output, accepted_ids)
            bonus_token = int(predictions[-1])
            updated_state = advance_prefix_state(model, accepted_state, bonus_token)
            return VerificationResult(
                accepted_ids=accepted_ids,
                emitted_ids=accepted_ids + [bonus_token],
                target_predictions=predictions,
                proposed_draft_tokens=len(draft_ids),
                accepted_draft_tokens=accepted_count,
                mismatch_index=None,
                bonus_token=bonus_token,
            ), updated_state

        return greedy_verify_with_state(model, state, draft_ids)

    target_state = prefill_prefix(target_model, prompt_ids)
    draft_state = prefill_prefix(draft_model, prompt_ids)
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
        draft_ids, proposal_states = propose_draft_tokens(
            draft_model=draft_model,
            draft_state=draft_state,
            requested=requested,
            eos_token_id=eos_token_id,
        )
        result, target_state = verify_fn(target_model, target_state, draft_ids)
        counters["speculation_steps"] += 1
        counters["target_forwards"] += result.accepted_draft_tokens + 1
        counters["draft_forwards"] += len(draft_ids)
        counters["proposed_draft_tokens"] += result.proposed_draft_tokens
        counters["accepted_draft_tokens"] += result.accepted_draft_tokens

        accepted_count = result.accepted_draft_tokens
        if accepted_count > 0:
            draft_state = proposal_states[accepted_count - 1]

        for token in result.emitted_ids[accepted_count:]:
            if len(generated) >= max_new_tokens:
                break
            draft_state = advance_prefix_state(draft_model, draft_state, token)
            counters["draft_forwards"] += 1

        for token in result.emitted_ids:
            if len(generated) >= max_new_tokens:
                break
            generated.append(token)
            if eos_token_id is not None and token == eos_token_id:
                return generated, counters

    return generated, counters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draft-model speculative decoding inference.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--output", default="")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--draft-len", type=int, default=DEFAULT_DRAFT_LEN)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-target", action="store_true")
    parser.add_argument("--compile-draft", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--warmup-prompts", type=int, default=DEFAULT_WARMUP_PROMPTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--allow-divergence", action="store_true")
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

    compile_target = args.compile or args.compile_target
    compile_draft = args.compile or args.compile_draft
    compile_enabled = compile_target or compile_draft
    if compile_target:
        target_model = torch.compile(target_model, mode="reduce-overhead")
    if compile_draft:
        draft_model = torch.compile(draft_model, mode="reduce-overhead")

    output_path = (
        Path(args.output)
        if args.output
        else Path(f"runs/draft_model_qwen25_05b_2ep_len{args.draft_len}.jsonl")
    )
    if output_path.exists():
        output_path.unlink()

    def synchronize() -> None:
        if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(torch.device(args.device))

    warmup_count = min(max(args.warmup_prompts, 0), len(prompts))
    for prompt in prompts[:warmup_count]:
        prompt_ids = tokenizer.encode(prompt.prompt, add_special_tokens=False)
        if not args.skip_baseline:
            autoregressive_generate(
                model=target_model,
                prompt_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
            )
        run_draft_model_speculative_decode(
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

    divergence_count = 0
    first_diverged_prompt_id = ""
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
            fast_verify=args.allow_divergence,
        )
        synchronize()
        method_time_s = time.perf_counter() - wall_start
        matches_baseline = True
        if not args.skip_baseline and generated_tokens != baseline_tokens:
            matches_baseline = False
            divergence_count += 1
            if not first_diverged_prompt_id:
                first_diverged_prompt_id = prompt.prompt_id
        if not args.skip_baseline and generated_tokens != baseline_tokens and not args.allow_divergence:
            raise RuntimeError(
                f"greedy speculative output diverged for {prompt.prompt_id}:"
                f" baseline={baseline_tokens} speculative={generated_tokens}"
            )

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
        record = stats.to_record()
        record["matches_baseline"] = matches_baseline
        write_jsonl_record(output_path, record)

    (output_path.with_suffix(".summary.json")).write_text(
        json.dumps(
            summarize_jsonl(output_path)
            | {
                "method": "draft_model",
                "num_prompts": len(prompts),
                "output": str(output_path),
                "matches_baseline": divergence_count == 0,
                "diverged_prompts": divergence_count,
                "first_diverged_prompt_id": first_diverged_prompt_id,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

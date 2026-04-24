from __future__ import annotations

"""
Minimal EAGLE-3 inference: fuse low/mid/high target states, sequentially draft
tokens with the Eagle transformer head, then verify against the target model.
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
from common.tokenizer import render_prompt, load_tokenizer
from common.verification import (
    PrefixState,
    VerificationResult,
    advance_prefix_state,
    greedy_verify_with_state,
    prefill_prefix,
    state_after_decoded_tokens,
)
from methods.draft_model.training.train import parse_dtype
from methods.eagle3.training.train import (
    Eagle3Config,
    Eagle3Drafter,
    fuse_hidden_states,
    load_eagle3_checkpoint,
)


DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CHECKPOINT_PATH = "checkpoints/eagle3_qwen25_7b"
DEFAULT_PROMPTS_PATH = "data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_DRAFT_LEN = 3
DEFAULT_WARMUP_PROMPTS = 2


def load_prompt_records(path: str | Path, tokenizer) -> list[tuple[str, list[int], str]]:
    prompt_path = Path(path)
    records: list[tuple[str, list[int], str]] = []
    with prompt_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            prompt_id = str(sample.get("prompt_id", sample.get("id", f"prompt_{line_number:04d}")))
            if "prompt_ids" in sample:
                prompt_ids = [int(token_id) for token_id in sample["prompt_ids"]]
                prompt_text = str(sample.get("prompt", tokenizer.decode(prompt_ids, skip_special_tokens=False)))
            else:
                prompt_sample = render_prompt(tokenizer, sample)
                prompt_ids = tokenizer.encode(prompt_sample.prompt, add_special_tokens=False)
                prompt_text = prompt_sample.prompt
            records.append((prompt_id, prompt_ids, prompt_text))
    return records


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
    cache = None
    proposals: list[int] = []
    base_position = 0
    if target_state.cache and target_state.cache[0] is not None:
        base_position = target_state.cache[0].key.shape[2] - 1

    prev_token = torch.tensor([prev_token_id], device=fused.device, dtype=torch.long)

    for step in range(draft_len):
        position_ids = torch.tensor([[base_position + step]], device=fused.device, dtype=torch.long)
        logits, state, cache = drafter.forward_step(
            state,
            prev_token,
            cache=cache,
            position_ids=position_ids,
        )
        prev_token = torch.argmax(logits, dim=-1)
        proposals.append(int(prev_token.item()))
    return proposals


def verify_seeded_draft_with_state(
    model: torch.nn.Module,
    state: PrefixState,
    seed_token: int,
    draft_ids: Sequence[int],
    selected_layers: Sequence[int],
) -> tuple[list[int], PrefixState, int]:
    """Verify the target-seeded EAGLE proposal.

    The seed token is the target model's own next-token prediction and is not
    counted as a draft token. EAGLE proposes the following tokens.
    """
    device = getattr(model, "device", torch.device("cpu"))
    candidate_ids = [int(seed_token)] + [int(token) for token in draft_ids]
    if hasattr(model, "decode_many") and state.cache:
        candidate_tensor = torch.tensor([candidate_ids], dtype=torch.long, device=device)
        output = model.decode_many(
            candidate_tensor,
            cache=state.cache,
            output_hidden_states=True,
            hidden_state_indices=selected_layers,
        )
        predictions = [int(seed_token)] + [int(token) for token in torch.argmax(output.logits[0], dim=-1).tolist()]
        accepted_draft_count = 0
        for index, draft_token in enumerate(draft_ids):
            if predictions[index + 1] != int(draft_token):
                break
            accepted_draft_count += 1

        emitted = candidate_ids[: 1 + accepted_draft_count]
        if accepted_draft_count < len(draft_ids):
            mismatch_token = int(predictions[accepted_draft_count + 1])
            emitted.append(mismatch_token)
            accepted_state = state_after_decoded_tokens(
                state,
                output,
                candidate_ids[: 1 + accepted_draft_count],
            )
            updated_state = advance_prefix_state(
                model,
                accepted_state,
                mismatch_token,
                hidden_state_indices=selected_layers,
            )
            return emitted, updated_state, accepted_draft_count

        bonus_token = int(predictions[len(draft_ids) + 1]) if len(predictions) > len(draft_ids) + 1 else None
        accepted_state = state_after_decoded_tokens(state, output, candidate_ids)
        if bonus_token is not None:
            emitted.append(bonus_token)
            updated_state = advance_prefix_state(
                model,
                accepted_state,
                bonus_token,
                hidden_state_indices=selected_layers,
            )
        else:
            updated_state = accepted_state
        return emitted, updated_state, accepted_draft_count

    seed_state = advance_prefix_state(
        model,
        state,
        int(seed_token),
        hidden_state_indices=selected_layers,
    )
    result, updated_state = greedy_verify_with_state(
        model,
        seed_state,
        draft_ids,
        hidden_state_indices=selected_layers,
    )
    return [int(seed_token)] + result.emitted_ids, updated_state, result.accepted_draft_tokens


def run_eagle3_speculative_decode(
    target_model: torch.nn.Module,
    drafter: Eagle3Drafter,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    selected_layers: Sequence[int],
    draft_len: int,
    eos_token_id: int | None = None,
    fast_verify: bool = False,
) -> tuple[list[int], dict[str, int]]:
    def verify_fn(
        model: torch.nn.Module,
        state: PrefixState,
        draft_ids: Sequence[int],
        selected_layers_: Sequence[int],
    ) -> tuple[VerificationResult, PrefixState]:
        if not fast_verify:
            return greedy_verify_with_state(
                model,
                state,
                draft_ids,
                hidden_state_indices=selected_layers_,
            )
        if hasattr(model, "decode_many") and state.cache:
            first_prediction = int(torch.argmax(state.last_logits).item())
            if draft_ids and first_prediction != int(draft_ids[0]):
                updated_state = advance_prefix_state(
                    model,
                    state,
                    first_prediction,
                    hidden_state_indices=selected_layers_,
                )
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
            wants_hidden_states = selected_layers_ is not None
            if draft_ids:
                draft_tensor = torch.tensor([list(draft_ids)], dtype=torch.long, device=device)
                if wants_hidden_states:
                    output = model.decode_many(
                        draft_tensor,
                        cache=state.cache,
                        output_hidden_states=True,
                        hidden_state_indices=selected_layers_,
                    )
                else:
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
                updated_state = advance_prefix_state(
                    model,
                    accepted_state,
                    mismatch_token,
                    hidden_state_indices=selected_layers_,
                )
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
            updated_state = advance_prefix_state(
                model,
                accepted_state,
                bonus_token,
                hidden_state_indices=selected_layers_,
            )
            return VerificationResult(
                accepted_ids=accepted_ids,
                emitted_ids=accepted_ids + [bonus_token],
                target_predictions=predictions,
                proposed_draft_tokens=len(draft_ids),
                accepted_draft_tokens=accepted_count,
                mismatch_index=None,
                bonus_token=bonus_token,
            ), updated_state

        return greedy_verify_with_state(
            model,
            state,
            draft_ids,
            hidden_state_indices=selected_layers_,
        )

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
            remaining = max_new_tokens - len(generated)
            seed_token = int(torch.argmax(target_state.last_logits).item())
            requested = max(0, min(draft_len, remaining - 1))
            draft_ids = propose_eagle3_tokens(drafter, target_state, seed_token, selected_layers, requested)
            emitted_ids, target_state, accepted_count = verify_seeded_draft_with_state(
                target_model,
                target_state,
                seed_token,
                draft_ids,
                selected_layers,
            )
            counters["speculation_steps"] += 1
            counters["target_forwards"] += 1
            counters["draft_forwards"] += len(draft_ids)
            counters["proposed_draft_tokens"] += len(draft_ids)
            counters["accepted_draft_tokens"] += accepted_count
            for token in emitted_ids:
                if len(generated) >= max_new_tokens:
                    break
                prev_token_id = int(token)
                generated.append(token)
                if eos_token_id is not None and token == eos_token_id:
                    return generated, counters
    return generated, counters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EAGLE-3 speculative decoding inference.")
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
    parser.add_argument("--export-vllm-config", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.temperature != 0.0 or args.top_p != 1.0:
        raise ValueError("phase-5 EAGLE-3 inference currently supports greedy decoding only")

    torch.manual_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    prompts = load_prompt_records(args.prompts, tokenizer=tokenizer)
    target_model = Qwen3ForCausalLM.from_pretrained(args.model_path, device=args.device, dtype=dtype)
    drafter, metadata = load_eagle3_checkpoint(args.checkpoint_path, device=args.device)
    if args.device != "cpu":
        drafter = drafter.to(dtype=dtype)
    config = Eagle3Config(**metadata["eagle3_config"])

    compile_target = args.compile or args.compile_target
    compile_draft = args.compile or args.compile_draft
    compile_enabled = compile_target or compile_draft
    if compile_target:
        target_model = torch.compile(target_model, mode="reduce-overhead")
    if compile_draft:
        drafter = torch.compile(drafter, mode="reduce-overhead")

    output_path = (
        Path(args.output)
        if args.output
        else Path(f"runs/eagle3_len{args.draft_len}.jsonl")
    )
    if output_path.exists():
        output_path.unlink()

    def synchronize() -> None:
        if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(torch.device(args.device))

    warmup_count = min(max(args.warmup_prompts, 0), len(prompts))
    for _, prompt_ids, _ in prompts[:warmup_count]:
        if not args.skip_baseline:
            autoregressive_generate(
                target_model,
                prompt_ids,
                args.max_new_tokens,
                0.0,
                1.0,
                tokenizer.eos_token_id,
            )
        run_eagle3_speculative_decode(
            target_model=target_model,
            drafter=drafter,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            selected_layers=config.selected_layers,
            draft_len=args.draft_len,
            eos_token_id=tokenizer.eos_token_id,
            fast_verify=args.allow_divergence,
        )
        synchronize()

    divergence_count = 0
    first_diverged_prompt_id = ""
    for prompt_id, prompt_ids, _ in prompts:
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
            fast_verify=args.allow_divergence,
        )
        synchronize()
        method_time_s = time.perf_counter() - wall_start
        matches_baseline = True
        if not args.skip_baseline and generated_tokens != baseline_tokens:
            matches_baseline = False
            divergence_count += 1
            if not first_diverged_prompt_id:
                first_diverged_prompt_id = prompt_id
        stats = SpecDecodeStats(
            method="eagle3",
            model=args.model_path,
            prompt_id=prompt_id,
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
        record = stats.to_record()
        record["matches_baseline"] = matches_baseline
        write_jsonl_record(output_path, record)

    output_path.with_suffix(".summary.json").write_text(
        json.dumps(
            summarize_jsonl(output_path)
            | {
                "method": "eagle3",
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

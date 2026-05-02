"""Non-vLLM MTP speculative decoding."""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from common.metrics import SpecDecodeStats, summarize_jsonl, write_jsonl_record
from common.qwen3 import Qwen3ForCausalLM
from common.sampling import autoregressive_generate
from common.tokenizer import load_tokenizer, render_prompt
from common.verification import PrefixState, advance_prefix_state
from methods.draft_model.training.train import parse_dtype
from methods.mtp.training.train import MTPModel, load_mtp_checkpoint


DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CHECKPOINT_PATH = "checkpoints/mtp_qwen25_7b_eval100_steps1"
DEFAULT_PROMPTS_PATH = "data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_NUM_SPECULATIVE_STEPS = 1
DEFAULT_WARMUP_PROMPTS = 1


@dataclass(slots=True)
class MTPRuntimeState:
    target: PrefixState
    mtp_caches: list[object | None]
    mtp_last_hidden: list[torch.Tensor | None]


def load_prompt_records(path: str | Path, tokenizer) -> list[tuple[str, list[int]]]:
    records: list[tuple[str, list[int]]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
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
            records.append((prompt_id, prompt_ids))
    if not records:
        raise ValueError(f"{path} did not contain any prompts")
    return records


def prefill_mtp_runtime(
    target_model: torch.nn.Module,
    mtp: MTPModel,
    prompt_ids: Sequence[int],
    selected_layers: Sequence[int],
) -> MTPRuntimeState:
    device = getattr(target_model, "device", torch.device("cpu"))
    input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    output = target_model.prefill(
        input_ids,
        output_hidden_states=True,
        hidden_state_indices=selected_layers,
    )
    final_layer = mtp.config.target_num_hidden_layers
    full_target_hidden = output.hidden_states[final_layer]
    target_state = PrefixState(
        prefix_ids=list(prompt_ids),
        cache=output.cache or [],
        last_logits=output.logits[0, -1].detach(),
        hidden_states={final_layer: full_target_hidden[:, -1:, :].detach()},
    )
    mtp_caches: list[object | None] = [None] * mtp.config.num_nextn_predict_layers
    mtp_last_hidden: list[torch.Tensor | None] = [None] * mtp.config.num_nextn_predict_layers
    prefix_len = input_ids.shape[1]

    if prefix_len >= 2:
        position_ids = torch.arange(1, prefix_len, device=device, dtype=torch.long).unsqueeze(0)
        hidden0, cache0 = mtp.mtp_layers[0](
            mtp.embed_tokens(input_ids[:, 1:]),
            full_target_hidden[:, :-1, :],
            position_ids=position_ids,
        )
        mtp_caches[0] = cache0
        mtp_last_hidden[0] = hidden0[:, -1, :].detach()
    if mtp.config.num_nextn_predict_layers >= 2 and prefix_len >= 3 and mtp_last_hidden[0] is not None:
        position_ids = torch.arange(2, prefix_len, device=device, dtype=torch.long).unsqueeze(0)
        hidden1, cache1 = mtp.mtp_layers[1](
            mtp.embed_tokens(input_ids[:, 2:]),
            hidden0[:, :-1, :],
            position_ids=position_ids,
        )
        mtp_caches[1] = cache1
        mtp_last_hidden[1] = hidden1[:, -1, :].detach()

    return MTPRuntimeState(target=target_state, mtp_caches=mtp_caches, mtp_last_hidden=mtp_last_hidden)


def propose_mtp_tokens(
    mtp: MTPModel,
    target_state: PrefixState | MTPRuntimeState,
    seed_token: int,
    num_speculative_steps: int,
) -> list[int]:
    if num_speculative_steps <= 0:
        return []
    runtime = target_state if isinstance(target_state, MTPRuntimeState) else None
    prefix_state = runtime.target if runtime is not None else target_state
    if prefix_state.hidden_states is None:
        raise ValueError("target hidden states are required for MTP proposals")
    final_layer = mtp.config.target_num_hidden_layers
    previous_hidden = prefix_state.hidden_states[final_layer][:, -1, :]
    prefix_len = len(prefix_state.prefix_ids)
    input_token = int(seed_token)
    proposals: list[int] = []

    if runtime is None:
        for depth in range(num_speculative_steps):
            logits, previous_hidden, _ = mtp.forward_step(
                previous_hidden,
                input_token,
                depth=depth,
                position_id=prefix_len + depth,
            )
            input_token = int(torch.argmax(logits, dim=-1).item())
            proposals.append(input_token)
        return proposals

    logits, hidden0, _ = mtp.forward_step(
        previous_hidden,
        input_token,
        depth=0,
        position_id=prefix_len,
        cache=runtime.mtp_caches[0],
    )
    input_token = int(torch.argmax(logits, dim=-1).item())
    proposals.append(input_token)
    if num_speculative_steps == 1:
        return proposals

    if mtp.config.num_nextn_predict_layers < 2:
        return proposals
    working_cache = runtime.mtp_caches[1]
    if runtime.mtp_last_hidden[0] is not None:
        _, _, working_cache = mtp.forward_step(
            runtime.mtp_last_hidden[0],
            int(seed_token),
            depth=1,
            position_id=prefix_len,
            cache=working_cache,
        )
    logits, _, _ = mtp.forward_step(
        hidden0,
        input_token,
        depth=1,
        position_id=prefix_len + 1,
        cache=working_cache,
    )
    proposals.append(int(torch.argmax(logits, dim=-1).item()))
    return proposals[:num_speculative_steps]


def advance_runtime_state(
    target_model: torch.nn.Module,
    mtp: MTPModel,
    runtime: MTPRuntimeState,
    token_id: int,
    selected_layers: Sequence[int],
) -> None:
    if runtime.target.hidden_states is None:
        raise ValueError("target hidden states are required for MTP cache updates")
    final_layer = mtp.config.target_num_hidden_layers
    previous_target_hidden = runtime.target.hidden_states[final_layer][:, -1, :]
    position_id = len(runtime.target.prefix_ids)
    old_depth0_hidden = runtime.mtp_last_hidden[0]

    logits0, hidden0, cache0 = mtp.forward_step(
        previous_target_hidden,
        int(token_id),
        depth=0,
        position_id=position_id,
        cache=runtime.mtp_caches[0],
    )
    del logits0
    runtime.mtp_caches[0] = cache0
    runtime.mtp_last_hidden[0] = hidden0.detach()

    if mtp.config.num_nextn_predict_layers >= 2 and old_depth0_hidden is not None:
        logits1, hidden1, cache1 = mtp.forward_step(
            old_depth0_hidden,
            int(token_id),
            depth=1,
            position_id=position_id,
            cache=runtime.mtp_caches[1],
        )
        del logits1
        runtime.mtp_caches[1] = cache1
        runtime.mtp_last_hidden[1] = hidden1.detach()

    runtime.target = advance_prefix_state(
        target_model,
        runtime.target,
        int(token_id),
        hidden_state_indices=selected_layers,
    )


def verify_seeded_draft_with_runtime(
    target_model: torch.nn.Module,
    mtp: MTPModel,
    runtime: MTPRuntimeState,
    seed_token: int,
    draft_ids: Sequence[int],
    selected_layers: Sequence[int],
) -> tuple[list[int], int]:
    emitted = [int(seed_token)]
    advance_runtime_state(target_model, mtp, runtime, int(seed_token), selected_layers)
    accepted_count = 0
    for draft_token in draft_ids:
        target_token = int(torch.argmax(runtime.target.last_logits).item())
        if target_token != int(draft_token):
            emitted.append(target_token)
            advance_runtime_state(target_model, mtp, runtime, target_token, selected_layers)
            return emitted, accepted_count
        emitted.append(int(draft_token))
        accepted_count += 1
        advance_runtime_state(target_model, mtp, runtime, int(draft_token), selected_layers)

    bonus_token = int(torch.argmax(runtime.target.last_logits).item())
    emitted.append(bonus_token)
    advance_runtime_state(target_model, mtp, runtime, bonus_token, selected_layers)
    return emitted, accepted_count


@torch.inference_mode()
def run_mtp_speculative_decode(
    target_model: torch.nn.Module,
    mtp: MTPModel,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    num_speculative_steps: int,
    eos_token_id: int | None,
) -> tuple[list[int], dict[str, int]]:
    selected_layers = (mtp.config.target_num_hidden_layers,)
    runtime = prefill_mtp_runtime(target_model, mtp, prompt_ids, selected_layers)
    generated: list[int] = []
    counters = {
        "speculation_steps": 0,
        "target_forwards": 0,
        "draft_forwards": 0,
        "proposed_draft_tokens": 0,
        "accepted_draft_tokens": 0,
    }
    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        seed_token = int(torch.argmax(runtime.target.last_logits).item())
        requested = min(num_speculative_steps, max(0, remaining - 1))
        draft_ids = propose_mtp_tokens(mtp, runtime, seed_token, requested)
        emitted_ids, accepted_count = verify_seeded_draft_with_runtime(
            target_model,
            mtp,
            runtime,
            seed_token,
            draft_ids,
            selected_layers,
        )
        counters["speculation_steps"] += 1
        counters["target_forwards"] += len(emitted_ids)
        counters["draft_forwards"] += len(draft_ids)
        counters["proposed_draft_tokens"] += len(draft_ids)
        counters["accepted_draft_tokens"] += accepted_count
        for token in emitted_ids:
            if len(generated) >= max_new_tokens:
                break
            generated.append(int(token))
            if eos_token_id is not None and int(token) == eos_token_id:
                return generated, counters
    return generated, counters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MTP speculative decoding.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--output", default="")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--num-speculative-steps", type=int, choices=(1, 2), default=DEFAULT_NUM_SPECULATIVE_STEPS)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup-prompts", type=int, default=DEFAULT_WARMUP_PROMPTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    prompts = load_prompt_records(args.prompts, tokenizer)
    target_model = Qwen3ForCausalLM.from_pretrained(args.model_path, device=args.device, dtype=dtype)
    mtp, _ = load_mtp_checkpoint(args.checkpoint_path, device=args.device, dtype=dtype)

    output_path = Path(args.output) if args.output else Path(f"runs/mtp_steps{args.num_speculative_steps}_nonvllm.jsonl")
    if output_path.exists():
        output_path.unlink()

    def synchronize() -> None:
        if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(torch.device(args.device))

    for _, prompt_ids in prompts[: max(0, args.warmup_prompts)]:
        if not args.skip_baseline:
            autoregressive_generate(target_model, prompt_ids, args.max_new_tokens, 0.0, 1.0, tokenizer.eos_token_id)
        run_mtp_speculative_decode(
            target_model,
            mtp,
            prompt_ids,
            args.max_new_tokens,
            args.num_speculative_steps,
            tokenizer.eos_token_id,
        )
        synchronize()

    divergence_count = 0
    for prompt_id, prompt_ids in prompts:
        baseline_tokens: list[int] = []
        baseline_time_s = 0.0
        if not args.skip_baseline:
            synchronize()
            start = time.perf_counter()
            baseline_tokens = autoregressive_generate(target_model, prompt_ids, args.max_new_tokens, 0.0, 1.0, tokenizer.eos_token_id)
            synchronize()
            baseline_time_s = time.perf_counter() - start
        synchronize()
        start = time.perf_counter()
        generated_tokens, counters = run_mtp_speculative_decode(
            target_model,
            mtp,
            prompt_ids,
            args.max_new_tokens,
            args.num_speculative_steps,
            tokenizer.eos_token_id,
        )
        synchronize()
        method_time_s = time.perf_counter() - start
        matches_baseline = args.skip_baseline or generated_tokens == baseline_tokens
        divergence_count += 0 if matches_baseline else 1
        stats = SpecDecodeStats(
            method="mtp",
            model=args.model_path,
            prompt_id=prompt_id,
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(generated_tokens),
            generated_text=tokenizer.decode(generated_tokens, skip_special_tokens=True),
            temperature=0.0,
            draft_len=args.num_speculative_steps,
            speculation_steps=counters["speculation_steps"],
            target_forwards=counters["target_forwards"],
            draft_forwards=counters["draft_forwards"],
            proposed_draft_tokens=counters["proposed_draft_tokens"],
            accepted_draft_tokens=counters["accepted_draft_tokens"],
            baseline_wall_time_s=baseline_time_s,
            method_wall_time_s=method_time_s,
            torch_compile=False,
            cuda_graphs=False,
            cuda_graphs_reason="disabled",
            seed=args.seed,
        )
        record = stats.to_record()
        record["matches_baseline"] = matches_baseline
        write_jsonl_record(output_path, record)

    output_path.with_suffix(".summary.json").write_text(
        json.dumps(
            summarize_jsonl(output_path)
            | {
                "method": "mtp",
                "num_prompts": len(prompts),
                "num_speculative_steps": args.num_speculative_steps,
                "matches_baseline": divergence_count == 0,
                "diverged_prompts": divergence_count,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

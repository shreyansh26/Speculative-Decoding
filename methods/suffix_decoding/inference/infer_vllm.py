from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from common.tokenizer import load_tokenizer, render_prompt

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

try:
    from vllm import LLM, SamplingParams
    from vllm.v1.metrics.reader import Counter, Vector
except ImportError as exc:  # pragma: no cover - runtime dependency for experiments
    raise SystemExit(
        "vLLM is not installed in the project environment. "
        "Install it with `uv pip install --python .venv/bin/python vllm --torch-backend=auto`."
    ) from exc


DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_PROMPTS_PATH = "data/wiki_extract_ngram_eval100_qwen25_7b.jsonl"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_DRAFT_LEN = 24
DEFAULT_MAX_TREE_DEPTH = 24
DEFAULT_MAX_CACHED_REQUESTS = 10000
DEFAULT_MAX_SPEC_FACTOR = 1.0
DEFAULT_MIN_TOKEN_PROB = 0.1
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_WARMUP_PROMPTS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark suffix decoding with vLLM.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--output", default="")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--draft-len", type=int, default=DEFAULT_DRAFT_LEN)
    parser.add_argument("--max-tree-depth", type=int, default=DEFAULT_MAX_TREE_DEPTH)
    parser.add_argument("--max-cached-requests", type=int, default=DEFAULT_MAX_CACHED_REQUESTS)
    parser.add_argument("--max-spec-factor", type=float, default=DEFAULT_MAX_SPEC_FACTOR)
    parser.add_argument("--min-token-prob", type=float, default=DEFAULT_MIN_TOKEN_PROB)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--serial-prompts", action="store_true")
    parser.add_argument("--warmup-prompts", type=int, default=DEFAULT_WARMUP_PROMPTS)
    parser.add_argument("--limit-prompts", type=int, default=0)
    parser.add_argument("--mode", choices=("both", "baseline", "speculative"), default="both")
    parser.add_argument("--baseline-summary-path", default="")
    parser.add_argument("--require-baseline-match", action="store_true")
    return parser.parse_args()


def build_prompt_token_inputs(
    model_path: str,
    prompts_path: str,
    limit_prompts: int,
) -> tuple[list[dict[str, list[int]]], list[str], dict[str, dict[str, Any]]]:
    tokenizer = load_tokenizer(model_path)
    prompt_inputs: list[dict[str, list[int]]] = []
    prompt_ids: list[str] = []
    metadata_by_prompt: dict[str, dict[str, Any]] = {}
    with Path(prompts_path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if limit_prompts > 0:
        rows = rows[:limit_prompts]
    for line_number, sample in enumerate(rows, start=1):
        prompt_id = str(sample.get("prompt_id", sample.get("id", f"prompt_{line_number:04d}")))
        if "prompt_ids" in sample:
            token_ids = [int(token_id) for token_id in sample["prompt_ids"]]
        else:
            rendered = render_prompt(tokenizer, sample)
            token_ids = tokenizer.encode(rendered.prompt, add_special_tokens=False)
        prompt_inputs.append({"prompt_token_ids": token_ids})
        prompt_ids.append(prompt_id)
        metadata_by_prompt[prompt_id] = {
            key: value
            for key, value in sample.items()
            if key not in {"prompt", "prompt_ids", "messages"}
        }
    if not prompt_inputs:
        raise ValueError(f"{prompts_path} did not contain any prompts")
    return prompt_inputs, prompt_ids, metadata_by_prompt


def collect_spec_decode_metrics(metrics, draft_len: int) -> dict[str, float | int | list[float]]:
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0.0] * draft_len

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += int(metric.value)
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += int(metric.value)
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += int(metric.value)
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for index, value in enumerate(metric.values[:draft_len]):
                acceptance_counts[index] += float(value)

    return {
        "speculation_steps": num_drafts,
        "proposed_draft_tokens": num_draft_tokens,
        "accepted_draft_tokens": num_accepted_tokens,
        "acceptance_rate": (num_accepted_tokens / num_draft_tokens) if num_draft_tokens else 0.0,
        "mean_accepted_tokens_per_step": (num_accepted_tokens / num_drafts) if num_drafts else 0.0,
        "acceptance_rate_by_position": [
            (count / num_drafts) if num_drafts else 0.0 for count in acceptance_counts
        ],
    }


def destroy_llm(llm: LLM | None) -> None:
    if llm is None:
        return
    engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown(timeout=0)
        except TypeError:
            shutdown()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(round((len(sorted_values) - 1) * q))
    index = min(len(sorted_values) - 1, max(0, index))
    return sorted_values[index]


def summarize_latencies(values: list[float], prefix: str) -> dict[str, float]:
    sorted_values = sorted(values)
    return {
        f"{prefix}_mean_s": (sum(sorted_values) / len(sorted_values)) if sorted_values else 0.0,
        f"{prefix}_p50_s": percentile(sorted_values, 0.5),
        f"{prefix}_p95_s": percentile(sorted_values, 0.95),
    }


def request_decode_time_s(output) -> float:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return 0.0
    first = float(getattr(metrics, "first_token_ts", 0.0) or 0.0)
    last = float(getattr(metrics, "last_token_ts", 0.0) or 0.0)
    return max(0.0, last - first) if first and last else 0.0


def run_serial_generate(
    llm: LLM,
    prompt_inputs: list[dict[str, list[int]]],
    prompt_ids: list[str],
    sampling_params: SamplingParams,
    warmup_prompts: int,
) -> tuple[dict[str, list[int]], dict[str, float], dict[str, float], int]:
    warmup_count = min(max(warmup_prompts, 0), len(prompt_inputs))
    for prompt_input in prompt_inputs[:warmup_count]:
        llm.generate([prompt_input], sampling_params=sampling_params, use_tqdm=False)

    outputs_by_prompt: dict[str, list[int]] = {}
    latency_by_prompt: dict[str, float] = {}
    decode_latency_by_prompt: dict[str, float] = {}
    generated_tokens = 0
    for prompt_id, prompt_input in zip(prompt_ids, prompt_inputs, strict=True):
        start = time.perf_counter()
        output = llm.generate([prompt_input], sampling_params=sampling_params, use_tqdm=False)[0]
        latency_s = time.perf_counter() - start
        token_ids = [int(token_id) for token_id in output.outputs[0].token_ids]
        outputs_by_prompt[prompt_id] = token_ids
        latency_by_prompt[prompt_id] = latency_s
        decode_latency_by_prompt[prompt_id] = request_decode_time_s(output)
        generated_tokens += len(token_ids)
    return outputs_by_prompt, latency_by_prompt, decode_latency_by_prompt, generated_tokens


def make_engine_kwargs(args: argparse.Namespace) -> dict[str, object]:
    dtype = "bfloat16" if args.dtype == "bf16" else args.dtype
    kwargs: dict[str, object] = {
        "model": args.model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": dtype,
        "gpu_memory_utilization": DEFAULT_GPU_MEMORY_UTILIZATION,
        "max_model_len": args.max_model_len,
        "disable_log_stats": False,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
    }
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs
    return kwargs


def build_baseline_summary(
    args: argparse.Namespace,
    baseline_llm: LLM,
    prompt_inputs: list[dict[str, list[int]]],
    prompt_ids: list[str],
    sampling_params: SamplingParams,
) -> dict[str, Any]:
    if args.serial_prompts:
        outputs_by_prompt, latency_by_prompt, decode_latency_by_prompt, generated_tokens = run_serial_generate(
            baseline_llm,
            prompt_inputs,
            prompt_ids,
            sampling_params,
            args.warmup_prompts,
        )
        wall_time_s = sum(latency_by_prompt.values())
        decode_wall_time_s = sum(decode_latency_by_prompt.values())
        return {
            "method": "suffix_vllm_baseline_serial",
            "model": args.model_path,
            "num_prompts": len(prompt_inputs),
            "draft_len": args.draft_len,
            "generated_tokens": generated_tokens,
            "baseline_wall_time_s": wall_time_s,
            "baseline_decode_wall_time_s": decode_wall_time_s,
            "baseline_tokens_per_s": generated_tokens / wall_time_s if wall_time_s else 0.0,
            "baseline_decode_tokens_per_s": generated_tokens / decode_wall_time_s if decode_wall_time_s else 0.0,
            "seed": args.seed,
            "gpu_memory_utilization": DEFAULT_GPU_MEMORY_UTILIZATION,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "serial_prompts": True,
            "warmup_prompts": args.warmup_prompts,
            "outputs": outputs_by_prompt,
            "latency_by_prompt_s": latency_by_prompt,
            "decode_latency_by_prompt_s": decode_latency_by_prompt,
        }

    start = time.perf_counter()
    outputs = baseline_llm.generate(prompt_inputs, sampling_params=sampling_params, use_tqdm=False)
    wall_time_s = time.perf_counter() - start
    outputs_by_prompt = {
        prompt_id: [int(token_id) for token_id in output.outputs[0].token_ids]
        for prompt_id, output in zip(prompt_ids, outputs, strict=True)
    }
    generated_tokens = sum(len(token_ids) for token_ids in outputs_by_prompt.values())
    decode_start_values = [
        float(getattr(output.metrics, "first_token_ts", 0.0) or 0.0)
        for output in outputs
        if getattr(output, "metrics", None) is not None
    ]
    decode_end_values = [
        float(getattr(output.metrics, "last_token_ts", 0.0) or 0.0)
        for output in outputs
        if getattr(output, "metrics", None) is not None
    ]
    decode_wall_time_s = (
        max(decode_end_values) - min(decode_start_values)
        if decode_start_values and decode_end_values
        else 0.0
    )
    return {
        "method": "suffix_vllm_baseline",
        "model": args.model_path,
        "num_prompts": len(prompt_inputs),
        "draft_len": args.draft_len,
        "generated_tokens": generated_tokens,
        "baseline_wall_time_s": wall_time_s,
        "baseline_decode_wall_time_s": decode_wall_time_s,
        "baseline_tokens_per_s": generated_tokens / wall_time_s if wall_time_s else 0.0,
        "baseline_decode_tokens_per_s": generated_tokens / decode_wall_time_s if decode_wall_time_s else 0.0,
        "seed": args.seed,
        "gpu_memory_utilization": DEFAULT_GPU_MEMORY_UTILIZATION,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "serial_prompts": False,
        "warmup_prompts": 0,
        "outputs": outputs_by_prompt,
    }


def run_speculative(
    args: argparse.Namespace,
    prompt_inputs: list[dict[str, list[int]]],
    prompt_ids: list[str],
    sampling_params: SamplingParams,
) -> tuple[dict[str, list[int]], dict[str, float], dict[str, float], int, float, float, dict[str, Any]]:
    if importlib.util.find_spec("arctic_inference") is None:
        raise SystemExit(
            "vLLM suffix decoding requires Arctic Inference. "
            "Install it with `uv pip install --python .venv/bin/python arctic-inference==0.1.1`."
        )
    speculative_config = {
        "method": "suffix",
        "num_speculative_tokens": args.draft_len,
        "suffix_decoding_max_tree_depth": args.max_tree_depth,
        "suffix_decoding_max_cached_requests": args.max_cached_requests,
        "suffix_decoding_max_spec_factor": args.max_spec_factor,
        "suffix_decoding_min_token_prob": args.min_token_prob,
    }
    speculative_llm = LLM(**(make_engine_kwargs(args) | {"speculative_config": speculative_config}))
    try:
        if args.serial_prompts:
            outputs_by_prompt, latency_by_prompt, decode_latency_by_prompt, generated_tokens = run_serial_generate(
                speculative_llm,
                prompt_inputs,
                prompt_ids,
                sampling_params,
                args.warmup_prompts,
            )
            wall_time_s = sum(latency_by_prompt.values())
            decode_wall_time_s = sum(decode_latency_by_prompt.values())
        else:
            start = time.perf_counter()
            outputs = speculative_llm.generate(prompt_inputs, sampling_params=sampling_params, use_tqdm=False)
            wall_time_s = time.perf_counter() - start
            outputs_by_prompt = {
                prompt_id: [int(token_id) for token_id in output.outputs[0].token_ids]
                for prompt_id, output in zip(prompt_ids, outputs, strict=True)
            }
            generated_tokens = sum(len(token_ids) for token_ids in outputs_by_prompt.values())
            latency_by_prompt = {}
            decode_latency_by_prompt = {}
            decode_start_values = [
                float(getattr(output.metrics, "first_token_ts", 0.0) or 0.0)
                for output in outputs
                if getattr(output, "metrics", None) is not None
            ]
            decode_end_values = [
                float(getattr(output.metrics, "last_token_ts", 0.0) or 0.0)
                for output in outputs
                if getattr(output, "metrics", None) is not None
            ]
            decode_wall_time_s = (
                max(decode_end_values) - min(decode_start_values)
                if decode_start_values and decode_end_values
                else 0.0
            )
        spec_metrics = collect_spec_decode_metrics(speculative_llm.get_metrics(), args.draft_len)
    finally:
        destroy_llm(speculative_llm)
    return (
        outputs_by_prompt,
        latency_by_prompt,
        decode_latency_by_prompt,
        generated_tokens,
        wall_time_s,
        decode_wall_time_s,
        spec_metrics,
    )


def main() -> None:
    args = parse_args()
    if args.max_num_seqs is None:
        args.max_num_seqs = 1 if args.serial_prompts else 4
    torch.manual_seed(args.seed)

    prompt_inputs, prompt_ids, _metadata_by_prompt = build_prompt_token_inputs(
        args.model_path,
        args.prompts,
        args.limit_prompts,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    baseline_summary_path = (
        Path(args.baseline_summary_path)
        if args.baseline_summary_path
        else Path(f"runs/suffix_vllm_len{args.draft_len}.baseline.json")
    )
    baseline_summary: dict[str, Any] | None = None
    if args.mode in {"both", "baseline"}:
        baseline_llm = LLM(**make_engine_kwargs(args))
        try:
            baseline_summary = build_baseline_summary(
                args,
                baseline_llm,
                prompt_inputs,
                prompt_ids,
                sampling_params,
            )
        finally:
            destroy_llm(baseline_llm)
        baseline_summary_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_summary_path.write_text(json.dumps(baseline_summary, indent=2) + "\n", encoding="utf-8")
        if args.mode == "baseline":
            return

    if baseline_summary is None:
        baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))

    baseline_outputs_by_prompt = baseline_summary["outputs"]
    baseline_latency_by_prompt = baseline_summary.get("latency_by_prompt_s", {})
    baseline_decode_latency_by_prompt = baseline_summary.get("decode_latency_by_prompt_s", {})
    baseline_generated_tokens = int(baseline_summary["generated_tokens"])
    baseline_wall_time_s = float(baseline_summary["baseline_wall_time_s"])
    baseline_decode_wall_time_s = float(baseline_summary.get("baseline_decode_wall_time_s", 0.0))

    (
        speculative_outputs_by_prompt,
        method_latency_by_prompt,
        method_decode_latency_by_prompt,
        speculative_generated_tokens,
        method_wall_time_s,
        method_decode_wall_time_s,
        spec_metrics,
    ) = run_speculative(args, prompt_inputs, prompt_ids, sampling_params)

    diverged_prompt_id: str | None = None
    diverged_prompts = 0
    token_count_mismatches = 0
    for prompt_id in prompt_ids:
        baseline_ids = baseline_outputs_by_prompt[prompt_id]
        speculative_ids = speculative_outputs_by_prompt[prompt_id]
        if len(baseline_ids) != len(speculative_ids):
            token_count_mismatches += 1
        if baseline_ids != speculative_ids:
            diverged_prompts += 1
            if diverged_prompt_id is None:
                diverged_prompt_id = prompt_id
    if args.require_baseline_match and diverged_prompts:
        raise RuntimeError(
            "vLLM baseline/speculative outputs diverged: "
            f"diverged_prompts={diverged_prompts} first={diverged_prompt_id}"
        )

    output_path = (
        Path(args.output)
        if args.output
        else Path(f"runs/suffix_vllm_len{args.draft_len}.summary.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "suffix_vllm",
        "model": args.model_path,
        "num_prompts": len(prompt_inputs),
        "draft_len": args.draft_len,
        "max_tree_depth": args.max_tree_depth,
        "max_cached_requests": args.max_cached_requests,
        "max_spec_factor": args.max_spec_factor,
        "min_token_prob": args.min_token_prob,
        "generated_tokens": speculative_generated_tokens,
        "baseline_generated_tokens": baseline_generated_tokens,
        "speculative_generated_tokens": speculative_generated_tokens,
        "baseline_wall_time_s": baseline_wall_time_s,
        "method_wall_time_s": method_wall_time_s,
        "baseline_decode_wall_time_s": baseline_decode_wall_time_s,
        "method_decode_wall_time_s": method_decode_wall_time_s,
        "baseline_tokens_per_s": baseline_generated_tokens / baseline_wall_time_s if baseline_wall_time_s else 0.0,
        "method_tokens_per_s": speculative_generated_tokens / method_wall_time_s if method_wall_time_s else 0.0,
        "baseline_decode_tokens_per_s": (
            baseline_generated_tokens / baseline_decode_wall_time_s
            if baseline_decode_wall_time_s
            else 0.0
        ),
        "method_decode_tokens_per_s": (
            speculative_generated_tokens / method_decode_wall_time_s
            if method_decode_wall_time_s
            else 0.0
        ),
        "speedup": baseline_wall_time_s / method_wall_time_s if method_wall_time_s else 0.0,
        "decode_speedup": (
            baseline_decode_wall_time_s / method_decode_wall_time_s
            if method_decode_wall_time_s
            else 0.0
        ),
        "matches_baseline": diverged_prompts == 0,
        "diverged_prompts": diverged_prompts,
        "first_diverged_prompt_id": diverged_prompt_id or "",
        "token_count_mismatches": token_count_mismatches,
        "seed": args.seed,
        "gpu_memory_utilization": DEFAULT_GPU_MEMORY_UTILIZATION,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "serial_prompts": args.serial_prompts,
        "warmup_prompts": args.warmup_prompts if args.serial_prompts else 0,
    } | spec_metrics
    if args.serial_prompts:
        baseline_latencies = [float(baseline_latency_by_prompt[prompt_id]) for prompt_id in prompt_ids]
        method_latencies = [float(method_latency_by_prompt[prompt_id]) for prompt_id in prompt_ids]
        baseline_decode_latencies = [
            float(baseline_decode_latency_by_prompt.get(prompt_id, 0.0))
            for prompt_id in prompt_ids
        ]
        method_decode_latencies = [
            float(method_decode_latency_by_prompt.get(prompt_id, 0.0))
            for prompt_id in prompt_ids
        ]
        speedups = [
            (float(baseline_latency_by_prompt[prompt_id]) / float(method_latency_by_prompt[prompt_id]))
            if float(method_latency_by_prompt[prompt_id])
            else 0.0
            for prompt_id in prompt_ids
        ]
        decode_speedups = [
            (
                float(baseline_decode_latency_by_prompt.get(prompt_id, 0.0))
                / float(method_decode_latency_by_prompt.get(prompt_id, 0.0))
            )
            if float(method_decode_latency_by_prompt.get(prompt_id, 0.0))
            else 0.0
            for prompt_id in prompt_ids
        ]
        summary |= summarize_latencies(baseline_latencies, "baseline_latency")
        summary |= summarize_latencies(method_latencies, "method_latency")
        summary |= summarize_latencies(speedups, "latency_speedup")
        summary |= summarize_latencies(baseline_decode_latencies, "baseline_decode_latency")
        summary |= summarize_latencies(method_decode_latencies, "method_decode_latency")
        summary |= summarize_latencies(decode_speedups, "decode_latency_speedup")
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

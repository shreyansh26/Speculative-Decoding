from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from common.tokenizer import load_tokenizer, render_prompt
from methods.draft_model.training.train import load_draft_checkpoint

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
DEFAULT_DRAFT_BASE_MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_CHECKPOINT_PATH = "checkpoints/draft_model_qwen25_05b_ultrachat3000"
DEFAULT_PROMPTS_PATH = "data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_DRAFT_LEN = 2
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85
DEFAULT_MAX_MODEL_LEN = 1280
DEFAULT_WARMUP_PROMPTS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the draft-model method with vLLM.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--draft-base-model-path", default=DEFAULT_DRAFT_BASE_MODEL_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--draft-model-path", default="")
    parser.add_argument("--export-dir", default="")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--output", default="")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--draft-len", type=int, default=DEFAULT_DRAFT_LEN)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--serial-prompts", action="store_true")
    parser.add_argument("--warmup-prompts", type=int, default=DEFAULT_WARMUP_PROMPTS)
    parser.add_argument("--mode", choices=("both", "baseline", "speculative"), default="both")
    parser.add_argument("--baseline-summary-path", default="")
    parser.add_argument("--require-baseline-match", action="store_true")
    return parser.parse_args()


def export_checkpoint_to_hf(
    checkpoint_path: str | Path,
    export_dir: str | Path,
    base_model_path: str,
    tokenizer_source_path: str,
) -> Path:
    export_path = Path(export_dir).resolve()
    export_path.mkdir(parents=True, exist_ok=True)

    target_tokenizer = AutoTokenizer.from_pretrained(tokenizer_source_path, trust_remote_code=True)
    target_config = AutoConfig.from_pretrained(tokenizer_source_path, trust_remote_code=True)
    target_tokenizer.save_pretrained(export_path)

    hf_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        dtype=torch.float32,
    )
    draft_model = load_draft_checkpoint(checkpoint_path, device="cpu")
    incompatible = hf_model.load_state_dict(draft_model.state_dict(), strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint keys during HF export: {incompatible.unexpected_keys[:10]}")
    if incompatible.missing_keys:
        raise RuntimeError(f"missing checkpoint keys during HF export: {incompatible.missing_keys[:10]}")
    target_vocab_size = int(target_config.vocab_size)
    if hf_model.config.vocab_size != target_vocab_size:
        hf_model.resize_token_embeddings(target_vocab_size)

    hf_model.save_pretrained(export_path, safe_serialization=True)
    return export_path


def is_vllm_ready_checkpoint(model_path: str | Path) -> bool:
    path = Path(model_path)
    config_path = path / "config.json"
    weights_path = path / "model.safetensors"
    if not config_path.exists() or not weights_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return "qwen3_config" not in config and (
        "model_type" in config or "architectures" in config
    )


def checkpoint_vocab_size(model_path: str | Path) -> int | None:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = config.get("vocab_size")
    return int(value) if value is not None else None


def build_prompt_token_inputs(model_path: str, prompts_path: str) -> tuple[list[dict[str, list[int]]], list[str]]:
    tokenizer = load_tokenizer(model_path)
    prompt_inputs: list[dict[str, list[int]]] = []
    prompt_ids: list[str] = []
    with Path(prompts_path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for line_number, sample in enumerate(rows, start=1):
        prompt_id = str(sample.get("prompt_id", sample.get("id", f"prompt_{line_number:04d}")))
        if "prompt_ids" in sample:
            token_ids = [int(token_id) for token_id in sample["prompt_ids"]]
        else:
            rendered = render_prompt(tokenizer, sample)
            token_ids = tokenizer.encode(rendered.prompt, add_special_tokens=False)
        prompt_inputs.append({"prompt_token_ids": token_ids})
        prompt_ids.append(prompt_id)
    if not prompt_inputs:
        raise ValueError(f"{prompts_path} did not contain any prompts")
    return prompt_inputs, prompt_ids


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


def run_serial_generate(
    llm: LLM,
    prompt_inputs: list[dict[str, list[int]]],
    prompt_ids: list[str],
    sampling_params: SamplingParams,
    warmup_prompts: int,
) -> tuple[dict[str, list[int]], dict[str, float], int]:
    warmup_count = min(max(warmup_prompts, 0), len(prompt_inputs))
    for prompt_input in prompt_inputs[:warmup_count]:
        llm.generate([prompt_input], sampling_params=sampling_params)

    outputs_by_prompt: dict[str, list[int]] = {}
    latency_by_prompt: dict[str, float] = {}
    generated_tokens = 0
    for prompt_id, prompt_input in zip(prompt_ids, prompt_inputs, strict=True):
        start = time.perf_counter()
        output = llm.generate([prompt_input], sampling_params=sampling_params)[0]
        latency_s = time.perf_counter() - start
        token_ids = output.outputs[0].token_ids
        outputs_by_prompt[prompt_id] = token_ids
        latency_by_prompt[prompt_id] = latency_s
        generated_tokens += len(token_ids)
    return outputs_by_prompt, latency_by_prompt, generated_tokens


def make_engine_kwargs(args: argparse.Namespace) -> dict[str, object]:
    dtype = "bfloat16" if args.dtype == "bf16" else args.dtype
    kwargs: dict[str, object] = {
        "model": args.model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "disable_log_stats": False,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
    }
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs
    return kwargs


def main() -> None:
    args = parse_args()
    if args.max_num_seqs is None:
        args.max_num_seqs = 1 if args.serial_prompts else 16
    torch.manual_seed(args.seed)

    prompt_inputs, prompt_ids = build_prompt_token_inputs(args.model_path, args.prompts)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    baseline_summary_path = (
        Path(args.baseline_summary_path)
        if args.baseline_summary_path
        else Path(f"runs/draft_model_vllm_len{args.draft_len}.baseline.json")
    )
    baseline_outputs = None
    baseline_summary: dict[str, object] | None = None
    if args.mode in {"both", "baseline"}:
        baseline_llm = LLM(**make_engine_kwargs(args))
        if args.serial_prompts:
            baseline_outputs_by_prompt, baseline_latency_by_prompt, baseline_generated_tokens = run_serial_generate(
                baseline_llm,
                prompt_inputs,
                prompt_ids,
                sampling_params,
                args.warmup_prompts,
            )
            baseline_wall_time_s = sum(baseline_latency_by_prompt.values())
            baseline_summary = {
                "method": "draft_model_vllm_baseline_serial",
                "model": args.model_path,
                "num_prompts": len(prompt_inputs),
                "draft_len": args.draft_len,
                "generated_tokens": baseline_generated_tokens,
                "baseline_wall_time_s": baseline_wall_time_s,
                "baseline_tokens_per_s": (
                    baseline_generated_tokens / baseline_wall_time_s if baseline_wall_time_s else 0.0
                ),
                "seed": args.seed,
                "serial_prompts": True,
                "warmup_prompts": args.warmup_prompts,
                "outputs": baseline_outputs_by_prompt,
                "latency_by_prompt_s": baseline_latency_by_prompt,
            }
        else:
            baseline_start = time.perf_counter()
            baseline_outputs = baseline_llm.generate(prompt_inputs, sampling_params=sampling_params)
            baseline_wall_time_s = time.perf_counter() - baseline_start
            baseline_generated_tokens = sum(len(output.outputs[0].token_ids) for output in baseline_outputs)
            baseline_summary = {
                "method": "draft_model_vllm_baseline",
                "model": args.model_path,
                "num_prompts": len(prompt_inputs),
                "draft_len": args.draft_len,
                "generated_tokens": baseline_generated_tokens,
                "baseline_wall_time_s": baseline_wall_time_s,
                "baseline_tokens_per_s": (
                    baseline_generated_tokens / baseline_wall_time_s if baseline_wall_time_s else 0.0
                ),
                "seed": args.seed,
                "serial_prompts": False,
                "warmup_prompts": 0,
                "outputs": {
                    prompt_id: output.outputs[0].token_ids
                    for prompt_id, output in zip(prompt_ids, baseline_outputs, strict=True)
                },
            }
        baseline_summary_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_summary_path.write_text(json.dumps(baseline_summary, indent=2) + "\n", encoding="utf-8")
        destroy_llm(baseline_llm)
        if args.mode == "baseline":
            return

    if baseline_summary is None:
        baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    baseline_outputs_by_prompt = baseline_summary["outputs"]
    baseline_latency_by_prompt = baseline_summary.get("latency_by_prompt_s", {})
    baseline_generated_tokens = int(baseline_summary["generated_tokens"])
    baseline_wall_time_s = float(baseline_summary["baseline_wall_time_s"])

    if args.draft_model_path:
        draft_model_path = Path(args.draft_model_path).resolve()
        if not draft_model_path.exists():
            raise FileNotFoundError(draft_model_path)
    else:
        checkpoint_path = Path(args.checkpoint_path).resolve()
        if args.skip_export:
            draft_model_path = (
                Path(args.export_dir).resolve() if args.export_dir else checkpoint_path
            )
            if not draft_model_path.exists():
                raise FileNotFoundError(
                    f"--skip-export was set but draft model was not found at {draft_model_path}"
                )
        elif (
            is_vllm_ready_checkpoint(checkpoint_path)
            and checkpoint_vocab_size(checkpoint_path)
            == int(AutoConfig.from_pretrained(args.model_path, trust_remote_code=True).vocab_size)
        ):
            draft_model_path = checkpoint_path
        else:
            export_dir = (
                Path(args.export_dir)
                if args.export_dir
                else Path(f"checkpoints/vllm_exports/draft_model_len{args.draft_len}")
            )
            draft_model_path = export_dir.resolve()
            if draft_model_path.exists():
                shutil.rmtree(draft_model_path)
            export_checkpoint_to_hf(
                checkpoint_path=checkpoint_path,
                export_dir=draft_model_path,
                base_model_path=args.draft_base_model_path,
                tokenizer_source_path=args.model_path,
            )

    speculative_config = {
        "method": "draft_model",
        "model": str(draft_model_path),
        "num_speculative_tokens": args.draft_len,
        "draft_tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
    }
    speculative_llm = LLM(
        **(make_engine_kwargs(args) | {"speculative_config": speculative_config})
    )
    if args.serial_prompts:
        speculative_outputs_by_prompt, method_latency_by_prompt, speculative_generated_tokens = run_serial_generate(
            speculative_llm,
            prompt_inputs,
            prompt_ids,
            sampling_params,
            args.warmup_prompts,
        )
        method_wall_time_s = sum(method_latency_by_prompt.values())
    else:
        method_start = time.perf_counter()
        speculative_outputs = speculative_llm.generate(prompt_inputs, sampling_params=sampling_params)
        method_wall_time_s = time.perf_counter() - method_start
        speculative_generated_tokens = sum(len(output.outputs[0].token_ids) for output in speculative_outputs)
        speculative_outputs_by_prompt = {
            prompt_id: output.outputs[0].token_ids
            for prompt_id, output in zip(prompt_ids, speculative_outputs, strict=True)
        }
        method_latency_by_prompt = {}

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

    spec_metrics = collect_spec_decode_metrics(speculative_llm.get_metrics(), args.draft_len)
    destroy_llm(speculative_llm)

    output_path = (
        Path(args.output)
        if args.output
        else Path(f"runs/draft_model_vllm_len{args.draft_len}.summary.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "draft_model_vllm",
        "model": args.model_path,
        "draft_model": str(draft_model_path),
        "num_prompts": len(prompt_inputs),
        "draft_len": args.draft_len,
        "generated_tokens": speculative_generated_tokens,
        "baseline_generated_tokens": baseline_generated_tokens,
        "speculative_generated_tokens": speculative_generated_tokens,
        "baseline_wall_time_s": baseline_wall_time_s,
        "method_wall_time_s": method_wall_time_s,
        "baseline_tokens_per_s": (
            baseline_generated_tokens / baseline_wall_time_s if baseline_wall_time_s else 0.0
        ),
        "method_tokens_per_s": (
            speculative_generated_tokens / method_wall_time_s if method_wall_time_s else 0.0
        ),
        "speedup": (baseline_wall_time_s / method_wall_time_s) if method_wall_time_s else 0.0,
        "matches_baseline": diverged_prompts == 0,
        "diverged_prompts": diverged_prompts,
        "first_diverged_prompt_id": diverged_prompt_id or "",
        "token_count_mismatches": token_count_mismatches,
        "seed": args.seed,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "serial_prompts": args.serial_prompts,
        "warmup_prompts": args.warmup_prompts if args.serial_prompts else 0,
    } | spec_metrics
    if args.serial_prompts:
        baseline_latencies = [float(baseline_latency_by_prompt[prompt_id]) for prompt_id in prompt_ids]
        method_latencies = [float(method_latency_by_prompt[prompt_id]) for prompt_id in prompt_ids]
        speedups = [
            (float(baseline_latency_by_prompt[prompt_id]) / float(method_latency_by_prompt[prompt_id]))
            if float(method_latency_by_prompt[prompt_id]) else 0.0
            for prompt_id in prompt_ids
        ]
        summary |= summarize_latencies(baseline_latencies, "baseline_latency")
        summary |= summarize_latencies(method_latencies, "method_latency")
        summary |= summarize_latencies(speedups, "latency_speedup")
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

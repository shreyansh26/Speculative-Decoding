"""vLLM benchmark for the MTP method."""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoConfig

from common.tokenizer import load_tokenizer, render_prompt

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

try:
    from vllm import LLM, SamplingParams
    from vllm.v1.metrics.reader import Counter, Vector
except ImportError as exc:  # pragma: no cover - runtime dependency for experiments
    raise SystemExit(
        "vLLM is not installed in the project environment. "
        "Install it with `uv pip install --python .venv/bin/python vllm --torch-backend=auto`."
    ) from exc


DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CHECKPOINT_PATH = "checkpoints/mtp_qwen25_7b_eval100_steps1"
DEFAULT_PROMPTS_PATH = "data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_NUM_SPECULATIVE_STEPS = 1
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85
DEFAULT_MAX_MODEL_LEN = 1280
DEFAULT_WARMUP_PROMPTS = 1


def patch_mimo_mtp_multi_step() -> None:
    """Allow vLLM's Qwen2/MiMo MTP module to use layer 1 for the second step."""
    def requested_mtp_layers() -> int:
        try:
            return int(os.environ.get("CODEX_MTP_NUM_SPEC_TOKENS", "1"))
        except ValueError:
            return 1

    def resolve_mtp_layers(*values: object) -> int:
        candidates = [requested_mtp_layers()]
        for value in values:
            if value is None:
                continue
            try:
                candidates.append(int(value))
            except (TypeError, ValueError):
                continue
        return max(candidates)

    try:
        from vllm.config.speculative import SpeculativeConfig
    except Exception:
        SpeculativeConfig = None
    if SpeculativeConfig is not None and not getattr(SpeculativeConfig, "_codex_mtp_override_patched", False):
        original_override = SpeculativeConfig.hf_config_override

        def hf_config_override(hf_config):
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            updated = original_override(hf_config)
            if updated.architectures[0] == "MiMoMTPModel" and n_predict is not None:
                layer_count = resolve_mtp_layers(
                    n_predict,
                    getattr(updated, "num_nextn_predict_layers", None),
                    getattr(updated, "n_predict", None),
                )
                updated.update({"num_nextn_predict_layers": layer_count, "n_predict": layer_count})
            return updated

        SpeculativeConfig.hf_config_override = staticmethod(hf_config_override)
        SpeculativeConfig._codex_mtp_override_patched = True

    try:
        from vllm.model_executor.models import mimo_mtp
    except Exception:
        return
    if not getattr(mimo_mtp.MiMoMultiTokenPredictor, "_codex_mtp_init_patched", False):
        original_init = mimo_mtp.MiMoMultiTokenPredictor.__init__

        def init(self, *args, **kwargs):
            vllm_config = kwargs.get("vllm_config", args[0] if args else None)
            if vllm_config is not None:
                config = vllm_config.model_config.hf_config
                config.num_nextn_predict_layers = resolve_mtp_layers(
                    getattr(config, "num_nextn_predict_layers", None),
                    getattr(config, "n_predict", None),
                )
                config.n_predict = config.num_nextn_predict_layers
            original_init(self, *args, **kwargs)

        mimo_mtp.MiMoMultiTokenPredictor.__init__ = init
        mimo_mtp.MiMoMultiTokenPredictor._codex_mtp_init_patched = True
    if not getattr(mimo_mtp.MiMoMTP, "_codex_mtp_load_patched", False):
        original_load_weights = mimo_mtp.MiMoMTP.load_weights

        def load_weights(self, weights):
            def fixed():
                for name, tensor in weights:
                    if name == "embed_tokens.weight":
                        name = "model.embed_tokens.weight"
                    elif name.startswith("mtp_layers."):
                        name = "model." + name
                    yield name, tensor

            return original_load_weights(self, fixed())

        mimo_mtp.MiMoMTP.load_weights = load_weights
        mimo_mtp.MiMoMTP._codex_mtp_load_patched = True

    def forward(
        self,
        input_ids,
        positions,
        hidden_states,
        intermediate_tensors=None,
        inputs_embeds=None,
        spec_step_idx: int = 0,
    ):
        del intermediate_tensors
        return self.model(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            spec_step_idx,
        )

    mimo_mtp.MiMoMTP.forward = forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MTP speculative decoding with vLLM.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--draft-model-path", default="")
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--output", default="")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--num-speculative-steps", type=int, choices=(1, 2), default=DEFAULT_NUM_SPECULATIVE_STEPS)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--draft-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--serial-prompts", action="store_true")
    parser.add_argument("--warmup-prompts", type=int, default=DEFAULT_WARMUP_PROMPTS)
    parser.add_argument("--mode", choices=("both", "baseline", "speculative"), default="both")
    parser.add_argument("--baseline-summary-path", default="")
    return parser.parse_args()


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
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
    except Exception:
        pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(round((len(sorted_values) - 1) * q))
    return sorted_values[min(len(sorted_values) - 1, max(0, index))]


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
    for prompt_input in prompt_inputs[: min(max(warmup_prompts, 0), len(prompt_inputs))]:
        llm.generate([prompt_input], sampling_params=sampling_params)
    outputs_by_prompt: dict[str, list[int]] = {}
    latency_by_prompt: dict[str, float] = {}
    generated_tokens = 0
    for prompt_id, prompt_input in zip(prompt_ids, prompt_inputs, strict=True):
        start = time.perf_counter()
        output = llm.generate([prompt_input], sampling_params=sampling_params)[0]
        latency_s = time.perf_counter() - start
        token_ids = list(output.outputs[0].token_ids)
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


def validate_mtp_vllm_checkpoint(target_model_path: str, draft_model_path: str | Path, num_steps: int) -> None:
    target_config = AutoConfig.from_pretrained(target_model_path, trust_remote_code=True)
    draft_config = AutoConfig.from_pretrained(draft_model_path, trust_remote_code=True)
    if int(target_config.vocab_size) != int(draft_config.vocab_size):
        raise ValueError("MTP vLLM checkpoint must share the target vocabulary")
    if draft_config.model_type != "qwen2" or draft_config.architectures[0] != "MiMoForCausalLM":
        raise ValueError("expected a vLLM MiMo MTP export with qwen2/MiMoForCausalLM config")
    n_predict = int(getattr(draft_config, "n_predict", getattr(draft_config, "num_nextn_predict_layers", 0)))
    if n_predict < num_steps:
        raise ValueError(f"draft checkpoint supports {n_predict} MTP steps, requested {num_steps}")


def main() -> None:
    args = parse_args()
    if args.max_num_seqs is None:
        args.max_num_seqs = 1 if args.serial_prompts else 16
    os.environ["CODEX_MTP_NUM_SPEC_TOKENS"] = str(args.num_speculative_steps)
    patch_mimo_mtp_multi_step()
    torch.manual_seed(args.seed)
    draft_model_path = args.draft_model_path or args.checkpoint_path
    if not args.draft_model_path and (Path(draft_model_path) / "vllm_export").exists():
        draft_model_path = str(Path(draft_model_path) / "vllm_export")
    validate_mtp_vllm_checkpoint(args.model_path, draft_model_path, args.num_speculative_steps)
    prompt_inputs, prompt_ids = build_prompt_token_inputs(args.model_path, args.prompts)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)

    baseline_summary_path = (
        Path(args.baseline_summary_path)
        if args.baseline_summary_path
        else Path(f"runs/mtp_steps{args.num_speculative_steps}_vllm.baseline.json")
    )
    baseline_summary: dict[str, object] | None = None
    if args.mode in {"both", "baseline"}:
        baseline_llm = LLM(**make_engine_kwargs(args))
        if args.serial_prompts:
            outputs_by_prompt, latency_by_prompt, generated_tokens = run_serial_generate(
                baseline_llm, prompt_inputs, prompt_ids, sampling_params, args.warmup_prompts
            )
            wall_time_s = sum(latency_by_prompt.values())
            baseline_summary = {
                "method": "mtp_vllm_baseline_serial",
                "model": args.model_path,
                "num_prompts": len(prompt_inputs),
                "generated_tokens": generated_tokens,
                "baseline_wall_time_s": wall_time_s,
                "baseline_tokens_per_s": generated_tokens / wall_time_s if wall_time_s else 0.0,
                "outputs": outputs_by_prompt,
                "latency_by_prompt_s": latency_by_prompt,
                **summarize_latencies(list(latency_by_prompt.values()), "baseline_latency"),
            }
        else:
            start = time.perf_counter()
            outputs = baseline_llm.generate(prompt_inputs, sampling_params=sampling_params)
            wall_time_s = time.perf_counter() - start
            generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            baseline_summary = {
                "method": "mtp_vllm_baseline",
                "model": args.model_path,
                "num_prompts": len(prompt_inputs),
                "generated_tokens": generated_tokens,
                "baseline_wall_time_s": wall_time_s,
                "baseline_tokens_per_s": generated_tokens / wall_time_s if wall_time_s else 0.0,
                "outputs": {
                    prompt_id: list(output.outputs[0].token_ids)
                    for prompt_id, output in zip(prompt_ids, outputs, strict=True)
                },
            }
        baseline_summary_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_summary_path.write_text(json.dumps(baseline_summary, indent=2) + "\n", encoding="utf-8")
        destroy_llm(baseline_llm)
        baseline_llm = None
    elif baseline_summary_path.exists():
        baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))

    if args.mode == "baseline":
        return
    if baseline_summary is None:
        raise ValueError("speculative mode requires a baseline summary")

    spec_kwargs = make_engine_kwargs(args)
    spec_kwargs["speculative_config"] = {
        "method": "mtp",
        "model": str(draft_model_path),
        "num_speculative_tokens": args.num_speculative_steps,
        "draft_tensor_parallel_size": args.draft_tensor_parallel_size,
    }
    speculative_llm = LLM(**spec_kwargs)
    if args.serial_prompts:
        spec_outputs_by_prompt, spec_latency_by_prompt, spec_generated_tokens = run_serial_generate(
            speculative_llm, prompt_inputs, prompt_ids, sampling_params, args.warmup_prompts
        )
        method_wall_time_s = sum(spec_latency_by_prompt.values())
        latency_summary = summarize_latencies(list(spec_latency_by_prompt.values()), "method_latency")
    else:
        start = time.perf_counter()
        spec_outputs = speculative_llm.generate(prompt_inputs, sampling_params=sampling_params)
        method_wall_time_s = time.perf_counter() - start
        spec_generated_tokens = sum(len(output.outputs[0].token_ids) for output in spec_outputs)
        spec_outputs_by_prompt = {
            prompt_id: list(output.outputs[0].token_ids)
            for prompt_id, output in zip(prompt_ids, spec_outputs, strict=True)
        }
        latency_summary = {}
    metrics = collect_spec_decode_metrics(speculative_llm.llm_engine.get_metrics(), args.num_speculative_steps)
    baseline_outputs = baseline_summary.get("outputs", {})
    diverged = sum(
        1
        for prompt_id in prompt_ids
        if list(spec_outputs_by_prompt.get(prompt_id, [])) != list(baseline_outputs.get(prompt_id, []))
    )
    baseline_wall = float(baseline_summary["baseline_wall_time_s"])
    baseline_tps = float(baseline_summary["baseline_tokens_per_s"])
    method_tps = spec_generated_tokens / method_wall_time_s if method_wall_time_s else 0.0
    summary = {
        "method": "mtp_vllm_serial" if args.serial_prompts else "mtp_vllm_batched",
        "model": args.model_path,
        "draft_model_path": str(draft_model_path),
        "num_prompts": len(prompt_inputs),
        "num_speculative_steps": args.num_speculative_steps,
        "generated_tokens": spec_generated_tokens,
        "baseline_wall_time_s": baseline_wall,
        "method_wall_time_s": method_wall_time_s,
        "baseline_tokens_per_s": baseline_tps,
        "method_tokens_per_s": method_tps,
        "speedup": baseline_wall / method_wall_time_s if method_wall_time_s else 0.0,
        "throughput_speedup": method_tps / baseline_tps if baseline_tps else 0.0,
        "matches_baseline": diverged == 0,
        "diverged_prompts": diverged,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "serial_prompts": args.serial_prompts,
        **metrics,
        **latency_summary,
    }
    output_path = Path(args.output) if args.output else Path(f"runs/mtp_steps{args.num_speculative_steps}_vllm.summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    destroy_llm(speculative_llm)


if __name__ == "__main__":
    main()

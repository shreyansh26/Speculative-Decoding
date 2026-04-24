import argparse
import copy
import gc
import json
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, Cache
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs

from common.tokenizer import render_prompt, load_tokenizer
from methods.eagle3.training.train import Eagle3Config, load_eagle3_checkpoint

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

try:
    from speculators.models import base_components
    from speculators.models.eagle3 import Eagle3DraftModel
    from speculators.models.eagle3 import core as eagle_core
    from speculators.models.eagle3 import model_definitions
    from vllm import LLM, SamplingParams
    from vllm.v1.metrics.reader import Counter, Vector
except ImportError as exc:  # pragma: no cover - runtime dependency for experiments
    raise SystemExit(
        "vLLM/speculators are not installed in the project environment. "
        "Install them with `uv pip install --python .venv/bin/python vllm speculators`."
    ) from exc


DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CHECKPOINT_PATH = "checkpoints/eagle3_qwen25_7b"
DEFAULT_PROMPTS_PATH = "data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_DRAFT_LEN = 3
DEFAULT_GPU_MEMORY_UTILIZATION = 0.4
DEFAULT_MAX_MODEL_LEN = 1280
DEFAULT_WARMUP_PROMPTS = 1

_QWEN2_SPECULATORS_PATCHED = False

EAGLE3_CONFIG_PY = '''from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig

from speculators import SpeculatorModelConfig

__all__ = ["Eagle3SpeculatorConfig"]


@SpeculatorModelConfig.register("eagle3")
class Eagle3SpeculatorConfig(SpeculatorModelConfig):
    speculators_model_type: Literal["eagle3"] = "eagle3"
    architectures: list[str] = Field(default_factory=lambda: ["Eagle3Speculator"])
    transformer_layer_config: PretrainedConfig = Field(default_factory=LlamaConfig)
    draft_vocab_size: int = 32000
    norm_before_residual: bool = False
    target_hidden_size: int | None = None
    eagle_aux_hidden_state_layer_ids: list[int] | None = None
    embed_requires_grad: bool = False

    @property
    def target_vocab_size(self) -> int:
        return self.transformer_layer_config.vocab_size

    @field_serializer("transformer_layer_config")
    def serialize_transformer_config(self, value: PretrainedConfig) -> dict:
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_config(cls, value: Any) -> PretrainedConfig:
        if isinstance(value, dict):
            config_class: type[PretrainedConfig] = LlamaConfig
            if "model_type" in value:
                config_class = AutoConfig.for_model(model_type=value["model_type"]).__class__
            return config_class(**value)
        return value
'''


class Qwen2DecoderEagle3FirstLayer(Qwen2DecoderLayer):
    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: int,
        norm_before_residual: bool = False,
    ) -> None:
        super().__init__(config, layer_idx)
        self.norm_before_residual = norm_before_residual
        self.hidden_norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        q_bias = self.self_attn.q_proj.bias is not None
        k_bias = self.self_attn.k_proj.bias is not None
        v_bias = self.self_attn.v_proj.bias is not None
        self.self_attn.q_proj = torch.nn.Linear(
            2 * config.hidden_size,
            self.self_attn.q_proj.out_features,
            bias=q_bias,
        )
        self.self_attn.k_proj = torch.nn.Linear(
            2 * config.hidden_size,
            self.self_attn.k_proj.out_features,
            bias=k_bias,
        )
        self.self_attn.v_proj = torch.nn.Linear(
            2 * config.hidden_size,
            self.self_attn.v_proj.out_features,
            bias=v_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],  # type: ignore[valid-type]
    ) -> torch.Tensor:
        mid = hidden_states.shape[2] // 2
        embeds, hidden = hidden_states.split(mid, dim=-1)
        residual = hidden
        embeds = self.input_layernorm(embeds)
        hidden = self.hidden_norm(hidden)
        if self.norm_before_residual:
            residual = hidden
        hidden_states = torch.cat([embeds, hidden], dim=-1)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def ensure_qwen2_speculators_support() -> None:
    global _QWEN2_SPECULATORS_PATCHED
    if _QWEN2_SPECULATORS_PATCHED:
        return
    if "qwen2" not in base_components.model_classes:
        base_components.model_classes["qwen2"] = base_components.ModelComponents(
            Qwen2DecoderLayer,
            Qwen2DecoderLayer,
            Qwen2RMSNorm,
            Qwen2RotaryEmbedding,
        )
    model_definitions.model_classes["qwen2"] = base_components.override_components(
        "qwen2",
        first_layer_class=Qwen2DecoderEagle3FirstLayer,
    )
    eagle_core.model_classes["qwen2"] = model_definitions.model_classes["qwen2"]
    _QWEN2_SPECULATORS_PATCHED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark EAGLE-3 speculative decoding with vLLM.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
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
    parser.add_argument("--draft-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--parallel-drafting", action="store_true")
    parser.add_argument("--serial-prompts", action="store_true")
    parser.add_argument("--warmup-prompts", type=int, default=DEFAULT_WARMUP_PROMPTS)
    parser.add_argument("--mode", choices=("both", "baseline", "speculative"), default="both")
    parser.add_argument("--baseline-summary-path", default="")
    return parser.parse_args()


def export_checkpoint_to_speculators(
    checkpoint_path: str | Path,
    export_dir: str | Path,
    verifier_model_path: str,
    speculative_tokens: int,
) -> Path:
    export_path = Path(export_dir).resolve()
    export_path.mkdir(parents=True, exist_ok=True)

    drafter, metadata = load_eagle3_checkpoint(checkpoint_path, device="cpu")
    draft_config = Eagle3Config(**metadata["eagle3_config"])
    verifier_config = AutoConfig.from_pretrained(verifier_model_path, trust_remote_code=True)
    draft_transformer_config = copy.deepcopy(verifier_config)
    draft_transformer_config.num_hidden_layers = draft_config.num_hidden_layers
    draft_transformer_config.attention_bias = True
    if hasattr(draft_transformer_config, "layer_types"):
        layer_types = list(getattr(draft_transformer_config, "layer_types"))
        draft_transformer_config.layer_types = layer_types[: draft_config.num_hidden_layers]
    if hasattr(draft_transformer_config, "max_window_layers"):
        draft_transformer_config.max_window_layers = min(
            int(getattr(draft_transformer_config, "max_window_layers")),
            draft_config.num_hidden_layers,
        )

    source_state = drafter.state_dict()
    export_state = {
        "fc.weight": source_state["feature_fuser.weight"].detach().clone(),
        "embed_tokens.weight": source_state["token_embedding.weight"].detach().clone(),
        "lm_head.weight": source_state["lm_head.weight"].detach().clone(),
        "norm.weight": source_state["norm.weight"].detach().clone(),
    }
    for layer_idx in range(draft_config.num_hidden_layers):
        source_prefix = f"layers.{layer_idx}."
        for suffix in (
            "hidden_norm.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ):
            source_key = source_prefix + suffix
            if source_key not in source_state:
                continue
            export_state[f"layers.{layer_idx}.{suffix}"] = source_state[source_key].detach().clone()
    save_file(export_state, str(export_path / "model.safetensors"))

    verifier_dict = verifier_config.to_dict()
    transformer_layer_config = draft_transformer_config.to_dict()
    speculator_config = {
        "architectures": ["Eagle3DraftModel"],
        "auto_map": {"": "config.Eagle3SpeculatorConfig"},
        "base_model_ep_plan": None,
        "draft_vocab_size": int(verifier_config.vocab_size),
        "dtype": "float32",
        "eagle_aux_hidden_state_layer_ids": list(draft_config.selected_layers),
        "embed_requires_grad": False,
        "has_no_defaults_at_init": False,
        "norm_before_residual": bool(draft_config.norm_before_residual),
        "speculators_config": {
            "algorithm": "eagle3",
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {
                    "accept_tolerance": 0.0,
                    "proposal_type": "greedy",
                    "speculative_tokens": int(speculative_tokens),
                    "verifier_accept_k": 1,
                }
            ],
            "verifier": {
                "architectures": verifier_dict.get("architectures", []),
                "name_or_path": verifier_model_path,
            },
        },
        "speculators_model_type": "eagle3",
        "speculators_version": "0.4.0.1",
        "target_hidden_size": None,
        "transformer_layer_config": transformer_layer_config,
        "transformers_version": getattr(verifier_config, "transformers_version", None),
    }
    (export_path / "config.json").write_text(
        json.dumps(speculator_config, indent=2) + "\n",
        encoding="utf-8",
    )
    (export_path / "config.py").write_text(EAGLE3_CONFIG_PY, encoding="utf-8")

    AutoTokenizer.from_pretrained(verifier_model_path, trust_remote_code=True).save_pretrained(export_path)
    export_metadata = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "verifier_model_path": verifier_model_path,
        "feature_fuser_bias_l2": (
            float(source_state["feature_fuser.bias"].norm().item())
            if "feature_fuser.bias" in source_state
            else 0.0
        ),
        "dropped_feature_fuser_bias": "feature_fuser.bias" in source_state,
        "exported_speculative_tokens": int(speculative_tokens),
        "eagle3_config": metadata["eagle3_config"],
    }
    (export_path / "codex_export_metadata.json").write_text(
        json.dumps(export_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return export_path


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
            prompt = render_prompt(tokenizer, sample)
            token_ids = tokenizer.encode(prompt.prompt, add_special_tokens=False)
        prompt_inputs.append({"prompt_token_ids": token_ids})
        prompt_ids.append(prompt_id)
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    prompt_inputs, prompt_ids = build_prompt_token_inputs(args.model_path, args.prompts)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)

    baseline_summary_path = (
        Path(args.baseline_summary_path)
        if args.baseline_summary_path
        else Path(f"runs/eagle3_vllm_len{args.draft_len}.baseline.json")
    )
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
                "method": "eagle3_vllm_baseline_serial",
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
                "method": "eagle3_vllm_baseline",
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
        baseline_llm = None
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
    else:
        export_dir = (
            Path(args.export_dir)
            if args.export_dir
            else Path(f"checkpoints/vllm_exports/eagle3_len{args.draft_len}")
        )
        draft_model_path = export_dir.resolve()
        if args.skip_export:
            if not draft_model_path.exists():
                raise FileNotFoundError(
                    f"--skip-export was set but Eagle3 export was not found at {draft_model_path}"
                )
        else:
            if draft_model_path.exists():
                shutil.rmtree(draft_model_path)
            export_checkpoint_to_speculators(
                checkpoint_path=args.checkpoint_path,
                export_dir=draft_model_path,
                verifier_model_path=args.model_path,
                speculative_tokens=args.draft_len,
            )

    speculative_config = {
        "method": "eagle3",
        "model": str(draft_model_path),
        "num_speculative_tokens": args.draft_len,
        "draft_tensor_parallel_size": args.draft_tensor_parallel_size,
        "max_model_len": args.max_model_len,
    }
    if args.parallel_drafting:
        speculative_config["parallel_drafting"] = True

    speculative_llm = LLM(**(make_engine_kwargs(args) | {"speculative_config": speculative_config}))
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

    spec_metrics = collect_spec_decode_metrics(speculative_llm.get_metrics(), args.draft_len)
    destroy_llm(speculative_llm)
    speculative_llm = None

    output_path = (
        Path(args.output)
        if args.output
        else Path(f"runs/eagle3_vllm_len{args.draft_len}.summary.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "eagle3_vllm",
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
        "parallel_drafting": args.parallel_drafting,
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

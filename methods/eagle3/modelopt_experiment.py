"""
Standalone NVIDIA ModelOpt EAGLE-3 comparison script.

This script is intentionally separate from train.py, infer.py, and infer_vllm.py so
the official-implementation experiment can be deleted without touching the local
implementation. It converts our cached UltraChat rows into ModelOpt chat training
rows, trains via modelopt.torch.speculative, exports a draft checkpoint, converts it
to the vLLM/speculators one-checkpoint format, and optionally launches the existing
vLLM benchmark script.
"""

import argparse
import copy
import gc
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import transformers


DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TRAIN_DATA = "data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl"
DEFAULT_PROMPTS = "data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl"
DEFAULT_MODELOPT_DATA = "data/modelopt_ultrachat_eval100_qwen25_7b_greedy128.jsonl"
DEFAULT_OUTPUT_DIR = "checkpoints/eagle3_modelopt_qwen25_7b_eval100_len3"
DEFAULT_RAW_EXPORT_DIR = "checkpoints/modelopt_exports/eagle3_qwen25_7b_eval100_len3_raw"
DEFAULT_VLLM_EXPORT_DIR = "checkpoints/vllm_exports/eagle3_modelopt_eval100_len2"
DEFAULT_SUMMARY_PATH = "runs/eagle3_modelopt_eval100_len2_vllm_batched.summary.json"


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
    norm_before_residual: bool = True
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


@dataclass
class ModelOptDataArgs:
    data_path: str
    offline_data_path: str | None = None
    lazy_preprocess: bool = True
    draft_vocab_cache: str | None = None
    chat_template: str | None = None
    vlm_img_dir: str | None = None
    vlm_processor: str | None = None
    sample_size: int = -1


@dataclass
class ModelOptTrainingArguments(transformers.TrainingArguments):
    training_seq_len: int = field(default=1152)
    mode: Literal["eagle3"] = "eagle3"
    estimate_ar: bool = False
    ar_validate_steps: int = 0
    answer_only_loss: bool = False
    cp_size: int = 1
    dp_shard_size: int | None = 1


class JsonlDataset:
    def __init__(self, path: str | Path, limit: int = 0) -> None:
        self.path = Path(path)
        self.rows: list[dict] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                self.rows.append(json.loads(line))
                if limit > 0 and len(self.rows) >= limit:
                    break
        if not self.rows:
            raise ValueError(f"No rows found in {self.path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/export/benchmark EAGLE-3 using NVIDIA ModelOpt for comparison."
    )
    parser.add_argument(
        "--mode",
        choices=("prepare", "train", "export", "convert", "bench", "all"),
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-data", default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS)
    parser.add_argument("--modelopt-data", default=DEFAULT_MODELOPT_DATA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-export-dir", default=DEFAULT_RAW_EXPORT_DIR)
    parser.add_argument("--vllm-export-dir", default=DEFAULT_VLLM_EXPORT_DIR)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--data-limit", type=int, default=0)
    parser.add_argument("--overwrite-data", action="store_true")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--overwrite-exports", action="store_true")

    parser.add_argument("--seq-len", type=int, default=1152)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--draft-layers", type=int, default=1)
    parser.add_argument("--draft-len", type=int, default=2)
    parser.add_argument("--ttt-steps", type=int, default=3)
    parser.add_argument("--loss-decay", type=float, default=1.0)
    parser.add_argument("--selected-layers", default="")
    parser.add_argument(
        "--official-arch-defaults",
        action="store_true",
        help="Use ModelOpt's generic draft architecture defaults instead of Qwen-derived dimensions.",
    )

    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=1280)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=16,
        help="Default matches the EAGLE-3 README batched throughput benchmark.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--draft-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--serial-prompts", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--parallel-drafting", action="store_true")
    parser.add_argument("--baseline-summary-path", default="")
    return parser.parse_args()


def ensure_project_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def import_modelopt_runtime():
    try:
        import torch
        from transformers import AutoConfig, AutoTokenizer, Trainer

        import modelopt.torch.opt as mto
        import modelopt.torch.speculative as mtsp
        from modelopt.torch.export import export_speculative_decoding
        from modelopt.torch.speculative.config import EagleConfig
        from modelopt.torch.speculative.utils import (
            load_vlm_or_llm,
            patch_transformers5_params_loading,
        )
        from modelopt.torch.utils.plugins.transformers_dataset import LanguageDataCollator
    except Exception as exc:  # pragma: no cover - runtime environment guard
        raise SystemExit(
            "ModelOpt training dependencies are not installed in this environment.\n"
            "Install them in the project venv, for example:\n"
            "  uv pip install --python .venv/bin/python -e ref_repos/Model-Optimizer "
            "accelerate peft scipy pulp nvidia-ml-py omegaconf\n"
        ) from exc
    return {
        "torch": torch,
        "AutoConfig": AutoConfig,
        "AutoTokenizer": AutoTokenizer,
        "Trainer": Trainer,
        "mto": mto,
        "mtsp": mtsp,
        "export_speculative_decoding": export_speculative_decoding,
        "EagleConfig": EagleConfig,
        "LanguageDataCollator": LanguageDataCollator,
        "load_vlm_or_llm": load_vlm_or_llm,
        "patch_transformers5_params_loading": patch_transformers5_params_loading,
    }


def prepare_modelopt_data(args: argparse.Namespace) -> Path:
    source_path = Path(args.train_data)
    output_path = Path(args.modelopt_data)
    if output_path.exists() and not args.overwrite_data:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with source_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as dest:
        for line in source:
            if not line.strip():
                continue
            sample = json.loads(line)
            messages = copy.deepcopy(sample.get("messages") or [])
            completion = sample.get("completion")
            if completion and (not messages or messages[-1].get("role") != "assistant"):
                messages.append({"role": "assistant", "content": completion})
            if not any(message.get("role") == "assistant" for message in messages):
                continue
            row = {
                "prompt_id": sample.get("prompt_id", sample.get("id", f"row_{written:06d}")),
                "messages": messages,
                "target_model": sample.get("target_model", args.model_path),
            }
            dest.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if args.data_limit > 0 and written >= args.data_limit:
                break
    if written == 0:
        raise ValueError(f"No usable ModelOpt chat rows were written from {source_path}")
    print(f"Wrote {written} ModelOpt training rows to {output_path}")
    return output_path


def parse_selected_layers(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_eagle_config(args: argparse.Namespace, verifier_config) -> dict:
    arch_config: dict[str, object] = {
        "num_hidden_layers": args.draft_layers,
        "eagle_aux_hidden_state_layer_ids": parse_selected_layers(args.selected_layers),
    }
    if not args.official_arch_defaults:
        for name in (
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_theta",
            "hidden_act",
            "attention_bias",
            "attention_dropout",
            "mlp_bias",
            "head_dim",
        ):
            if hasattr(verifier_config, name):
                value = getattr(verifier_config, name)
                if value is not None:
                    arch_config[name] = value
        if "head_dim" not in arch_config and hasattr(verifier_config, "hidden_size"):
            arch_config["head_dim"] = int(verifier_config.hidden_size) // int(
                verifier_config.num_attention_heads
            )

    return {
        "eagle_decoder_type": "llama",
        "eagle_ttt_steps": args.ttt_steps,
        "eagle_mix_hidden_states": False,
        "eagle_use_torch_compile": args.torch_compile,
        "eagle_self_logit_distillation": True,
        "eagle_freeze_base_model": True,
        "eagle_loss_decay_factor": args.loss_decay,
        "eagle_hidden_state_distillation": False,
        "eagle_reuse_base_decoder": False,
        "eagle_report_acc": True,
        "eagle_architecture_config": arch_config,
        "eagle_export_rope_scaling": {},
    }


def train_modelopt(args: argparse.Namespace) -> None:
    runtime = import_modelopt_runtime()
    torch = runtime["torch"]
    AutoConfig = runtime["AutoConfig"]
    AutoTokenizer = runtime["AutoTokenizer"]
    Trainer = runtime["Trainer"]
    LanguageDataCollator = runtime["LanguageDataCollator"]
    EagleConfig = runtime["EagleConfig"]
    load_vlm_or_llm = runtime["load_vlm_or_llm"]
    mto = runtime["mto"]
    mtsp = runtime["mtsp"]

    mto.enable_huggingface_checkpointing()
    torch.manual_seed(0)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite_output_dir:
        shutil.rmtree(output_dir)

    modelopt_data_path = prepare_modelopt_data(args)
    data_args = ModelOptDataArgs(data_path=str(modelopt_data_path))
    training_args = ModelOptTrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=True,
        do_eval=False,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        dataloader_num_workers=args.dataloader_num_workers,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="no",
        report_to=[],
        optim="adamw_torch",
        bf16=args.dtype == "bf16",
        fp16=args.dtype == "fp16",
        tf32=args.tf32,
        training_seq_len=args.seq_len,
        answer_only_loss=False,
        ar_validate_steps=0,
        disable_tqdm=False,
    )

    print("Loading base model on CPU and converting to ModelOpt EAGLE-3...")
    verifier_config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    model = load_vlm_or_llm(
        args.model_path,
        dtype="auto",
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=args.seq_len,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eagle_cfg = EagleConfig.model_validate(
        build_eagle_config(args, verifier_config),
        context={"training_args": training_args, "data_args": data_args},
    ).model_dump()
    mtsp.convert(model, [("eagle", eagle_cfg)])

    train_dataset = JsonlDataset(modelopt_data_path)
    data_collator = LanguageDataCollator(
        tokenizer=tokenizer,
        train_len=args.seq_len,
        return_labels=True,
        answer_only_loss=False,
        shift_labels=True,
    )
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.can_return_loss = True
    print(f"Training ModelOpt EAGLE-3 for {args.max_steps} steps on {len(train_dataset)} rows...")
    trainer.train()
    trainer.save_state()
    trainer.save_model(str(output_dir))

    metadata = {
        "model_path": args.model_path,
        "train_data": str(Path(args.train_data).resolve()),
        "modelopt_data": str(modelopt_data_path.resolve()),
        "max_steps": args.max_steps,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "loss_decay": args.loss_decay,
        "ttt_steps": args.ttt_steps,
        "draft_layers": args.draft_layers,
        "selected_layers": list(model.eagle_config.eagle_aux_hidden_state_layer_ids),
        "official_arch_defaults": args.official_arch_defaults,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "codex_modelopt_experiment.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def export_modelopt(args: argparse.Namespace) -> Path:
    runtime = import_modelopt_runtime()
    torch = runtime["torch"]
    mto = runtime["mto"]
    export_speculative_decoding = runtime["export_speculative_decoding"]
    load_vlm_or_llm = runtime["load_vlm_or_llm"]
    patch_transformers5_params_loading = runtime["patch_transformers5_params_loading"]

    mto.enable_huggingface_checkpointing()
    raw_export_dir = Path(args.raw_export_dir)
    if raw_export_dir.exists() and args.overwrite_exports:
        shutil.rmtree(raw_export_dir)
    raw_export_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ModelOpt checkpoint {args.output_dir} to {raw_export_dir}...")
    with patch_transformers5_params_loading():
        model = load_vlm_or_llm(
            args.output_dir,
            dtype="auto",
            trust_remote_code=args.trust_remote_code,
        )
    model.eval()
    with torch.inference_mode():
        export_speculative_decoding(model, export_dir=raw_export_dir)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return raw_export_dir


def convert_modelopt_export_to_vllm(args: argparse.Namespace) -> Path:
    from transformers import AutoConfig, AutoTokenizer

    raw_export_dir = Path(args.raw_export_dir)
    vllm_export_dir = Path(args.vllm_export_dir)
    if vllm_export_dir.exists() and args.overwrite_exports:
        shutil.rmtree(vllm_export_dir)
    vllm_export_dir.mkdir(parents=True, exist_ok=True)

    draft_cfg_path = raw_export_dir / "config.json"
    draft_model_path = raw_export_dir / "model.safetensors"
    if not draft_cfg_path.exists() or not draft_model_path.exists():
        raise FileNotFoundError(f"ModelOpt raw export is incomplete at {raw_export_dir}")

    draft_cfg = json.loads(draft_cfg_path.read_text(encoding="utf-8"))
    verifier_config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    verifier_dict = verifier_config.to_dict()
    transformer_keys = (
        "attention_bias",
        "attention_dropout",
        "head_dim",
        "hidden_act",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "max_position_embeddings",
        "mlp_bias",
        "model_type",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "pretraining_tp",
        "rms_norm_eps",
        "rope_scaling",
        "rope_theta",
        "use_cache",
        "vocab_size",
    )
    transformer_layer_config = {key: draft_cfg.get(key) for key in transformer_keys}
    transformer_layer_config["use_cache"] = True

    eagle_cfg = draft_cfg.get("eagle_config", {})
    aux_layers = eagle_cfg.get("eagle_aux_hidden_state_layer_ids") or []
    config = {
        "architectures": ["Eagle3DraftModel"],
        "auto_map": {"": "config.Eagle3SpeculatorConfig"},
        "base_model_ep_plan": None,
        "draft_vocab_size": int(draft_cfg["draft_vocab_size"]),
        "dtype": str(draft_cfg.get("torch_dtype", "bfloat16")).replace("torch.", ""),
        "eagle_aux_hidden_state_layer_ids": aux_layers,
        "embed_requires_grad": False,
        "has_no_defaults_at_init": False,
        "norm_before_residual": True,
        "speculators_config": {
            "algorithm": "eagle3",
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {
                    "accept_tolerance": 0.0,
                    "proposal_type": "greedy",
                    "speculative_tokens": int(args.draft_len),
                    "verifier_accept_k": 1,
                }
            ],
            "verifier": {
                "architectures": verifier_dict.get("architectures", []),
                "name_or_path": args.model_path,
            },
        },
        "speculators_model_type": "eagle3",
        "speculators_version": "0.4.0.1",
        "target_hidden_size": verifier_dict.get("hidden_size"),
        "transformer_layer_config": transformer_layer_config,
        "transformers_version": draft_cfg.get("transformers_version"),
    }

    shutil.copyfile(draft_model_path, vllm_export_dir / "model.safetensors")
    (vllm_export_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (vllm_export_dir / "config.py").write_text(EAGLE3_CONFIG_PY, encoding="utf-8")
    AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code).save_pretrained(
        vllm_export_dir
    )
    metadata = {
        "raw_export_dir": str(raw_export_dir.resolve()),
        "model_path": args.model_path,
        "draft_len": args.draft_len,
        "selected_layers": aux_layers,
    }
    (vllm_export_dir / "codex_modelopt_vllm_export.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote vLLM/speculators draft checkpoint to {vllm_export_dir}")
    return vllm_export_dir


def run_vllm_benchmark(args: argparse.Namespace) -> None:
    ensure_project_root_on_path()
    command = [
        sys.executable,
        "methods/eagle3/inference/infer_vllm.py",
        "--model-path",
        args.model_path,
        "--draft-model-path",
        str(Path(args.vllm_export_dir).resolve()),
        "--skip-export",
        "--prompts",
        args.prompts,
        "--output",
        args.summary_output,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--draft-len",
        str(args.draft_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--draft-tensor-parallel-size",
        str(args.draft_tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
    ]
    if args.serial_prompts:
        command.append("--serial-prompts")
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.parallel_drafting:
        command.append("--parallel-drafting")
    baseline_summary_path = (
        Path(args.baseline_summary_path)
        if args.baseline_summary_path
        else Path(args.summary_output).with_name(
            Path(args.summary_output).name.replace(".summary.json", ".baseline.json")
        )
    )
    command.extend(["--baseline-summary-path", str(baseline_summary_path)])

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    print("Running vLLM benchmark:")
    print(" ".join(command))
    subprocess.run(command, check=True, env=env)


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    ensure_project_root_on_path()

    if args.mode in {"prepare", "all"}:
        prepare_modelopt_data(args)
    if args.mode in {"train", "all"}:
        train_modelopt(args)
    if args.mode in {"export", "all"}:
        export_modelopt(args)
    if args.mode in {"convert", "all"}:
        convert_modelopt_export_to_vllm(args)
    if args.mode in {"bench", "all"}:
        run_vllm_benchmark(args)


if __name__ == "__main__":
    main()

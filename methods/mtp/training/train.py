"""
Train a DeepSeek-V3-style multi-token prediction (MTP) head.

The target model is frozen.  Each MTP layer receives the previous hidden state
and the embedding of the next accepted token, then predicts one additional
future token.  Checkpoints are saved in vLLM's Qwen2/MiMo MTP layout.
"""

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoTokenizer

from common.qwen3 import LayerCache, Qwen3Config, Qwen3DecoderLayer, Qwen3ForCausalLM, RMSNorm
from common.tokenizer import load_tokenizer
from methods.draft_model.training.train import TrainingExample, build_distillation_examples, parse_dtype


DEFAULT_TARGET_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TRAIN_DATA = "data/ultrachat_3000_trunc1024_qwen25_7b_greedy128_ids.jsonl"
DEFAULT_EVAL_DATA = "data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl"
DEFAULT_OUTPUT = "checkpoints/mtp_qwen25_7b_eval100_steps1"
DEFAULT_SEQ_LEN = 1152
IGNORE_INDEX = -100


@dataclass(slots=True)
class MTPConfig:
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    num_nextn_predict_layers: int
    target_num_hidden_layers: int

    @classmethod
    def from_qwen(cls, config: Qwen3Config, num_nextn_predict_layers: int) -> "MTPConfig":
        return cls(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            num_nextn_predict_layers=num_nextn_predict_layers,
            target_num_hidden_layers=config.num_hidden_layers,
        )

    def to_block_config(self) -> Qwen3Config:
        return Qwen3Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=1,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            rms_norm_eps=self.rms_norm_eps,
            tie_word_embeddings=False,
        )


class MTPLayer(nn.Module):
    def __init__(self, config: MTPConfig) -> None:
        super().__init__()
        self.token_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hidden_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.input_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.mtp_block = Qwen3DecoderLayer(config.to_block_config())
        self.final_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_embeds: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        layer_cache: LayerCache | None = None,
    ) -> tuple[torch.Tensor, LayerCache]:
        hidden_states = self.input_proj(
            torch.cat(
                [
                    self.hidden_layernorm(previous_hidden_states),
                    self.token_layernorm(input_embeds),
                ],
                dim=-1,
            )
        )
        hidden_states, cache = self.mtp_block(hidden_states, position_ids=position_ids, layer_cache=layer_cache)
        return self.final_layernorm(hidden_states), cache


class MTPModel(nn.Module):
    def __init__(self, config: MTPConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.mtp_layers = nn.ModuleList([MTPLayer(config) for _ in range(config.num_nextn_predict_layers)])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward_teacher_forced(
        self,
        target_hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        max_depth: int | None = None,
    ) -> list[torch.Tensor]:
        depth_count = self.config.num_nextn_predict_layers if max_depth is None else max_depth
        previous_hidden = target_hidden_states
        logits_by_depth: list[torch.Tensor] = []
        for depth, layer in enumerate(self.mtp_layers[:depth_count]):
            input_ids = token_ids[:, depth + 1 : -(depth_count - depth)]
            input_embeds = self.embed_tokens(input_ids)
            position_ids = torch.arange(
                depth + 1,
                depth + 1 + input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(input_ids.shape[0], -1)
            previous_hidden, _ = layer(input_embeds, previous_hidden, position_ids=position_ids)
            logits_by_depth.append(self.lm_head(previous_hidden))
        return logits_by_depth

    def forward_step(
        self,
        previous_hidden_state: torch.Tensor,
        input_token_id: int,
        depth: int,
        position_id: int,
        cache: LayerCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, LayerCache]:
        if depth >= len(self.mtp_layers):
            raise ValueError(f"depth {depth} exceeds configured MTP layers")
        input_ids = torch.tensor([[int(input_token_id)]], dtype=torch.long, device=previous_hidden_state.device)
        input_embeds = self.embed_tokens(input_ids)
        position_ids = torch.tensor([[int(position_id)]], dtype=torch.long, device=previous_hidden_state.device)
        hidden, updated_cache = self.mtp_layers[depth](
            input_embeds,
            previous_hidden_state[:, None, :],
            position_ids=position_ids,
            layer_cache=cache,
        )
        logits = self.lm_head(hidden[:, -1, :])
        return logits, hidden[:, -1, :], updated_cache


def _copy_layer_weights(dst: MTPLayer, src) -> None:
    dst.mtp_block.load_state_dict(copy.deepcopy(src.state_dict()))


def build_mtp_from_target(target_model: Qwen3ForCausalLM, num_nextn_predict_layers: int) -> MTPModel:
    config = MTPConfig.from_qwen(target_model.config, num_nextn_predict_layers)
    model = MTPModel(config)
    model.embed_tokens.weight.data.copy_(target_model.model.embed_tokens.weight.detach())
    model.lm_head.weight.data.copy_(target_model.lm_head.weight.detach())
    start = max(0, len(target_model.model.layers) - num_nextn_predict_layers)
    for mtp_layer, target_layer in zip(model.mtp_layers, target_model.model.layers[start:], strict=True):
        _copy_layer_weights(mtp_layer, target_layer)
    return model


def _select_batch(
    examples: Sequence[TrainingExample],
    order: list[int],
    *,
    step: int,
    batch_size: int,
    rng: random.Random,
) -> list[TrainingExample]:
    start = (step * batch_size) % len(order)
    if start == 0 and step > 0:
        rng.shuffle(order)
    return [examples[order[(start + offset) % len(order)]] for offset in range(batch_size)]


def collate_examples(
    examples: Sequence[TrainingExample],
    *,
    pad_token_id: int,
    num_nextn_predict_layers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(example.token_ids.numel() for example in examples)
    min_len = num_nextn_predict_layers + 3
    max_len = max(max_len, min_len)
    input_ids = torch.full((len(examples), max_len), int(pad_token_id), dtype=torch.long)
    loss_mask = torch.zeros((len(examples), max_len), dtype=torch.float32)
    for row, example in enumerate(examples):
        length = example.token_ids.numel()
        input_ids[row, :length] = example.token_ids
        loss_mask[row, :length] = example.loss_mask
    return input_ids, loss_mask


def mtp_loss(
    model: MTPModel,
    target_model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    depth_count = model.config.num_nextn_predict_layers
    with torch.no_grad():
        output = target_model(input_ids[:, : -(depth_count + 1)], output_hidden_states=True)
        target_hidden = output.hidden_states[target_model.config.num_hidden_layers]

    logits_by_depth = model.forward_teacher_forced(target_hidden, input_ids, max_depth=depth_count)
    losses: list[torch.Tensor] = []
    accuracies: list[float] = []
    for depth, logits in enumerate(logits_by_depth):
        labels = input_ids[:, depth + 2 : depth + 2 + logits.shape[1]]
        mask = loss_mask[:, depth + 2 : depth + 2 + logits.shape[1]] > 0
        if not mask.any():
            continue
        flat_logits = logits[mask]
        flat_labels = labels[mask]
        losses.append(F.cross_entropy(flat_logits.float(), flat_labels))
        accuracies.append(float((flat_logits.argmax(dim=-1) == flat_labels).float().mean().item()))
    if not losses:
        raise ValueError("batch did not contain any MTP loss positions")
    loss = torch.stack(losses).mean()
    metrics = {
        "loss": float(loss.detach().item()),
        "mean_accuracy": sum(accuracies) / len(accuracies),
    }
    for index, value in enumerate(accuracies, start=1):
        metrics[f"accuracy_depth_{index}"] = value
    return loss, metrics


def save_mtp_checkpoint(
    model: MTPModel,
    output_dir: str | Path,
    *,
    target_model_path: str,
    tokenizer,
    train_summary: dict[str, object],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    save_file(state, str(output / "mtp_model.safetensors"))
    metadata = {
        "target_model_path": target_model_path,
        "mtp_config": asdict(model.config),
        "train_summary": train_summary,
        "vllm_export": "vllm_export",
    }
    (output / "mtp_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    tokenizer.save_pretrained(output)
    export_mtp_for_vllm(model, output / "vllm_export", target_model_path=target_model_path, tokenizer=tokenizer)


def export_mtp_for_vllm(
    model: MTPModel,
    export_dir: str | Path,
    *,
    target_model_path: str,
    tokenizer,
) -> None:
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    hf_config = AutoConfig.from_pretrained(target_model_path, trust_remote_code=True)
    hf_dict = hf_config.to_dict()
    hf_dict.update(
        {
            # vLLM's SpeculativeConfig rewrites this MiMo marker to the
            # inference-only MiMoMTPModel while preserving Qwen2 config parsing.
            "architectures": ["MiMoForCausalLM"],
            "model_type": "qwen2",
            "num_nextn_predict_layers": model.config.num_nextn_predict_layers,
            "n_predict": model.config.num_nextn_predict_layers,
            "vocab_size": model.config.vocab_size,
        }
    )
    (export_path / "config.json").write_text(json.dumps(hf_dict, indent=2), encoding="utf-8")
    tokenizer.save_pretrained(export_path)

    state = model.state_dict()
    export_state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": state["embed_tokens.weight"].detach().clone(),
        "lm_head.weight": state["lm_head.weight"].detach().clone(),
    }
    for layer_idx in range(model.config.num_nextn_predict_layers):
        src_prefix = f"mtp_layers.{layer_idx}."
        dst_prefix = f"model.mtp_layers.{layer_idx}."
        for suffix in (
            "token_layernorm.weight",
            "hidden_layernorm.weight",
            "input_proj.weight",
            "final_layernorm.weight",
            "mtp_block.input_layernorm.weight",
            "mtp_block.post_attention_layernorm.weight",
            "mtp_block.self_attn.q_proj.weight",
            "mtp_block.self_attn.q_proj.bias",
            "mtp_block.self_attn.k_proj.weight",
            "mtp_block.self_attn.k_proj.bias",
            "mtp_block.self_attn.v_proj.weight",
            "mtp_block.self_attn.v_proj.bias",
            "mtp_block.self_attn.o_proj.weight",
            "mtp_block.mlp.gate_proj.weight",
            "mtp_block.mlp.up_proj.weight",
            "mtp_block.mlp.down_proj.weight",
        ):
            key = src_prefix + suffix
            if key not in state:
                continue
            export_suffix = suffix.replace("mtp_block.", "")
            export_state[dst_prefix + export_suffix] = state[key].detach().clone()
    save_file(export_state, str(export_path / "model.safetensors"))


def load_mtp_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[MTPModel, dict[str, object]]:
    checkpoint = Path(checkpoint_path)
    metadata = json.loads((checkpoint / "mtp_metadata.json").read_text(encoding="utf-8"))
    config = MTPConfig(**metadata["mtp_config"])
    model = MTPModel(config)
    model.load_state_dict(load_file(str(checkpoint / "mtp_model.safetensors")))
    if dtype is not None:
        model = model.to(dtype=dtype)
    model = model.to(device)
    model.eval()
    return model, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DeepSeek-style MTP head for Qwen2.5-7B.")
    parser.add_argument("--target-model-path", default=DEFAULT_TARGET_MODEL_PATH)
    parser.add_argument("--data", default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--eval-data", default=DEFAULT_EVAL_DATA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--num-speculative-steps", type=int, choices=(1, 2), default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.target_model_path)
    target_model = Qwen3ForCausalLM.from_pretrained(args.target_model_path, device=args.device, dtype=dtype)
    target_model.eval()
    for parameter in target_model.parameters():
        parameter.requires_grad_(False)

    train_examples = build_distillation_examples(args.data, tokenizer, seq_len=args.seq_len)
    if args.limit > 0:
        train_examples = train_examples[: args.limit]
    eval_examples = build_distillation_examples(args.eval_data, tokenizer, seq_len=args.seq_len)
    model = build_mtp_from_target(target_model, args.num_speculative_steps).to(device=args.device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    order = list(range(len(train_examples)))
    rng = random.Random(args.seed)
    rng.shuffle(order)
    last_metrics: dict[str, float] = {}

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(args.steps):
        batch_examples = _select_batch(
            train_examples,
            order,
            step=step,
            batch_size=args.batch_size,
            rng=rng,
        )
        input_ids, loss_mask = collate_examples(
            batch_examples,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            num_nextn_predict_layers=args.num_speculative_steps,
        )
        input_ids = input_ids.to(args.device)
        loss_mask = loss_mask.to(args.device)
        loss, metrics = mtp_loss(model, target_model, input_ids, loss_mask)
        (loss / args.grad_accum).backward()
        if (step + 1) % args.grad_accum == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        last_metrics = metrics
        if step == 0 or (step + 1) % 25 == 0:
            print(
                f"step={step + 1} loss={metrics['loss']:.4f} "
                f"mean_accuracy={metrics['mean_accuracy']:.4f}",
                flush=True,
            )

    model.eval()
    eval_batch = eval_examples[: min(8, len(eval_examples))]
    with torch.no_grad():
        input_ids, loss_mask = collate_examples(
            eval_batch,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            num_nextn_predict_layers=args.num_speculative_steps,
        )
        _, eval_metrics = mtp_loss(model, target_model, input_ids.to(args.device), loss_mask.to(args.device))
    summary = {
        "steps": args.steps,
        "num_speculative_steps": args.num_speculative_steps,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "last_train_metrics": last_metrics,
        "eval_metrics": eval_metrics,
    }
    save_mtp_checkpoint(
        model,
        args.output,
        target_model_path=args.target_model_path,
        tokenizer=tokenizer,
        train_summary=summary,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

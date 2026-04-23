from __future__ import annotations

"""
Minimal EAGLE-3 trainer: fuse low/mid/high target hidden states, then predict a
short draft sequence with a lightweight transformer drafter.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

from common.qwen3 import LayerCache, Qwen3ForCausalLM, RMSNorm, RotaryEmbedding, apply_rotary_embeddings
from common.tokenizer import load_tokenizer
from methods.draft_model.training.train import (
    build_distillation_examples,
    build_training_sequences,
    dataset_uses_completion,
    parse_dtype,
)


@dataclass(slots=True)
class Eagle3Config:
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    selected_layers: tuple[int, int, int]
    draft_len: int
    ttt_steps: int = 0


class Eagle3Attention(nn.Module):
    def __init__(self, config: Eagle3Config, input_size: int) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_queries_per_kv = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(input_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(input_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(input_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, base=config.rope_theta)

    def _reshape(self, tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch, seq_len, _ = tensor.shape
        return tensor.view(batch, seq_len, num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        layer_cache: LayerCache | None = None,
    ) -> tuple[torch.Tensor, LayerCache]:
        batch_size, query_len, _ = hidden_states.shape
        query = self._reshape(self.q_proj(hidden_states), self.num_attention_heads)
        key = self._reshape(self.k_proj(hidden_states), self.num_key_value_heads)
        value = self._reshape(self.v_proj(hidden_states), self.num_key_value_heads)
        cos, sin = self.rotary_emb(position_ids)
        query, key = apply_rotary_embeddings(query, key, cos, sin)

        if layer_cache is not None:
            key = torch.cat([layer_cache.key, key], dim=2)
            value = torch.cat([layer_cache.value, value], dim=2)
        updated_cache = LayerCache(key=key, value=value)

        if self.num_queries_per_kv > 1:
            key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=1)
        key_len = key.shape[2]
        past_len = key_len - query_len
        query_positions = torch.arange(past_len, past_len + query_len, device=hidden_states.device).unsqueeze(-1)
        key_positions = torch.arange(key_len, device=hidden_states.device).unsqueeze(0)
        attn_mask = (key_positions <= query_positions).unsqueeze(0).unsqueeze(0)

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size,
            query_len,
            self.num_attention_heads * self.head_dim,
        )
        return self.o_proj(attn_output), updated_cache


class Eagle3MLP(nn.Module):
    def __init__(self, config: Eagle3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Eagle3DecoderLayer(nn.Module):
    def __init__(self, config: Eagle3Config, *, first_layer: bool) -> None:
        super().__init__()
        self.first_layer = first_layer
        input_size = 2 * config.hidden_size if first_layer else config.hidden_size
        self.self_attn = Eagle3Attention(config, input_size=input_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Eagle3MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        layer_cache: LayerCache | None = None,
    ) -> tuple[torch.Tensor, LayerCache]:
        residual = hidden_states
        if self.first_layer:
            attention_input = torch.cat(
                [self.input_layernorm(input_embeds), self.hidden_norm(hidden_states)],
                dim=-1,
            )
        else:
            attention_input = self.input_layernorm(hidden_states)
        attn_output, updated_cache = self.self_attn(attention_input, position_ids, layer_cache)
        hidden_states = residual + attn_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, updated_cache


class Eagle3Drafter(nn.Module):
    def __init__(self, config: Eagle3Config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.feature_fuser = nn.Linear(len(config.selected_layers) * config.hidden_size, config.hidden_size)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Eagle3DecoderLayer(config, first_layer=layer_idx == 0)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def init_state(self, fused_features: torch.Tensor) -> torch.Tensor:
        return fused_features

    def forward_step(
        self,
        fused_features: torch.Tensor,
        prev_token_ids: torch.Tensor,
        cache: list[LayerCache | None] | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[LayerCache | None]]:
        if fused_features.ndim != 2:
            raise ValueError(f"expected [batch, hidden] fused_features, got {tuple(fused_features.shape)}")
        if cache is None:
            cache = [None] * len(self.layers)
        if position_ids is None:
            past_len = 0 if cache[0] is None else cache[0].key.shape[2]
            position_ids = torch.full(
                (fused_features.shape[0], 1),
                past_len,
                dtype=torch.long,
                device=fused_features.device,
            )
        input_embeds = self.token_embedding(prev_token_ids).unsqueeze(1)
        hidden_states = fused_features.unsqueeze(1)
        updated_cache: list[LayerCache | None] = []
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            hidden_states, next_cache = layer(
                hidden_states=hidden_states,
                input_embeds=input_embeds,
                position_ids=position_ids,
                layer_cache=layer_cache,
            )
            updated_cache.append(next_cache)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states[:, -1, :])
        return logits, hidden_states[:, -1, :], updated_cache


def initialize_drafter_from_target(
    drafter: Eagle3Drafter,
    target_model: Qwen3ForCausalLM,
    *,
    num_fused_layers: int,
) -> None:
    hidden_size = drafter.hidden_size
    if num_fused_layers <= 0:
        raise ValueError("num_fused_layers must be positive")

    with torch.no_grad():
        drafter.token_embedding.weight.copy_(target_model.model.embed_tokens.weight.detach())
        drafter.lm_head.weight.copy_(target_model.lm_head.weight.detach())
        drafter.norm.weight.copy_(target_model.model.norm.weight.detach())

        drafter.feature_fuser.weight.zero_()
        drafter.feature_fuser.bias.zero_()
        scale = 1.0 / float(num_fused_layers)
        for layer_offset in range(num_fused_layers):
            start = layer_offset * hidden_size
            end = start + hidden_size
            drafter.feature_fuser.weight[:, start:end].add_(torch.eye(hidden_size) * scale)

        for draft_layer, target_layer in zip(drafter.layers, target_model.model.layers[-len(drafter.layers) :], strict=True):
            draft_layer.self_attn.q_proj.weight.zero_()
            draft_layer.self_attn.k_proj.weight.zero_()
            draft_layer.self_attn.v_proj.weight.zero_()
            source_start = hidden_size if draft_layer.first_layer else 0
            source_end = source_start + hidden_size
            draft_layer.self_attn.q_proj.weight[:, source_start:source_end].copy_(target_layer.self_attn.q_proj.weight.detach())
            draft_layer.self_attn.k_proj.weight[:, source_start:source_end].copy_(target_layer.self_attn.k_proj.weight.detach())
            draft_layer.self_attn.v_proj.weight[:, source_start:source_end].copy_(target_layer.self_attn.v_proj.weight.detach())
            draft_layer.self_attn.q_proj.bias.copy_(target_layer.self_attn.q_proj.bias.detach())
            draft_layer.self_attn.k_proj.bias.copy_(target_layer.self_attn.k_proj.bias.detach())
            draft_layer.self_attn.v_proj.bias.copy_(target_layer.self_attn.v_proj.bias.detach())
            draft_layer.self_attn.o_proj.weight.copy_(target_layer.self_attn.o_proj.weight.detach())
            draft_layer.input_layernorm.weight.copy_(target_layer.input_layernorm.weight.detach())
            draft_layer.hidden_norm.weight.copy_(target_layer.input_layernorm.weight.detach())
            draft_layer.post_attention_layernorm.weight.copy_(target_layer.post_attention_layernorm.weight.detach())
            draft_layer.mlp.gate_proj.weight.copy_(target_layer.mlp.gate_proj.weight.detach())
            draft_layer.mlp.up_proj.weight.copy_(target_layer.mlp.up_proj.weight.detach())
            draft_layer.mlp.down_proj.weight.copy_(target_layer.mlp.down_proj.weight.detach())


def freeze_base_model(model: Qwen3ForCausalLM) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def default_selected_layers(num_layers: int) -> tuple[int, int, int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    return (
        min(num_layers - 1, 1),
        max(0, num_layers // 2 - 1),
        max(0, num_layers - 4),
    )


def parse_selected_layers(value: str) -> tuple[int, int, int]:
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(layers) != 3:
        raise ValueError("--selected-layers must provide exactly 3 comma-separated indices")
    return layers


def fuse_hidden_states(hidden_state_map: dict[int, torch.Tensor], selected_layers: Sequence[int]) -> torch.Tensor:
    tensors = [hidden_state_map[layer_idx] for layer_idx in selected_layers]
    if len(tensors) != 3:
        raise ValueError("selected_layers must contain exactly three layer indices")
    return torch.cat(tensors, dim=-1)


def run_drafter_training_step(
    drafter: Eagle3Drafter,
    fused_features: torch.Tensor,
    context_last_token: torch.Tensor,
    labels: torch.Tensor,
    mode: str,
    start_position: torch.Tensor | None = None,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    fused = drafter.feature_fuser(fused_features)
    state = drafter.init_state(fused)
    cache: list[LayerCache | None] | None = None
    prev_tokens = context_last_token
    for step in range(labels.shape[1]):
        position_ids = None
        if start_position is not None:
            position_ids = (start_position + step).view(-1, 1)
        logits, state, cache = drafter.forward_step(
            state,
            prev_tokens,
            cache=cache,
            position_ids=position_ids,
        )
        losses.append(torch.nn.functional.cross_entropy(logits, labels[:, step]))
        if mode == "teacher_forcing":
            prev_tokens = labels[:, step]
        elif mode == "training_time_test":
            prev_tokens = torch.argmax(logits, dim=-1)
        else:
            raise ValueError(f"unsupported mode: {mode}")
    return torch.stack(losses).sum()


def train_eagle3(
    target_model: Qwen3ForCausalLM,
    drafter: Eagle3Drafter,
    sequences: Sequence[torch.Tensor],
    *,
    selected_layers: Sequence[int],
    draft_len: int,
    ttt_steps: int,
    steps: int,
    batch_size: int,
    lr: float,
    device: str | torch.device,
    mode: str,
) -> list[float]:
    target_model = target_model.to(device=device)
    drafter = drafter.to(device=device)
    optimizer = torch.optim.AdamW(drafter.parameters(), lr=lr)
    rollout_len = ttt_steps if mode == "training_time_test" else draft_len
    if rollout_len < 1:
        raise ValueError("rollout_len must be positive")
    valid_sequences = [sequence for sequence in sequences if sequence.numel() > rollout_len]
    if not valid_sequences:
        raise ValueError("no sequences are long enough for EAGLE-3 training")
    losses: list[float] = []

    for step in range(steps):
        batch_items = [valid_sequences[(step + idx) % len(valid_sequences)] for idx in range(batch_size)]
        prefixes = [item[:-rollout_len] for item in batch_items]
        labels = torch.stack([item[-rollout_len:] for item in batch_items]).to(device=device)
        max_len = max(prefix.numel() for prefix in prefixes)
        batch = torch.zeros((batch_size, max_len), dtype=torch.long)
        for row, prefix in enumerate(prefixes):
            batch[row, : prefix.numel()] = prefix
        batch = batch.to(device=device)
        with torch.no_grad():
            outputs = target_model(
                batch,
                output_hidden_states=True,
                hidden_state_indices=selected_layers,
            )
        fused = fuse_hidden_states(outputs.hidden_states, selected_layers)
        fused_last = fused[:, -1, :]
        context_last_token = batch[:, -1]
        start_position = torch.tensor(
            [prefix.numel() - 1 for prefix in prefixes],
            dtype=torch.long,
            device=device,
        )
        loss = run_drafter_training_step(
            drafter,
            fused_last,
            context_last_token,
            labels,
            mode,
            start_position=start_position,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses


def save_eagle3_checkpoint(
    drafter: Eagle3Drafter,
    checkpoint_dir: str | Path,
    config: Eagle3Config,
    target_model_path: str,
) -> None:
    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.detach().cpu() for name, tensor in drafter.state_dict().items()},
        str(output_dir / "drafter.safetensors"),
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "eagle3_config": asdict(config),
                "target_model_path": target_model_path,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_eagle3_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[Eagle3Drafter, dict[str, object]]:
    checkpoint_path = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_path / "config.json").read_text(encoding="utf-8"))
    config = Eagle3Config(**metadata["eagle3_config"])
    drafter = Eagle3Drafter(config)
    drafter.load_state_dict(load_file(str(checkpoint_path / "drafter.safetensors")), strict=True)
    drafter = drafter.to(device=device)
    return drafter, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal EAGLE-3 drafter.")
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--draft-len", type=int, default=4)
    parser.add_argument("--ttt-len", type=int, default=None)
    parser.add_argument("--num-draft-layers", type=int, default=1)
    parser.add_argument("--selected-layers", default="")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher-forcing-only", action="store_true")
    parser.add_argument("--training-time-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ttt_len is not None and not args.training_time_test:
        raise ValueError("--ttt-len requires --training-time-test")
    dtype = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.target_model_path)
    if dataset_uses_completion(args.data):
        sequences = [
            example.token_ids
            for example in build_distillation_examples(args.data, tokenizer=tokenizer, seq_len=args.seq_len)
        ]
    else:
        sequences = build_training_sequences(args.data, tokenizer=tokenizer, seq_len=args.seq_len)
    target_model = Qwen3ForCausalLM.from_pretrained(args.target_model_path, device=args.device, dtype=dtype)
    freeze_base_model(target_model)
    num_layers = target_model.config.num_hidden_layers
    selected_layers = (
        parse_selected_layers(args.selected_layers)
        if args.selected_layers
        else default_selected_layers(num_layers)
    )
    config = Eagle3Config(
        hidden_size=target_model.config.hidden_size,
        vocab_size=target_model.config.vocab_size,
        intermediate_size=target_model.config.intermediate_size,
        num_hidden_layers=args.num_draft_layers,
        num_attention_heads=target_model.config.num_attention_heads,
        num_key_value_heads=target_model.config.num_key_value_heads,
        rms_norm_eps=target_model.config.rms_norm_eps,
        rope_theta=target_model.config.rope_theta,
        selected_layers=selected_layers,
        draft_len=args.draft_len,
        ttt_steps=args.ttt_len if args.ttt_len is not None else args.draft_len,
    )
    drafter = Eagle3Drafter(config)
    if args.device != "cpu":
        drafter = drafter.to(dtype=dtype)
    initialize_drafter_from_target(
        drafter,
        target_model,
        num_fused_layers=len(selected_layers),
    )
    mode = "training_time_test" if args.training_time_test else "teacher_forcing"
    losses = train_eagle3(
        target_model=target_model,
        drafter=drafter,
        sequences=sequences,
        selected_layers=selected_layers,
        draft_len=args.draft_len,
        ttt_steps=config.ttt_steps,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        mode=mode,
    )
    save_eagle3_checkpoint(
        drafter=drafter,
        checkpoint_dir=args.output,
        config=config,
        target_model_path=args.target_model_path,
    )
    (Path(args.output) / "training_summary.json").write_text(
        json.dumps(
            {
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "mode": mode,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "draft_len": args.draft_len,
                "ttt_steps": config.ttt_steps,
                "num_draft_layers": config.num_hidden_layers,
                "selected_layers": list(config.selected_layers),
                "dtype": args.dtype,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

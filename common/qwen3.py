from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig


@dataclass(slots=True)
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool = False
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int = 151643

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_hf_config(cls, config: AutoConfig) -> "Qwen3Config":
        def _normalize_token_id(value: object, default: int) -> int:
            if value is None:
                return default
            if isinstance(value, (list, tuple)):
                if not value:
                    return default
                return int(value[0])
            return int(value)

        return cls(
            vocab_size=int(config.vocab_size),
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.intermediate_size),
            num_hidden_layers=int(config.num_hidden_layers),
            num_attention_heads=int(config.num_attention_heads),
            num_key_value_heads=int(getattr(config, "num_key_value_heads", config.num_attention_heads)),
            max_position_embeddings=int(config.max_position_embeddings),
            rope_theta=float(getattr(config, "rope_theta", 10000.0)),
            rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-6)),
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
            bos_token_id=_normalize_token_id(getattr(config, "bos_token_id", 151643), 151643),
            eos_token_id=_normalize_token_id(getattr(config, "eos_token_id", 151645), 151645),
            pad_token_id=_normalize_token_id(
                getattr(config, "pad_token_id", None),
                _normalize_token_id(getattr(config, "eos_token_id", None), 151645),
            ),
        )

    @classmethod
    def from_pretrained(cls, model_path: str) -> "Qwen3Config":
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return cls.from_hf_config(config)


@dataclass(slots=True)
class LayerCache:
    key: torch.Tensor
    value: torch.Tensor


@dataclass(slots=True)
class Qwen3Output:
    logits: torch.Tensor
    hidden_states: dict[int, torch.Tensor] | None = None
    cache: list[LayerCache | None] | None = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("bi,j->bij", position_ids.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first_half, second_half = tensor.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_embeddings(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.to(dtype=query.dtype)
    sin = sin.to(dtype=query.dtype)
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    query = (query * cos) + (rotate_half(query) * sin)
    key = (key * cos) + (rotate_half(key) * sin)
    return query, key


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_queries_per_kv = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.fast_single_token_gqa = False

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
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
        use_manual_gqa = self.num_queries_per_kv > 1 and not (
            self.fast_single_token_gqa and query_len == 1
        )
        if use_manual_gqa:
            key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=1)
        key_len = key.shape[2]
        past_len = key_len - query_len
        query_positions = torch.arange(past_len, past_len + query_len, device=hidden_states.device).unsqueeze(-1)
        key_positions = torch.arange(key_len, device=hidden_states.device).unsqueeze(0)
        attn_mask = key_positions <= query_positions
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=self.num_queries_per_kv > 1 and not use_manual_gqa,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size,
            query_len,
            self.num_attention_heads * self.head_dim,
        )
        return self.o_proj(attn_output), updated_cache


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        layer_cache: LayerCache | None = None,
    ) -> tuple[torch.Tensor, LayerCache]:
        residual = hidden_states
        attn_output, updated_cache = self.self_attn(
            self.input_layernorm(hidden_states),
            position_ids=position_ids,
            layer_cache=layer_cache,
        )
        hidden_states = residual + attn_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, updated_cache


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: list[LayerCache | None] | None = None,
        output_hidden_states: bool = False,
        hidden_state_indices: Iterable[int] | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor] | None, list[LayerCache | None] | None]:
        if input_ids.ndim != 2:
            raise ValueError(f"expected [batch, seq] input_ids, got {tuple(input_ids.shape)}")

        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        hidden_state_map: dict[int, torch.Tensor] | None = {} if output_hidden_states else None

        if cache is None:
            cache = [None] * len(self.layers)
            start_position = 0
        else:
            if len(cache) != len(self.layers):
                raise ValueError("cache length does not match number of layers")
            first_cache = next((item for item in cache if item is not None), None)
            start_position = 0 if first_cache is None else first_cache.key.shape[2]

        position_ids = torch.arange(
            start_position,
            start_position + seq_len,
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1)

        selected = set(hidden_state_indices or [])
        next_cache: list[LayerCache | None] = []

        for layer_idx, layer in enumerate(self.layers):
            hidden_states, updated_cache = layer(
                hidden_states,
                position_ids=position_ids,
                layer_cache=cache[layer_idx],
            )
            next_cache.append(updated_cache)
            if output_hidden_states and (hidden_state_indices is None or layer_idx in selected):
                hidden_state_map[layer_idx] = hidden_states

        hidden_states = self.norm(hidden_states)
        if output_hidden_states and (
            hidden_state_indices is None or self.config.num_hidden_layers in selected
        ):
            hidden_state_map[self.config.num_hidden_layers] = hidden_states

        return hidden_states, hidden_state_map, next_cache


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def set_fast_single_token_gqa(self, enabled: bool) -> None:
        for layer in self.model.layers:
            layer.self_attn.fast_single_token_gqa = enabled


    def forward(
        self,
        input_ids: torch.Tensor,
        cache: list[LayerCache | None] | None = None,
        output_hidden_states: bool = False,
        hidden_state_indices: Iterable[int] | None = None,
    ) -> Qwen3Output:
        hidden_states, hidden_state_map, next_cache = self.model(
            input_ids=input_ids,
            cache=cache,
            output_hidden_states=output_hidden_states,
            hidden_state_indices=hidden_state_indices,
        )
        logits = self.lm_head(hidden_states)
        return Qwen3Output(logits=logits, hidden_states=hidden_state_map, cache=next_cache)

    def prefill(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
        hidden_state_indices: Iterable[int] | None = None,
    ) -> Qwen3Output:
        return self.forward(
            input_ids=input_ids,
            cache=None,
            output_hidden_states=output_hidden_states,
            hidden_state_indices=hidden_state_indices,
        )

    def decode_one(
        self,
        input_ids: torch.Tensor,
        cache: list[LayerCache | None],
        output_hidden_states: bool = False,
        hidden_state_indices: Iterable[int] | None = None,
    ) -> Qwen3Output:
        if input_ids.shape[1] != 1:
            raise ValueError("decode_one expects a single token per batch element")
        return self.forward(
            input_ids=input_ids,
            cache=cache,
            output_hidden_states=output_hidden_states,
            hidden_state_indices=hidden_state_indices,
        )

    def decode_many(
        self,
        input_ids: torch.Tensor,
        cache: list[LayerCache | None],
        output_hidden_states: bool = False,
        hidden_state_indices: Iterable[int] | None = None,
    ) -> Qwen3Output:
        if input_ids.shape[1] <= 0:
            raise ValueError("decode_many expects at least one token per batch element")
        # Verification needs one cached forward over the full drafted chunk. Iterating
        # token-by-token here collapses speculative decoding back to baseline behavior.
        return self.forward(
            input_ids=input_ids,
            cache=cache,
            output_hidden_states=output_hidden_states,
            hidden_state_indices=hidden_state_indices,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Qwen3ForCausalLM":
        resolved = Path(model_path)
        if not resolved.exists():
            resolved = Path(
                snapshot_download(
                    repo_id=model_path,
                    allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt"],
                )
            )

        config = Qwen3Config.from_pretrained(str(resolved))
        model = cls(config)
        state_dict = _load_state_dict(resolved)
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected checkpoint keys: {incompatible.unexpected_keys[:10]}")
        allowed_missing = {"lm_head.weight"} if config.tie_word_embeddings else set()
        missing = {key for key in incompatible.missing_keys if key not in allowed_missing}
        if missing:
            raise RuntimeError(f"missing checkpoint keys: {sorted(missing)[:10]}")
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device=device)
        return model


def _load_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    safetensor_paths = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_paths:
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as handle:
                index_data = json.load(handle)
            safetensor_paths = sorted({model_dir / value for value in index_data["weight_map"].values()})
    if not safetensor_paths:
        raise FileNotFoundError(f"no safetensors checkpoint found in {model_dir}")

    state_dict: dict[str, torch.Tensor] = {}
    for path in safetensor_paths:
        state_dict.update(load_file(str(path)))
    return state_dict

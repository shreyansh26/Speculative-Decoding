"""
Minimal EAGLE-3 trainer: fuse low/mid/high target hidden states, then predict a
short draft sequence with a lightweight transformer drafter.
"""

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

from common.qwen3 import LayerCache, Qwen3ForCausalLM, RMSNorm, RotaryEmbedding, apply_rotary_embeddings
from common.tokenizer import load_tokenizer, render_prompt
from methods.draft_model.training.train import build_training_sequences, parse_dtype


DEFAULT_TARGET_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATA_PATH = "data/ultrachat_3000_trunc1024_qwen25_7b_greedy128_ids.jsonl"
DEFAULT_EVAL_PATH = "data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl"
DEFAULT_OUTPUT_DIR = "checkpoints/eagle3_qwen25_7b"
DEFAULT_DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
DEFAULT_DATASET_SPLIT = "train_sft"
DEFAULT_MAX_PROMPT_TOKENS = 1024
DEFAULT_COMPLETION_TOKENS = 128


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
    loss_decay: float = 0.9
    norm_before_residual: bool = False


@dataclass(slots=True)
class Eagle3TrainingExample:
    prompt_id: str
    token_ids: torch.Tensor
    loss_mask: torch.Tensor
    prompt_tokens: int
    completion_tokens: int


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
        attention_mask: torch.Tensor | None = None,
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
        if attention_mask is None:
            attn_mask = (key_positions <= query_positions).unsqueeze(0).unsqueeze(0)
        else:
            attn_mask = attention_mask

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
        self.config = config
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
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LayerCache]:
        residual = hidden_states
        if self.first_layer:
            normalized_hidden = self.hidden_norm(hidden_states)
            if self.config.norm_before_residual:
                residual = normalized_hidden
            attention_input = torch.cat(
                [self.input_layernorm(input_embeds), normalized_hidden],
                dim=-1,
            )
        else:
            attention_input = self.input_layernorm(hidden_states)
        attn_output, updated_cache = self.self_attn(
            attention_input,
            position_ids,
            layer_cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + attn_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, updated_cache


class Eagle3Drafter(nn.Module):
    def __init__(self, config: Eagle3Config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.feature_fuser = nn.Linear(
            len(config.selected_layers) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
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

    def forward_sequence(
        self,
        hidden_states: torch.Tensor,
        input_token_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache: list[LayerCache | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[LayerCache | None]]:
        if hidden_states.ndim != 3:
            raise ValueError(f"expected [batch, seq, hidden] hidden_states, got {tuple(hidden_states.shape)}")
        if input_token_ids.ndim != 2:
            raise ValueError(f"expected [batch, seq] input_token_ids, got {tuple(input_token_ids.shape)}")
        batch_size, seq_len, _ = hidden_states.shape
        if cache is None:
            cache = [None] * len(self.layers)
            past_len = 0
        else:
            first_cache = next((item for item in cache if item is not None), None)
            past_len = 0 if first_cache is None else first_cache.key.shape[2]
        if position_ids is None:
            position_ids = torch.arange(
                past_len,
                past_len + seq_len,
                dtype=torch.long,
                device=hidden_states.device,
            ).unsqueeze(0).expand(batch_size, -1)

        input_embeds = self.token_embedding(input_token_ids)
        updated_cache: list[LayerCache | None] = []
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            hidden_states, next_cache = layer(
                hidden_states=hidden_states,
                input_embeds=input_embeds,
                position_ids=position_ids,
                layer_cache=layer_cache,
                attention_mask=attention_mask,
            )
            updated_cache.append(next_cache)
        pre_norm_hidden_states = hidden_states
        logits = self.lm_head(self.norm(hidden_states))
        return logits, pre_norm_hidden_states, updated_cache

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


def _row_prompt_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        prompt = row.get("prompt")
        if prompt is None:
            return None
        return [{"role": "user", "content": str(prompt)}]

    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", ""))
        if role in {"system", "user", "assistant"} and content:
            normalized.append({"role": role, "content": content})
    user_positions = [idx for idx, message in enumerate(normalized) if message["role"] == "user"]
    if not user_positions:
        return None
    return normalized[: user_positions[-1] + 1]


def _make_prompt_id(row: dict[str, Any], index: int) -> str:
    for key in ("prompt_id", "id", "conversation_id"):
        if row.get(key) is not None:
            return str(row[key])
    return f"ultrachat_{index:06d}"


def prepare_ultrachat_data(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - runtime dependency for data prep
        raise SystemExit(
            "Dataset prep requires `datasets` and `vllm` in the active environment."
        ) from exc

    tokenizer = load_tokenizer(args.target_model_path)
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)

    prompt_rows: list[dict[str, Any]] = []
    for index in indices:
        row = dict(dataset[index])
        messages = _row_prompt_messages(row)
        if not messages:
            continue
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(prompt_ids) > args.max_prompt_tokens:
            prompt_ids = prompt_ids[-args.max_prompt_tokens :]
        if not prompt_ids:
            continue
        prompt_rows.append(
            {
                "prompt_id": _make_prompt_id(row, index),
                "messages": messages,
                "prompt": tokenizer.decode(prompt_ids, skip_special_tokens=False),
                "prompt_ids": [int(token_id) for token_id in prompt_ids],
                "prompt_tokens": len(prompt_ids),
            }
        )
        if len(prompt_rows) >= args.num_samples:
            break

    if len(prompt_rows) < args.num_samples:
        raise ValueError(
            f"only built {len(prompt_rows)} prompts from {args.dataset_name}:{args.dataset_split}; "
            f"requested {args.num_samples}"
        )

    dtype = "bfloat16" if args.dtype == "bf16" else args.dtype
    llm = LLM(
        model=args.target_model_path,
        trust_remote_code=True,
        dtype=dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.completion_tokens,
    )
    prompt_inputs = [{"prompt_token_ids": row["prompt_ids"]} for row in prompt_rows]
    outputs = llm.generate(prompt_inputs, sampling_params=sampling_params)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path = Path(args.eval_output)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row, output in zip(prompt_rows, outputs, strict=True):
        completion_ids = [int(token_id) for token_id in output.outputs[0].token_ids]
        record = {
            **row,
            "completion": output.outputs[0].text,
            "completion_ids": completion_ids,
            "completion_tokens": len(completion_ids),
            "target_model": args.target_model_path,
        }
        records.append(record)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    with eval_path.open("w", encoding="utf-8") as handle:
        for record in records[: args.eval_samples]:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def parse_selected_layers(value: str) -> tuple[int, int, int]:
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(layers) != 3:
        raise ValueError("--selected-layers must provide exactly 3 comma-separated indices")
    return layers


def dataset_uses_completion(data_path: str | Path) -> bool:
    with Path(data_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            return "completion" in sample or "completion_ids" in sample
    return False


def build_eagle3_training_examples(
    data_path: str | Path,
    tokenizer,
    seq_len: int,
) -> list[Eagle3TrainingExample]:
    examples: list[Eagle3TrainingExample] = []
    with Path(data_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if "prompt_ids" in sample:
                prompt_ids = [int(token_id) for token_id in sample["prompt_ids"]]
            else:
                rendered = render_prompt(tokenizer, sample)
                prompt_ids = tokenizer.encode(rendered.prompt, add_special_tokens=False)
            if "completion_ids" in sample:
                completion_ids = [int(token_id) for token_id in sample["completion_ids"]]
            elif "completion" in sample:
                completion_ids = tokenizer.encode(str(sample["completion"]), add_special_tokens=False)
            else:
                raise ValueError(f"{data_path}:{line_number} is missing completion_ids/completion")
            if len(prompt_ids) < 1 or len(completion_ids) < 2:
                continue

            token_ids = prompt_ids + completion_ids
            prompt_tokens = len(prompt_ids)
            completion_tokens = len(completion_ids)
            loss_mask = [0.0] * len(token_ids)
            first_target_position = prompt_tokens
            last_known_target_position = len(token_ids) - 2
            for pos in range(first_target_position, last_known_target_position + 1):
                loss_mask[pos] = 1.0

            if len(token_ids) > seq_len:
                token_ids = token_ids[-seq_len:]
                loss_mask = loss_mask[-seq_len:]
                prompt_tokens = max(0, prompt_tokens - (len(prompt_ids) + completion_tokens - seq_len))
            if len(token_ids) < 3 or sum(loss_mask) <= 0:
                continue
            examples.append(
                Eagle3TrainingExample(
                    prompt_id=str(sample.get("prompt_id", f"row_{line_number:06d}")),
                    token_ids=torch.tensor(token_ids, dtype=torch.long),
                    loss_mask=torch.tensor(loss_mask, dtype=torch.float32),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
    if not examples:
        raise ValueError("dataset did not produce any EAGLE-3 training examples")
    return examples


def make_eagle3_batch(
    examples: Sequence[Eagle3TrainingExample],
    batch_size: int,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    start = (step * batch_size) % len(examples)
    selected = [examples[(start + offset) % len(examples)] for offset in range(batch_size)]
    max_len = max(example.token_ids.numel() for example in selected)
    batch = torch.zeros((batch_size, max_len), dtype=torch.long)
    loss_mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    for row, example in enumerate(selected):
        batch[row, : example.token_ids.numel()] = example.token_ids
        loss_mask[row, : example.loss_mask.numel()] = example.loss_mask
    return batch, loss_mask


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


def _masked_kl_loss(
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if logits.numel() == 0 or float(mask.sum().item()) <= 0.0:
        return logits.new_zeros(())
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    target_probs = torch.softmax(target_logits.float(), dim=-1)
    token_losses = F.kl_div(log_probs, target_probs, reduction="none", log_target=False).sum(dim=-1)
    return (token_losses * mask).sum() / (mask.sum() + 1e-5)


def _masked_ce_loss(
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if logits.numel() == 0 or float(mask.sum().item()) <= 0.0:
        return logits.new_zeros(())
    labels = torch.argmax(target_logits, dim=-1)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    return (losses * mask).sum() / (mask.sum() + 1e-5)


def run_eagle3_sequence_training_step(
    drafter: Eagle3Drafter,
    fused_features: torch.Tensor,
    input_ids: torch.Tensor,
    target_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    ttt_steps: int,
    loss_decay: float,
    loss_type: str,
) -> tuple[torch.Tensor, list[float]]:
    if ttt_steps <= 0:
        raise ValueError("ttt_steps must be positive")
    if input_ids.shape[1] < 3:
        raise ValueError("EAGLE-3 sequence training needs at least 3 tokens")
    hidden_states = drafter.feature_fuser(fused_features)
    token_inputs = torch.cat(
        [input_ids[:, 1:], input_ids.new_zeros((input_ids.shape[0], 1))],
        dim=1,
    )
    total_loss = hidden_states.new_zeros(())
    accuracies: list[float] = []
    loss_fn = _masked_kl_loss if loss_type == "kl" else _masked_ce_loss

    seq_len = input_ids.shape[1]
    for ttt_step in range(ttt_steps):
        logits, pre_norm_hidden_states, _ = drafter.forward_sequence(hidden_states, token_inputs)
        if ttt_step < seq_len - 1:
            draft_logits = logits[:, ttt_step : seq_len - 1]
            teacher_logits = target_logits[:, ttt_step + 1 : seq_len]
            shifted_mask = loss_mask[:, ttt_step + 1 : seq_len]
            step_loss = loss_fn(draft_logits, teacher_logits, shifted_mask)
            total_loss = total_loss + (loss_decay**ttt_step) * step_loss
            with torch.no_grad():
                if float(shifted_mask.sum().item()) > 0.0:
                    pred = torch.argmax(draft_logits, dim=-1)
                    target = torch.argmax(teacher_logits, dim=-1)
                    correct = ((pred == target).float() * shifted_mask).sum()
                    accuracies.append(float((correct / (shifted_mask.sum() + 1e-5)).item()))
                else:
                    accuracies.append(0.0)
        token_inputs = torch.argmax(logits.detach(), dim=-1)
        hidden_states = pre_norm_hidden_states.roll(1, dims=1)
    return total_loss, accuracies


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


def train_eagle3_on_examples(
    target_model: Qwen3ForCausalLM,
    drafter: Eagle3Drafter,
    examples: Sequence[Eagle3TrainingExample],
    *,
    selected_layers: Sequence[int],
    ttt_steps: int,
    steps: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    device: str | torch.device,
    loss_decay: float,
    loss_type: str,
    grad_clip: float,
) -> tuple[list[float], list[list[float]]]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if grad_accum <= 0:
        raise ValueError("grad_accum must be positive")
    if loss_type not in {"kl", "ce"}:
        raise ValueError("--loss-type must be either 'kl' or 'ce'")

    target_model = target_model.to(device=device)
    drafter = drafter.to(device=device)
    target_model.eval()
    optimizer = torch.optim.AdamW(drafter.parameters(), lr=lr, betas=(0.9, 0.95))
    losses: list[float] = []
    accs: list[list[float]] = []
    optimizer.zero_grad(set_to_none=True)

    for step in range(steps):
        batch, loss_mask = make_eagle3_batch(examples, batch_size=batch_size, step=step)
        batch = batch.to(device=device)
        loss_mask = loss_mask.to(device=device)
        with torch.no_grad():
            target_outputs = target_model(
                batch,
                output_hidden_states=True,
                hidden_state_indices=selected_layers,
            )
            fused_features = fuse_hidden_states(target_outputs.hidden_states, selected_layers).detach()
            target_logits = target_outputs.logits.detach()

        loss, step_accs = run_eagle3_sequence_training_step(
            drafter=drafter,
            fused_features=fused_features,
            input_ids=batch,
            target_logits=target_logits,
            loss_mask=loss_mask,
            ttt_steps=ttt_steps,
            loss_decay=loss_decay,
            loss_type=loss_type,
        )
        (loss / grad_accum).backward()
        losses.append(float(loss.item()))
        accs.append(step_accs)
        if (step + 1) % grad_accum == 0 or step == steps - 1:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(drafter.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return losses, accs


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
    state_dict = load_file(str(checkpoint_path / "drafter.safetensors"))
    incompatible = drafter.load_state_dict(state_dict, strict=False)
    unexpected = set(incompatible.unexpected_keys) - {"feature_fuser.bias"}
    if unexpected or incompatible.missing_keys:
        raise RuntimeError(
            "invalid EAGLE-3 checkpoint: "
            f"missing={incompatible.missing_keys[:10]} unexpected={sorted(unexpected)[:10]}"
        )
    drafter = drafter.to(device=device)
    return drafter, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare data and train an EAGLE-3 drafter.")
    subparsers = parser.add_subparsers(dest="command")

    prep = subparsers.add_parser("prepare-data", help="Build UltraChat prompts and target completions with vLLM.")
    prep.add_argument("--target-model-path", default=DEFAULT_TARGET_MODEL_PATH)
    prep.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    prep.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    prep.add_argument("--output", default=DEFAULT_DATA_PATH)
    prep.add_argument("--eval-output", default=DEFAULT_EVAL_PATH)
    prep.add_argument("--num-samples", type=int, default=3000)
    prep.add_argument("--eval-samples", type=int, default=100)
    prep.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    prep.add_argument("--completion-tokens", type=int, default=DEFAULT_COMPLETION_TOKENS)
    prep.add_argument("--dtype", default="bf16")
    prep.add_argument("--tensor-parallel-size", type=int, default=1)
    prep.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    prep.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_PROMPT_TOKENS + DEFAULT_COMPLETION_TOKENS + 32)
    prep.add_argument("--seed", type=int, default=0)

    train = subparsers.add_parser("train", help="Train the local EAGLE-3 drafter.")
    train.add_argument("--target-model-path", default=DEFAULT_TARGET_MODEL_PATH)
    train.add_argument("--data", default=DEFAULT_DATA_PATH)
    train.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    train.add_argument("--seq-len", type=int, default=DEFAULT_MAX_PROMPT_TOKENS + DEFAULT_COMPLETION_TOKENS)
    train.add_argument("--steps", type=int, default=3000)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--grad-accum", type=int, default=1)
    train.add_argument("--lr", type=float, default=5e-5)
    train.add_argument("--draft-len", type=int, default=3)
    train.add_argument("--ttt-steps", type=int, default=3)
    train.add_argument("--ttt-len", type=int, default=None, help="Backward-compatible alias for --ttt-steps.")
    train.add_argument("--num-draft-layers", type=int, default=1)
    train.add_argument("--selected-layers", default="")
    train.add_argument("--loss-decay", type=float, default=0.9)
    train.add_argument("--loss-type", choices=("kl", "ce"), default="kl")
    train.add_argument("--grad-clip", type=float, default=0.5)
    train.add_argument("--dtype", default="bf16")
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--teacher-forcing-only", action="store_true", help=argparse.SUPPRESS)
    train.add_argument("--training-time-test", action="store_true", help=argparse.SUPPRESS)

    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        argv = ["train", *argv]
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "prepare-data":
        prepare_ultrachat_data(args)
        return

    if args.ttt_len is not None:
        args.ttt_steps = args.ttt_len
    torch.manual_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.target_model_path)
    if dataset_uses_completion(args.data):
        examples = build_eagle3_training_examples(args.data, tokenizer=tokenizer, seq_len=args.seq_len)
    else:
        sequences = build_training_sequences(args.data, tokenizer=tokenizer, seq_len=args.seq_len)
        examples = [
            Eagle3TrainingExample(
                prompt_id=f"sequence_{index:06d}",
                token_ids=sequence,
                loss_mask=torch.cat(
                    [torch.zeros(1), torch.ones(max(0, sequence.numel() - 2)), torch.zeros(1)]
                ).float(),
                prompt_tokens=0,
                completion_tokens=int(sequence.numel()),
            )
            for index, sequence in enumerate(sequences)
            if sequence.numel() >= 3
        ]
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
        ttt_steps=args.ttt_steps,
        loss_decay=args.loss_decay,
        norm_before_residual=False,
    )
    drafter = Eagle3Drafter(config)
    if args.device != "cpu":
        drafter = drafter.to(dtype=dtype)
    initialize_drafter_from_target(
        drafter,
        target_model,
        num_fused_layers=len(selected_layers),
    )
    losses, train_accs = train_eagle3_on_examples(
        target_model=target_model,
        drafter=drafter,
        examples=examples,
        selected_layers=selected_layers,
        ttt_steps=args.ttt_steps,
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        device=args.device,
        loss_decay=args.loss_decay,
        loss_type=args.loss_type,
        grad_clip=args.grad_clip,
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
                "mode": "sequence_ttt",
                "steps": args.steps,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "draft_len": args.draft_len,
                "ttt_steps": config.ttt_steps,
                "num_draft_layers": config.num_hidden_layers,
                "selected_layers": list(config.selected_layers),
                "loss_decay": args.loss_decay,
                "loss_type": args.loss_type,
                "grad_clip": args.grad_clip,
                "examples": len(examples),
                "final_train_acc_by_ttt_step": train_accs[-1] if train_accs else [],
                "dtype": args.dtype,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

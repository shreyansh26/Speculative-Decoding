from __future__ import annotations

"""
Medusa-1 non-vLLM inference with tree verification.

The target model emits the root token from its own logits.  Medusa heads propose
future-token top-k choices, those choices are verified in one masked tree
forward through the local Qwen implementation, and the KV cache is compacted to
the accepted root-to-leaf path.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from common.metrics import SpecDecodeStats, summarize_jsonl, write_jsonl_record
from common.qwen3 import LayerCache, Qwen3ForCausalLM, apply_rotary_embeddings
from common.sampling import autoregressive_generate, sample_from_logits
from common.tokenizer import load_tokenizer, render_prompt
from common.verification import (
    PrefixState,
    advance_prefix_state,
    greedy_verify_with_state,
    prefill_prefix,
    state_after_decoded_tokens,
)
from methods.draft_model.training.train import parse_dtype
from methods.medusa_1.training.train import MedusaHeads, load_medusa_checkpoint


DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CHECKPOINT_PATH = "checkpoints/medusa_1_qwen25_7b_eval100"
DEFAULT_PROMPTS_PATH = "data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_DRAFT_LEN = 4
DEFAULT_TREE_TOPK = 5
DEFAULT_MAX_TREE_NODES = 31
DEFAULT_WARMUP_PROMPTS = 2

MEDUSA_REFERENCE_CHOICES: tuple[tuple[int, ...], ...] = (
    (0,), (0, 0), (1,), (0, 1), (0, 0, 0), (1, 0), (2,), (0, 2),
    (0, 0, 1), (0, 3), (3,), (0, 1, 0), (2, 0), (4,), (0, 0, 2),
    (0, 4), (1, 1), (1, 0, 0), (0, 0, 0, 0), (5,), (0, 0, 3),
    (0, 5), (0, 2, 0), (3, 0), (0, 1, 1), (0, 6), (6,), (0, 7),
    (0, 0, 4), (4, 0), (1, 2), (0, 8), (7,), (0, 3, 0),
    (0, 0, 0, 1), (0, 0, 5), (2, 1), (0, 0, 6), (1, 0, 1),
    (0, 0, 1, 0), (2, 0, 0), (5, 0), (0, 9), (0, 1, 2), (8,),
    (0, 4, 0), (0, 2, 1), (1, 3), (0, 0, 7), (0, 0, 0, 2),
    (0, 0, 8), (1, 1, 0), (0, 1, 0, 0), (6, 0), (9,), (0, 1, 3),
    (0, 0, 0, 3), (1, 0, 2), (0, 5, 0), (3, 1), (0, 0, 2, 0),
    (7, 0), (1, 4),
)


@dataclass(slots=True)
class PromptRecord:
    prompt_id: str
    prompt_ids: list[int]
    prompt_text: str


@dataclass(slots=True)
class MedusaBuffers:
    choices: tuple[tuple[int, ...], ...]
    attention_mask: torch.Tensor
    tree_indices: torch.Tensor
    position_offsets: torch.Tensor
    retrieve_indices: torch.Tensor
    retrieve_mask: torch.Tensor
    topk: int
    max_depth: int


@dataclass(slots=True)
class MedusaTreeOutput:
    logits: torch.Tensor
    hidden_states: dict[int, torch.Tensor]
    cache: list[LayerCache | None]


@dataclass(slots=True)
class MedusaTimings:
    prefill_wall_time_s: float = 0.0
    decode_wall_time_s: float = 0.0

    @property
    def total_wall_time_s(self) -> float:
        return self.prefill_wall_time_s + self.decode_wall_time_s


def load_prompt_records(path: str | Path, tokenizer) -> list[PromptRecord]:
    prompt_path = Path(path)
    records: list[PromptRecord] = []
    with prompt_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            prompt_id = str(sample.get("prompt_id", sample.get("id", f"prompt_{line_number:04d}")))
            if "prompt_ids" in sample:
                prompt_ids = [int(token_id) for token_id in sample["prompt_ids"]]
                prompt_text = str(sample.get("prompt", tokenizer.decode(prompt_ids, skip_special_tokens=False)))
            else:
                rendered = render_prompt(tokenizer, sample)
                prompt_ids = tokenizer.encode(rendered.prompt, add_special_tokens=False)
                prompt_text = rendered.prompt
            if prompt_ids:
                records.append(PromptRecord(prompt_id, prompt_ids, prompt_text))
    if not records:
        raise ValueError(f"{prompt_path} did not contain any prompts")
    return records


def select_medusa_choices(
    *,
    max_depth: int,
    topk: int,
    max_tree_nodes: int,
) -> tuple[tuple[int, ...], ...]:
    if max_depth <= 0 or max_tree_nodes <= 1:
        return ()
    filtered: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for choice in MEDUSA_REFERENCE_CHOICES:
        if len(choice) > max_depth or any(index >= topk for index in choice):
            continue
        parents_present = all(choice[:prefix_len] in seen for prefix_len in range(1, len(choice)))
        if not parents_present:
            continue
        filtered.append(choice)
        seen.add(choice)
        if len(filtered) >= max_tree_nodes - 1:
            break
    return tuple(filtered)


def generate_medusa_buffers(
    *,
    max_depth: int,
    topk: int,
    max_tree_nodes: int,
    device: torch.device | str,
) -> MedusaBuffers:
    if topk <= 0:
        raise ValueError("tree top-k must be positive")
    choices = select_medusa_choices(max_depth=max_depth, topk=topk, max_tree_nodes=max_tree_nodes)
    sorted_choices = tuple(sorted(choices, key=lambda path: (len(path), path)))
    tree_len = len(sorted_choices) + 1
    attention_mask = torch.eye(tree_len, tree_len, dtype=torch.bool)
    attention_mask[:, 0] = True

    choice_to_node = {choice: index + 1 for index, choice in enumerate(sorted_choices)}
    for node_index, choice in enumerate(sorted_choices, start=1):
        for prefix_len in range(1, len(choice)):
            attention_mask[node_index, choice_to_node[choice[:prefix_len]]] = True

    tree_indices = torch.zeros(tree_len, dtype=torch.long)
    position_offsets = torch.zeros(tree_len, dtype=torch.long)
    for node_index, choice in enumerate(sorted_choices, start=1):
        depth = len(choice)
        tree_indices[node_index] = choice[-1] + (topk * (depth - 1)) + 1
        position_offsets[node_index] = depth

    if sorted_choices:
        candidate_paths = sorted(sorted_choices, key=lambda path: (-len(path), path))
        max_len = max(len(path) for path in candidate_paths) + 1
        retrieve_indices = torch.full((len(candidate_paths), max_len), -1, dtype=torch.long)
        retrieve_mask = torch.zeros((len(candidate_paths), max_len), dtype=torch.bool)
        for row, path in enumerate(candidate_paths):
            node_indices = [0] + [choice_to_node[path[:prefix_len]] for prefix_len in range(1, len(path) + 1)]
            retrieve_indices[row, : len(node_indices)] = torch.tensor(node_indices, dtype=torch.long)
            retrieve_mask[row, : len(node_indices)] = True
    else:
        retrieve_indices = torch.zeros((1, 1), dtype=torch.long)
        retrieve_mask = torch.ones((1, 1), dtype=torch.bool)

    return MedusaBuffers(
        choices=sorted_choices,
        attention_mask=attention_mask.to(device=device),
        tree_indices=tree_indices.to(device=device),
        position_offsets=position_offsets.to(device=device),
        retrieve_indices=retrieve_indices.to(device=device),
        retrieve_mask=retrieve_mask.to(device=device),
        topk=topk,
        max_depth=max_depth,
    )


def _cache_length(cache: Sequence[LayerCache | None]) -> int:
    first_cache = next((item for item in cache if item is not None), None)
    return 0 if first_cache is None else int(first_cache.key.shape[2])


def _tree_attention_mask(
    cache: Sequence[LayerCache | None],
    node_attention_mask: torch.Tensor,
) -> torch.Tensor:
    past_len = _cache_length(cache)
    query_len = node_attention_mask.shape[0]
    if past_len:
        prefix = torch.ones(
            (query_len, past_len),
            dtype=torch.bool,
            device=node_attention_mask.device,
        )
        full = torch.cat([prefix, node_attention_mask], dim=-1)
    else:
        full = node_attention_mask
    return full.unsqueeze(0).unsqueeze(0)


def _reshape_projection(tensor: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    batch, seq_len, _ = tensor.shape
    return tensor.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)


def _qwen_tree_attention(
    attention_module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    layer_cache: LayerCache | None,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, LayerCache]:
    batch_size, query_len, _ = hidden_states.shape
    query = _reshape_projection(
        attention_module.q_proj(hidden_states),
        attention_module.num_attention_heads,
        attention_module.head_dim,
    )
    key = _reshape_projection(
        attention_module.k_proj(hidden_states),
        attention_module.num_key_value_heads,
        attention_module.head_dim,
    )
    value = _reshape_projection(
        attention_module.v_proj(hidden_states),
        attention_module.num_key_value_heads,
        attention_module.head_dim,
    )
    cos, sin = attention_module.rotary_emb(position_ids)
    query, key = apply_rotary_embeddings(query, key, cos, sin)

    if layer_cache is not None:
        key = torch.cat([layer_cache.key, key], dim=2)
        value = torch.cat([layer_cache.value, value], dim=2)
    updated_cache = LayerCache(key=key, value=value)

    use_manual_gqa = attention_module.num_queries_per_kv > 1
    attn_key = key
    attn_value = value
    if use_manual_gqa:
        attn_key = key.repeat_interleave(attention_module.num_queries_per_kv, dim=1)
        attn_value = value.repeat_interleave(attention_module.num_queries_per_kv, dim=1)

    attn_output = F.scaled_dot_product_attention(
        query,
        attn_key,
        attn_value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    attn_output = attn_output.transpose(1, 2).contiguous().view(
        batch_size,
        query_len,
        attention_module.num_attention_heads * attention_module.head_dim,
    )
    return attention_module.o_proj(attn_output), updated_cache


def qwen_tree_forward(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    cache: list[LayerCache | None],
    buffers: MedusaBuffers,
    hidden_layer: int,
) -> MedusaTreeOutput:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Medusa tree decoding currently supports batch size 1")
    if len(cache) != len(model.model.layers):
        raise ValueError("cache length does not match target model layers")

    past_len = _cache_length(cache)
    hidden_states = model.model.embed_tokens(input_ids)
    position_ids = (buffers.position_offsets[: input_ids.shape[1]] + past_len).unsqueeze(0)
    attention_mask = _tree_attention_mask(cache, buffers.attention_mask[: input_ids.shape[1], : input_ids.shape[1]])

    hidden_state_map: dict[int, torch.Tensor] = {}
    next_cache: list[LayerCache | None] = []
    for layer_idx, layer in enumerate(model.model.layers):
        residual = hidden_states
        attn_output, updated_cache = _qwen_tree_attention(
            layer.self_attn,
            layer.input_layernorm(hidden_states),
            position_ids=position_ids,
            layer_cache=cache[layer_idx],
            attention_mask=attention_mask,
        )
        hidden_states = residual + attn_output
        hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))
        next_cache.append(updated_cache)
        if hidden_layer == layer_idx:
            hidden_state_map[layer_idx] = hidden_states.detach()

    hidden_states = model.model.norm(hidden_states)
    if hidden_layer == model.config.num_hidden_layers:
        hidden_state_map[hidden_layer] = hidden_states.detach()
    logits = model.lm_head(hidden_states)
    return MedusaTreeOutput(logits=logits, hidden_states=hidden_state_map, cache=next_cache)


def compact_tree_cache(
    output_cache: Sequence[LayerCache | None],
    selected_tree_indices: torch.Tensor,
    prefix_len: int,
) -> list[LayerCache | None]:
    keep_indices = torch.cat(
        [
            torch.arange(prefix_len, device=selected_tree_indices.device, dtype=torch.long),
            selected_tree_indices.to(dtype=torch.long) + prefix_len,
        ]
    )
    compacted: list[LayerCache | None] = []
    for layer_cache in output_cache:
        if layer_cache is None:
            compacted.append(None)
            continue
        compacted.append(
            LayerCache(
                key=layer_cache.key.index_select(2, keep_indices).detach(),
                value=layer_cache.value.index_select(2, keep_indices).detach(),
            )
        )
    return compacted


def generate_candidates(
    medusa_heads: MedusaHeads,
    state: PrefixState,
    hidden_layer: int,
    buffers: MedusaBuffers,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if state.hidden_states is None or hidden_layer not in state.hidden_states:
        raise ValueError("target hidden state is required for Medusa proposals")
    root_token = sample_from_logits(
        state.last_logits,
        temperature=temperature,
        top_p=top_p,
        generator=generator,
    )
    hidden = state.hidden_states[hidden_layer]
    medusa_logits = medusa_heads(hidden)
    head_logits = medusa_logits[:, 0, -1, :]
    topk = min(buffers.topk, head_logits.shape[-1])
    top_tokens = torch.topk(head_logits, k=topk, dim=-1).indices
    if topk < buffers.topk:
        pad = top_tokens[:, -1:].expand(-1, buffers.topk - topk)
        top_tokens = torch.cat([top_tokens, pad], dim=-1)

    flat_candidates = torch.cat(
        [
            torch.tensor([root_token], dtype=torch.long, device=head_logits.device),
            top_tokens.reshape(-1),
        ],
        dim=0,
    )
    tree_candidates = flat_candidates[buffers.tree_indices]
    safe_retrieve = buffers.retrieve_indices.clamp_min(0)
    cart_candidates = tree_candidates[safe_retrieve]
    cart_candidates = cart_candidates.masked_fill(~buffers.retrieve_mask, -1)
    return cart_candidates, tree_candidates.unsqueeze(0), medusa_logits


def evaluate_medusa_posterior(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    allow_divergence: bool,
    posterior_temperature: float,
    posterior_threshold: float,
    posterior_alpha: float,
) -> tuple[int, int]:
    if candidates.shape[1] <= 1:
        return 0, 0

    candidate_tokens = candidates[:, 1:]
    valid_tokens = valid_mask[:, 1:]
    verifier_logits = logits[:, :-1, :]
    if verifier_logits.shape[1] == 0:
        return 0, 0

    if allow_divergence:
        temperature = posterior_temperature if posterior_temperature > 0.0 else 1.0
        probs = torch.softmax(verifier_logits / temperature, dim=-1)
        gather_ids = candidate_tokens.clamp_min(0).unsqueeze(-1)
        candidate_probs = torch.gather(probs, dim=-1, index=gather_ids).squeeze(-1)
        entropy = -(probs * torch.log(probs + 1e-5)).sum(dim=-1)
        threshold = torch.minimum(
            torch.full_like(entropy, posterior_threshold),
            torch.exp(-entropy) * posterior_alpha,
        )
        posterior_mask = (candidate_probs > threshold) & valid_tokens
        score_probs = candidate_probs.clamp_min(1e-12)
    else:
        predictions = torch.argmax(verifier_logits, dim=-1)
        posterior_mask = (candidate_tokens == predictions) & valid_tokens
        score_probs = torch.ones_like(verifier_logits[..., 0], dtype=torch.float32)

    accept_lengths = torch.cumprod(posterior_mask.to(torch.long), dim=1).sum(dim=1)
    max_accept = int(accept_lengths.max().item())
    if max_accept <= 0:
        return 0, 0

    best_rows = torch.where(accept_lengths == max_accept)[0]
    if best_rows.numel() == 1 or not allow_divergence:
        return int(best_rows[0].item()), max_accept
    likelihood = torch.log(score_probs[best_rows, :max_accept]).sum(dim=-1)
    best = best_rows[torch.argmax(likelihood)]
    return int(best.item()), max_accept


def tree_decode_step(
    target_model: Qwen3ForCausalLM,
    medusa_heads: MedusaHeads,
    state: PrefixState,
    buffers: MedusaBuffers,
    *,
    hidden_layer: int,
    remaining_tokens: int,
    temperature: float,
    top_p: float,
    allow_divergence: bool,
    posterior_threshold: float,
    posterior_alpha: float,
    eos_token_id: int | None,
    generator: torch.Generator | None,
) -> tuple[list[int], PrefixState, int, int]:
    candidates, tree_candidates, _ = generate_candidates(
        medusa_heads,
        state,
        hidden_layer,
        buffers,
        temperature=temperature,
        top_p=top_p,
        generator=generator,
    )
    tree_output = qwen_tree_forward(
        target_model,
        tree_candidates,
        state.cache,
        buffers,
        hidden_layer,
    )
    safe_retrieve = buffers.retrieve_indices.clamp_min(0)
    path_logits = tree_output.logits[0, safe_retrieve, :]
    best_row, accepted_medusa = evaluate_medusa_posterior(
        path_logits,
        candidates,
        buffers.retrieve_mask,
        allow_divergence=allow_divergence,
        posterior_temperature=temperature,
        posterior_threshold=posterior_threshold,
        posterior_alpha=posterior_alpha,
    )
    valid_len = int(buffers.retrieve_mask[best_row].sum().item())
    emitted_len = min(accepted_medusa + 1, valid_len, remaining_tokens)
    emitted = [int(token) for token in candidates[best_row, :emitted_len].tolist()]
    selected_tree_indices = buffers.retrieve_indices[best_row, :emitted_len]

    stop_after_update = False
    if eos_token_id is not None and eos_token_id in emitted:
        eos_index = emitted.index(eos_token_id)
        emitted = emitted[: eos_index + 1]
        selected_tree_indices = selected_tree_indices[: eos_index + 1]
        accepted_medusa = max(0, len(emitted) - 1)
        stop_after_update = True

    prefix_len = _cache_length(state.cache)
    next_cache = compact_tree_cache(tree_output.cache, selected_tree_indices, prefix_len)
    last_tree_index = int(selected_tree_indices[-1].item())
    next_hidden = {
        hidden_layer: tree_output.hidden_states[hidden_layer][:, last_tree_index : last_tree_index + 1, :].detach()
    }
    next_state = PrefixState(
        prefix_ids=state.prefix_ids + emitted,
        cache=next_cache,
        last_logits=tree_output.logits[0, last_tree_index].detach(),
        hidden_states=next_hidden,
    )
    proposed_medusa = max(0, min(buffers.max_depth, remaining_tokens - 1))
    if stop_after_update:
        proposed_medusa = max(proposed_medusa, accepted_medusa)
    return emitted, next_state, proposed_medusa, accepted_medusa


def propose_medusa_chain_after_root(
    medusa_heads: MedusaHeads,
    target_state: PrefixState,
    target_hidden_layer: int,
    draft_len: int,
) -> list[int]:
    if draft_len <= 0:
        return []
    if target_state.hidden_states is None or target_hidden_layer not in target_state.hidden_states:
        raise ValueError("target hidden state is required for Medusa proposals")
    hidden_states = target_state.hidden_states[target_hidden_layer]
    medusa_logits = medusa_heads(hidden_states)
    chain: list[int] = []
    for head_idx in range(min(draft_len, medusa_logits.shape[0])):
        token = int(torch.argmax(medusa_logits[head_idx, 0, -1], dim=-1).item())
        chain.append(token)
    return chain


def verify_seeded_chain_with_state(
    model: torch.nn.Module,
    state: PrefixState,
    seed_token: int,
    draft_ids: Sequence[int],
    hidden_layer: int,
) -> tuple[list[int], PrefixState, int]:
    if hasattr(model, "decode_many") and state.cache:
        candidate_ids = [int(seed_token)] + [int(token) for token in draft_ids]
        candidate_tensor = torch.tensor([candidate_ids], dtype=torch.long, device=getattr(model, "device", torch.device("cpu")))
        output = model.decode_many(
            candidate_tensor,
            cache=state.cache,
            output_hidden_states=True,
            hidden_state_indices=[hidden_layer],
        )
        predictions = [int(seed_token)] + [int(token) for token in torch.argmax(output.logits[0], dim=-1).tolist()]
        accepted = 0
        for index, draft_token in enumerate(draft_ids):
            if predictions[index + 1] != int(draft_token):
                break
            accepted += 1
        emitted = candidate_ids[: 1 + accepted]
        next_state = state_after_decoded_tokens(state, output, emitted)
        return emitted, next_state, accepted

    seed_state = advance_prefix_state(model, state, int(seed_token), hidden_state_indices=[hidden_layer])
    result, next_state = greedy_verify_with_state(
        model,
        seed_state,
        draft_ids,
        hidden_state_indices=[hidden_layer],
    )
    return [int(seed_token)] + result.emitted_ids, next_state, result.accepted_draft_tokens


def _supports_qwen_tree_path(model: torch.nn.Module, state: PrefixState) -> bool:
    return (
        bool(state.cache)
        and hasattr(model, "model")
        and hasattr(model.model, "layers")
        and hasattr(model.model, "embed_tokens")
    )


def run_medusa_speculative_decode(
    target_model: torch.nn.Module,
    medusa_heads: MedusaHeads,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    draft_len: int,
    eos_token_id: int | None = None,
    *,
    tree_topk: int = DEFAULT_TREE_TOPK,
    max_tree_nodes: int = DEFAULT_MAX_TREE_NODES,
    temperature: float = 0.0,
    top_p: float = 1.0,
    allow_divergence: bool = False,
    posterior_threshold: float = 0.09,
    posterior_alpha: float = 0.3,
    generator: torch.Generator | None = None,
) -> tuple[list[int], dict[str, int]]:
    with torch.inference_mode():
        generated: list[int] = []
        hidden_layer = target_model.config.num_hidden_layers
        state = prefill_prefix(target_model, prompt_ids, hidden_state_indices=[hidden_layer])
        counters = {
            "speculation_steps": 0,
            "target_forwards": 0,
            "draft_forwards": 0,
            "proposed_draft_tokens": 0,
            "accepted_draft_tokens": 0,
            "tree_nodes": 0,
        }

        use_tree = isinstance(target_model, Qwen3ForCausalLM) and _supports_qwen_tree_path(target_model, state)
        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            requested = max(0, min(draft_len, medusa_heads.num_heads, remaining - 1))

            if use_tree:
                buffers = generate_medusa_buffers(
                    max_depth=requested,
                    topk=tree_topk,
                    max_tree_nodes=max_tree_nodes,
                    device=getattr(target_model, "device", torch.device("cpu")),
                )
                emitted, state, proposed, accepted = tree_decode_step(
                    target_model,
                    medusa_heads,
                    state,
                    buffers,
                    hidden_layer=hidden_layer,
                    remaining_tokens=remaining,
                    temperature=temperature,
                    top_p=top_p,
                    allow_divergence=allow_divergence,
                    posterior_threshold=posterior_threshold,
                    posterior_alpha=posterior_alpha,
                    eos_token_id=eos_token_id,
                    generator=generator,
                )
                counters["target_forwards"] += 1
                counters["tree_nodes"] += buffers.attention_mask.shape[0]
            else:
                seed_token = sample_from_logits(state.last_logits, temperature, top_p, generator)
                draft_ids = propose_medusa_chain_after_root(medusa_heads, state, hidden_layer, requested)
                emitted, state, accepted = verify_seeded_chain_with_state(
                    target_model,
                    state,
                    seed_token,
                    draft_ids,
                    hidden_layer,
                )
                emitted = emitted[:remaining]
                proposed = len(draft_ids)
                counters["target_forwards"] += 1
                counters["tree_nodes"] += 0

            counters["speculation_steps"] += 1
            counters["draft_forwards"] += 1
            counters["proposed_draft_tokens"] += proposed
            counters["accepted_draft_tokens"] += accepted

            for token in emitted:
                if len(generated) >= max_new_tokens:
                    break
                generated.append(int(token))
                if eos_token_id is not None and int(token) == eos_token_id:
                    return generated, counters

    return generated, counters


def timed_cached_greedy_generate(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    *,
    eos_token_id: int | None,
    sync_device,
) -> tuple[list[int], MedusaTimings]:
    timings = MedusaTimings()
    sync_device()
    start = time.perf_counter()
    tokens = autoregressive_generate(
        model,
        prompt_ids,
        max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=eos_token_id,
    )
    sync_device()
    timings.decode_wall_time_s = time.perf_counter() - start
    return tokens, timings


def timed_medusa_decode(
    target_model: torch.nn.Module,
    medusa_heads: MedusaHeads,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    *,
    draft_len: int,
    tree_topk: int,
    max_tree_nodes: int,
    temperature: float,
    top_p: float,
    allow_divergence: bool,
    posterior_threshold: float,
    posterior_alpha: float,
    eos_token_id: int | None,
    generator: torch.Generator | None,
    sync_device,
) -> tuple[list[int], dict[str, int], MedusaTimings]:
    timings = MedusaTimings()
    sync_device()
    start = time.perf_counter()
    tokens, counters = run_medusa_speculative_decode(
        target_model=target_model,
        medusa_heads=medusa_heads,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        draft_len=draft_len,
        eos_token_id=eos_token_id,
        tree_topk=tree_topk,
        max_tree_nodes=max_tree_nodes,
        temperature=temperature,
        top_p=top_p,
        allow_divergence=allow_divergence,
        posterior_threshold=posterior_threshold,
        posterior_alpha=posterior_alpha,
        generator=generator,
    )
    sync_device()
    timings.decode_wall_time_s = time.perf_counter() - start
    return tokens, counters, timings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Medusa-1 speculative decoding inference.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--output", default="")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--draft-len", type=int, default=DEFAULT_DRAFT_LEN)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--posterior-threshold", type=float, default=0.09)
    parser.add_argument("--posterior-alpha", type=float, default=0.3)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-target", action="store_true")
    parser.add_argument("--compile-heads", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--warmup-prompts", type=int, default=DEFAULT_WARMUP_PROMPTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--allow-divergence", action="store_true")
    parser.add_argument("--require-baseline-match", action="store_true")
    parser.add_argument("--tree-topk", type=int, default=DEFAULT_TREE_TOPK)
    parser.add_argument("--max-tree-nodes", type=int, default=DEFAULT_MAX_TREE_NODES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.temperature != 0.0 and not args.allow_divergence:
        raise ValueError("non-divergent Medusa inference supports greedy decoding only")
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=args.device if torch.device(args.device).type == "cuda" else "cpu")
    generator.manual_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    prompts = load_prompt_records(args.prompts, tokenizer=tokenizer)
    target_model = Qwen3ForCausalLM.from_pretrained(args.model_path, device=args.device, dtype=dtype)
    medusa_heads, metadata = load_medusa_checkpoint(args.checkpoint_path)
    medusa_heads = medusa_heads.to(device=args.device)
    if torch.device(args.device).type != "cpu":
        medusa_heads = medusa_heads.to(dtype=dtype)
    medusa_heads.eval()
    target_model.eval()

    compile_target = args.compile or args.compile_target
    compile_heads = args.compile or args.compile_heads
    compile_enabled = compile_target or compile_heads
    if compile_target:
        target_model = torch.compile(target_model, mode="reduce-overhead")
    if compile_heads:
        medusa_heads = torch.compile(medusa_heads, mode="reduce-overhead")

    output_path = Path(args.output) if args.output else Path(f"runs/medusa_1_len{args.draft_len}.jsonl")
    if output_path.exists():
        output_path.unlink()

    def synchronize() -> None:
        if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(torch.device(args.device))

    warmup_count = min(max(args.warmup_prompts, 0), len(prompts))
    for record in prompts[:warmup_count]:
        if not args.skip_baseline:
            autoregressive_generate(
                target_model,
                record.prompt_ids,
                args.max_new_tokens,
                0.0,
                1.0,
                tokenizer.eos_token_id,
            )
        run_medusa_speculative_decode(
            target_model=target_model,
            medusa_heads=medusa_heads,
            prompt_ids=record.prompt_ids,
            max_new_tokens=args.max_new_tokens,
            draft_len=args.draft_len,
            eos_token_id=tokenizer.eos_token_id,
            tree_topk=args.tree_topk,
            max_tree_nodes=args.max_tree_nodes,
            temperature=args.temperature,
            top_p=args.top_p,
            allow_divergence=args.allow_divergence,
            posterior_threshold=args.posterior_threshold,
            posterior_alpha=args.posterior_alpha,
            generator=generator,
        )
        synchronize()

    divergence_count = 0
    first_diverged_prompt_id = ""
    token_count_mismatches = 0
    for record in prompts:
        baseline_tokens: list[int] = []
        baseline_timings = MedusaTimings()
        if not args.skip_baseline:
            baseline_tokens, baseline_timings = timed_cached_greedy_generate(
                target_model,
                record.prompt_ids,
                args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                sync_device=synchronize,
            )

        generated_tokens, counters, method_timings = timed_medusa_decode(
            target_model=target_model,
            medusa_heads=medusa_heads,
            prompt_ids=record.prompt_ids,
            max_new_tokens=args.max_new_tokens,
            draft_len=args.draft_len,
            tree_topk=args.tree_topk,
            max_tree_nodes=args.max_tree_nodes,
            temperature=args.temperature,
            top_p=args.top_p,
            allow_divergence=args.allow_divergence,
            posterior_threshold=args.posterior_threshold,
            posterior_alpha=args.posterior_alpha,
            eos_token_id=tokenizer.eos_token_id,
            generator=generator,
            sync_device=synchronize,
        )

        matches_baseline = True
        if baseline_tokens and generated_tokens != baseline_tokens:
            matches_baseline = False
            divergence_count += 1
            if len(generated_tokens) != len(baseline_tokens):
                token_count_mismatches += 1
            if not first_diverged_prompt_id:
                first_diverged_prompt_id = record.prompt_id
        if args.require_baseline_match and not matches_baseline:
            raise RuntimeError(f"Medusa greedy output diverged for {record.prompt_id}")

        stats = SpecDecodeStats(
            method="medusa_1",
            model=args.model_path,
            prompt_id=record.prompt_id,
            prompt_tokens=len(record.prompt_ids),
            generated_tokens=len(generated_tokens),
            generated_text=tokenizer.decode(generated_tokens, skip_special_tokens=True),
            temperature=args.temperature,
            draft_len=args.draft_len,
            speculation_steps=counters["speculation_steps"],
            target_forwards=counters["target_forwards"],
            draft_forwards=counters["draft_forwards"],
            proposed_draft_tokens=counters["proposed_draft_tokens"],
            accepted_draft_tokens=counters["accepted_draft_tokens"],
            baseline_wall_time_s=baseline_timings.total_wall_time_s,
            method_wall_time_s=method_timings.total_wall_time_s,
            torch_compile=compile_enabled,
            cuda_graphs=False,
            cuda_graphs_reason=(
                "disabled: variable Medusa tree paths and dynamic accepted lengths"
                if args.cuda_graphs
                else "disabled"
            ),
            seed=args.seed,
        )
        record_out = stats.to_record()
        record_out |= {
            "matches_baseline": matches_baseline,
            "allow_divergence": args.allow_divergence,
            "tree_topk": args.tree_topk,
            "max_tree_nodes": args.max_tree_nodes,
            "mean_tree_nodes_per_step": (
                counters["tree_nodes"] / counters["speculation_steps"]
                if counters["speculation_steps"]
                else 0.0
            ),
            "posterior_threshold": args.posterior_threshold,
            "posterior_alpha": args.posterior_alpha,
            "baseline_prefill_wall_time_s": baseline_timings.prefill_wall_time_s,
            "baseline_decode_wall_time_s": baseline_timings.decode_wall_time_s,
            "method_prefill_wall_time_s": method_timings.prefill_wall_time_s,
            "method_decode_wall_time_s": method_timings.decode_wall_time_s,
            "decode_speedup": (
                baseline_timings.decode_wall_time_s / method_timings.decode_wall_time_s
                if method_timings.decode_wall_time_s
                else 0.0
            ),
            "checkpoint_path": args.checkpoint_path,
            "checkpoint_target_model_path": str(metadata.get("target_model_path", "")),
        }
        write_jsonl_record(output_path, record_out)

    summary = summarize_jsonl(output_path) | {
        "method": "medusa_1",
        "num_prompts": len(prompts),
        "output": str(output_path),
        "matches_baseline": divergence_count == 0,
        "diverged_prompts": divergence_count,
        "first_diverged_prompt_id": first_diverged_prompt_id,
        "token_count_mismatches": token_count_mismatches,
        "draft_len": args.draft_len,
        "tree_topk": args.tree_topk,
        "max_tree_nodes": args.max_tree_nodes,
        "allow_divergence": args.allow_divergence,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

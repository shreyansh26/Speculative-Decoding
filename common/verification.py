from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch


@dataclass(slots=True)
class VerificationResult:
    accepted_ids: list[int]
    emitted_ids: list[int]
    target_predictions: list[int]
    proposed_draft_tokens: int
    accepted_draft_tokens: int
    mismatch_index: int | None
    bonus_token: int | None


@dataclass(slots=True)
class PrefixState:
    prefix_ids: list[int]
    cache: list[object | None]
    last_logits: torch.Tensor
    hidden_states: dict[int, torch.Tensor] | None = None


def _as_batch_tensor(token_ids: Sequence[int], device: torch.device | None = None) -> torch.Tensor:
    if not token_ids:
        raise ValueError("token sequence must not be empty")
    return torch.tensor([list(token_ids)], dtype=torch.long, device=device)


def _last_hidden_state_map(hidden_states: dict[int, torch.Tensor] | None) -> dict[int, torch.Tensor] | None:
    if hidden_states is None:
        return None
    return {layer_idx: tensor[:, -1:, :].detach() for layer_idx, tensor in hidden_states.items()}


def _slice_hidden_state_map(
    hidden_states: dict[int, torch.Tensor] | None,
    token_index: int,
) -> dict[int, torch.Tensor] | None:
    if hidden_states is None:
        return None
    return {layer_idx: tensor[:, token_index : token_index + 1, :].detach() for layer_idx, tensor in hidden_states.items()}


def _truncate_cache(cache: list[object | None], total_length: int) -> list[object | None]:
    truncated: list[object | None] = []
    for layer_cache in cache:
        if layer_cache is None:
            truncated.append(None)
            continue
        truncated.append(
            type(layer_cache)(
                key=layer_cache.key[:, :, :total_length, :].detach(),
                value=layer_cache.value[:, :, :total_length, :].detach(),
            )
        )
    return truncated


def state_after_decoded_tokens(
    state: PrefixState,
    output,
    consumed_ids: Sequence[int],
) -> PrefixState:
    if not consumed_ids:
        return state

    output_cache = getattr(output, "cache", None) or []
    output_hidden_states = getattr(output, "hidden_states", None)
    next_prefix_ids = state.prefix_ids + list(consumed_ids)
    consumed_count = len(consumed_ids)
    if consumed_count == len(output.logits[0]):
        next_cache = output_cache
        next_logits = output.logits[0, -1].detach()
        next_hidden_states = _last_hidden_state_map(output_hidden_states)
    else:
        total_length = len(next_prefix_ids)
        next_cache = _truncate_cache(output_cache, total_length)
        next_logits = output.logits[0, consumed_count - 1].detach()
        next_hidden_states = _slice_hidden_state_map(output_hidden_states, consumed_count - 1)

    return PrefixState(
        prefix_ids=next_prefix_ids,
        cache=next_cache,
        last_logits=next_logits,
        hidden_states=next_hidden_states,
    )


def _greedy_verify_with_state_sequential(
    model: torch.nn.Module,
    state: PrefixState,
    draft_ids: Sequence[int],
    *,
    hidden_state_indices: Iterable[int] | None = None,
) -> tuple[VerificationResult, PrefixState]:
    accepted: list[int] = []
    target_predictions: list[int] = []
    working_state = state

    for index, draft_token in enumerate(draft_ids):
        target_token = int(torch.argmax(working_state.last_logits).item())
        target_predictions.append(target_token)
        if target_token != int(draft_token):
            emitted = accepted + [target_token]
            updated_state = advance_prefix_state(
                model,
                working_state,
                target_token,
                hidden_state_indices=hidden_state_indices,
            )
            return VerificationResult(
                accepted_ids=accepted,
                emitted_ids=emitted,
                target_predictions=target_predictions,
                proposed_draft_tokens=len(draft_ids),
                accepted_draft_tokens=len(accepted),
                mismatch_index=index,
                bonus_token=None,
            ), updated_state

        accepted.append(int(draft_token))
        working_state = advance_prefix_state(
            model,
            working_state,
            int(draft_token),
            hidden_state_indices=hidden_state_indices,
        )

    bonus_token = int(torch.argmax(working_state.last_logits).item())
    target_predictions.append(bonus_token)
    updated_state = advance_prefix_state(
        model,
        working_state,
        bonus_token,
        hidden_state_indices=hidden_state_indices,
    )
    return VerificationResult(
        accepted_ids=accepted,
        emitted_ids=accepted + [bonus_token],
        target_predictions=target_predictions,
        proposed_draft_tokens=len(draft_ids),
        accepted_draft_tokens=len(accepted),
        mismatch_index=None,
        bonus_token=bonus_token,
    ), updated_state


def prefill_prefix(
    model: torch.nn.Module,
    prefix_ids: Sequence[int],
    *,
    hidden_state_indices: Iterable[int] | None = None,
) -> PrefixState:
    if not prefix_ids:
        raise ValueError("prefix_ids must not be empty")
    device = getattr(model, "device", torch.device("cpu"))
    wants_hidden_states = hidden_state_indices is not None
    input_ids = _as_batch_tensor(prefix_ids, device=device)
    if hasattr(model, "prefill"):
        if wants_hidden_states:
            output = model.prefill(
                input_ids,
                output_hidden_states=True,
                hidden_state_indices=hidden_state_indices,
            )
        else:
            output = model.prefill(input_ids)
    else:
        if wants_hidden_states:
            output = model(
                input_ids,
                output_hidden_states=True,
                hidden_state_indices=hidden_state_indices,
            )
        else:
            output = model(input_ids)
    return PrefixState(
        prefix_ids=list(prefix_ids),
        cache=getattr(output, "cache", None) or [],
        last_logits=output.logits[0, -1].detach(),
        hidden_states=_last_hidden_state_map(getattr(output, "hidden_states", None)),
    )


def advance_prefix_state(
    model: torch.nn.Module,
    state: PrefixState,
    token_id: int,
    *,
    hidden_state_indices: Iterable[int] | None = None,
) -> PrefixState:
    device = getattr(model, "device", torch.device("cpu"))
    wants_hidden_states = hidden_state_indices is not None
    next_prefix_ids = state.prefix_ids + [int(token_id)]
    if hasattr(model, "decode_one"):
        token_tensor = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
        if wants_hidden_states:
            output = model.decode_one(
                token_tensor,
                cache=state.cache,
                output_hidden_states=True,
                hidden_state_indices=hidden_state_indices,
            )
        else:
            output = model.decode_one(token_tensor, cache=state.cache)
    else:
        input_ids = _as_batch_tensor(next_prefix_ids, device=device)
        if wants_hidden_states:
            output = model(
                input_ids,
                output_hidden_states=True,
                hidden_state_indices=hidden_state_indices,
            )
        else:
            output = model(input_ids)
    return PrefixState(
        prefix_ids=next_prefix_ids,
        cache=getattr(output, "cache", None) or [],
        last_logits=output.logits[0, -1].detach(),
        hidden_states=_last_hidden_state_map(getattr(output, "hidden_states", None)),
    )


def greedy_verify(
    model: torch.nn.Module,
    prefix_ids: Sequence[int],
    draft_ids: Sequence[int],
) -> VerificationResult:
    if not prefix_ids:
        raise ValueError("prefix_ids must not be empty")

    device = getattr(model, "device", torch.device("cpu"))
    full_ids = list(prefix_ids) + list(draft_ids)
    output = model(_as_batch_tensor(full_ids, device=device))
    logits = output.logits[0]
    prefix_start = len(prefix_ids) - 1
    prediction_window = len(draft_ids) + 1
    predictions = torch.argmax(
        logits[prefix_start : prefix_start + prediction_window],
        dim=-1,
    ).tolist()

    accepted: list[int] = []
    mismatch_index: int | None = None
    emitted: list[int] = []

    for index, draft_token in enumerate(draft_ids):
        target_token = int(predictions[index])
        if target_token != int(draft_token):
            mismatch_index = index
            emitted = accepted + [target_token]
            return VerificationResult(
                accepted_ids=accepted,
                emitted_ids=emitted,
                target_predictions=[int(token) for token in predictions],
                proposed_draft_tokens=len(draft_ids),
                accepted_draft_tokens=len(accepted),
                mismatch_index=mismatch_index,
                bonus_token=None,
            )
        accepted.append(int(draft_token))

    bonus_token = int(predictions[-1])
    emitted = accepted + [bonus_token]
    return VerificationResult(
        accepted_ids=accepted,
        emitted_ids=emitted,
        target_predictions=[int(token) for token in predictions],
        proposed_draft_tokens=len(draft_ids),
        accepted_draft_tokens=len(accepted),
        mismatch_index=None,
        bonus_token=bonus_token,
    )


def greedy_verify_with_state(
    model: torch.nn.Module,
    state: PrefixState,
    draft_ids: Sequence[int],
    *,
    hidden_state_indices: Iterable[int] | None = None,
) -> tuple[VerificationResult, PrefixState]:
    if hasattr(model, "decode_many") and state.cache:
        first_prediction = int(torch.argmax(state.last_logits).item())
        if draft_ids and first_prediction != int(draft_ids[0]):
            updated_state = advance_prefix_state(
                model,
                state,
                first_prediction,
                hidden_state_indices=hidden_state_indices,
            )
            return VerificationResult(
                accepted_ids=[],
                emitted_ids=[first_prediction],
                target_predictions=[first_prediction],
                proposed_draft_tokens=len(draft_ids),
                accepted_draft_tokens=0,
                mismatch_index=0,
                bonus_token=None,
            ), updated_state

        wants_hidden_states = hidden_state_indices is not None
        device = getattr(model, "device", torch.device("cpu"))
        if draft_ids:
            draft_tensor = _as_batch_tensor(draft_ids, device=device)
            if wants_hidden_states:
                output = model.decode_many(
                    draft_tensor,
                    cache=state.cache,
                    output_hidden_states=True,
                    hidden_state_indices=hidden_state_indices,
                )
            else:
                output = model.decode_many(draft_tensor, cache=state.cache)
            prediction_tail = torch.argmax(output.logits[0], dim=-1).tolist()
        else:
            output = None
            prediction_tail = []

        predictions = [first_prediction] + [int(token) for token in prediction_tail]
        accepted_count = 0
        for index, draft_token in enumerate(draft_ids):
            if predictions[index] != int(draft_token):
                break
            accepted_count += 1

        accepted_ids = [int(token) for token in draft_ids[:accepted_count]]
        if accepted_count < len(draft_ids):
            accepted_state = (
                state if accepted_count == 0 else state_after_decoded_tokens(state, output, accepted_ids)
            )
            mismatch_token = int(predictions[accepted_count])
            updated_state = advance_prefix_state(
                model,
                accepted_state,
                mismatch_token,
                hidden_state_indices=hidden_state_indices,
            )
            return VerificationResult(
                accepted_ids=accepted_ids,
                emitted_ids=accepted_ids + [mismatch_token],
                target_predictions=predictions[: accepted_count + 1],
                proposed_draft_tokens=len(draft_ids),
                accepted_draft_tokens=accepted_count,
                mismatch_index=accepted_count,
                bonus_token=None,
            ), updated_state

        accepted_state = state if not accepted_ids else state_after_decoded_tokens(state, output, accepted_ids)
        bonus_token = int(predictions[-1])
        updated_state = advance_prefix_state(
            model,
            accepted_state,
            bonus_token,
            hidden_state_indices=hidden_state_indices,
        )
        return VerificationResult(
            accepted_ids=accepted_ids,
            emitted_ids=accepted_ids + [bonus_token],
            target_predictions=predictions,
            proposed_draft_tokens=len(draft_ids),
            accepted_draft_tokens=accepted_count,
            mismatch_index=None,
            bonus_token=bonus_token,
        ), updated_state

    return _greedy_verify_with_state_sequential(
        model,
        state,
        draft_ids,
        hidden_state_indices=hidden_state_indices,
    )


def run_greedy_speculative_decode(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    draft_provider: Callable[[list[int], int], list[int]],
    draft_len: int,
) -> tuple[list[int], dict[str, int]]:
    if draft_len < 0:
        raise ValueError("draft_len must be non-negative")

    prefix = list(prompt_ids)
    generated: list[int] = []
    stats = {
        "speculation_steps": 0,
        "target_forwards": 0,
        "draft_forwards": 0,
        "proposed_draft_tokens": 0,
        "accepted_draft_tokens": 0,
    }

    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        requested = min(draft_len, remaining)
        draft_ids = draft_provider(prefix, requested)
        if len(draft_ids) > requested:
            raise ValueError("draft_provider returned more tokens than requested")

        result = greedy_verify(model, prefix, draft_ids)
        stats["speculation_steps"] += 1
        stats["target_forwards"] += 1
        stats["draft_forwards"] += 1 if draft_ids else 0
        stats["proposed_draft_tokens"] += result.proposed_draft_tokens
        stats["accepted_draft_tokens"] += result.accepted_draft_tokens

        for token in result.emitted_ids:
            if len(generated) >= max_new_tokens:
                break
            prefix.append(token)
            generated.append(token)

    return generated, stats


def probabilistic_verify(
    model: torch.nn.Module,
    prefix_ids: Sequence[int],
    draft_ids: Sequence[int],
    draft_probs: torch.Tensor,
    generator: torch.Generator | None = None,
) -> VerificationResult:
    if draft_probs.ndim != 2 or draft_probs.shape[0] != len(draft_ids):
        raise ValueError("draft_probs must have shape [len(draft_ids), vocab_size]")

    device = getattr(model, "device", torch.device("cpu"))
    output = model(_as_batch_tensor(list(prefix_ids) + list(draft_ids), device=device))
    logits = output.logits[0]
    prefix_start = len(prefix_ids) - 1
    target_logits = logits[prefix_start : prefix_start + len(draft_ids)]
    target_probs = torch.softmax(target_logits, dim=-1)
    accepted: list[int] = []

    for index, draft_token in enumerate(draft_ids):
        target_prob = float(target_probs[index, draft_token].item())
        draft_prob = float(draft_probs[index, draft_token].item())
        accept_prob = 1.0 if draft_prob <= 0.0 else min(1.0, target_prob / draft_prob)
        draw = torch.rand(1, generator=generator).item() if generator is not None else float(torch.rand(1).item())
        if draw <= accept_prob:
            accepted.append(int(draft_token))
            continue

        residual = torch.clamp(target_probs[index] - draft_probs[index], min=0.0)
        if float(residual.sum().item()) == 0.0:
            fallback = int(torch.argmax(target_probs[index]).item())
            return VerificationResult(
                accepted_ids=accepted,
                emitted_ids=accepted + [fallback],
                target_predictions=[],
                proposed_draft_tokens=len(draft_ids),
                accepted_draft_tokens=len(accepted),
                mismatch_index=index,
                bonus_token=None,
            )
        residual = residual / residual.sum()
        sampled = int(torch.multinomial(residual, num_samples=1, generator=generator).item())
        return VerificationResult(
            accepted_ids=accepted,
            emitted_ids=accepted + [sampled],
            target_predictions=[],
            proposed_draft_tokens=len(draft_ids),
            accepted_draft_tokens=len(accepted),
            mismatch_index=index,
            bonus_token=None,
        )

    bonus_token = int(torch.argmax(logits[prefix_start + len(draft_ids)], dim=-1).item())
    return VerificationResult(
        accepted_ids=accepted,
        emitted_ids=accepted + [bonus_token],
        target_predictions=[],
        proposed_draft_tokens=len(draft_ids),
        accepted_draft_tokens=len(accepted),
        mismatch_index=None,
        bonus_token=bonus_token,
    )

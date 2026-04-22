from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

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


def _as_batch_tensor(token_ids: Sequence[int], device: torch.device | None = None) -> torch.Tensor:
    if not token_ids:
        raise ValueError("token sequence must not be empty")
    return torch.tensor([list(token_ids)], dtype=torch.long, device=device)


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

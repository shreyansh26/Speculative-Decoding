from __future__ import annotations

from typing import Sequence

import torch


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> int:
    if logits.ndim != 1:
        raise ValueError(f"expected 1D logits, got shape {tuple(logits.shape)}")
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())

    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= top_p
        keep[0] = True
        filtered = torch.zeros_like(probs)
        filtered[sorted_indices[keep]] = probs[sorted_indices[keep]]
        probs = filtered / filtered.sum()

    sample = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(sample.item())


def autoregressive_generate(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")

    device = getattr(model, "device", torch.device("cpu"))
    generated: list[int] = []

    with torch.inference_mode():
        if hasattr(model, "prefill") and hasattr(model, "decode_one"):
            sequence = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
            output = model.prefill(sequence)
            cache = output.cache
            next_logits = output.logits[0, -1]

            for _ in range(max_new_tokens):
                next_token = sample_from_logits(
                    next_logits,
                    temperature=temperature,
                    top_p=top_p,
                    generator=generator,
                )
                generated.append(next_token)
                if eos_token_id is not None and next_token == eos_token_id:
                    break
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
                output = model.decode_one(next_token_tensor, cache=cache)
                cache = output.cache
                next_logits = output.logits[0, -1]

            return generated

        sequence = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            output = model(sequence)
            next_token = sample_from_logits(
                output.logits[0, -1],
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            generated.append(next_token)
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            sequence = torch.cat([sequence, next_token_tensor], dim=1)
            if eos_token_id is not None and next_token == eos_token_id:
                break

    return generated

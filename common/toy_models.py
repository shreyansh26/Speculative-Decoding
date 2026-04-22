from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class ToyModelOutput:
    logits: torch.Tensor


def _build_logits(predictions: torch.Tensor, vocab_size: int) -> torch.Tensor:
    batch, seq_len = predictions.shape
    logits = torch.full((batch, seq_len, vocab_size), -50.0, dtype=torch.float32)
    logits.scatter_(2, predictions.unsqueeze(-1), 50.0)
    return logits


class _ToyBaseLM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")

    def predict_from_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, input_ids: torch.Tensor) -> ToyModelOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"expected [batch, seq] input_ids, got {tuple(input_ids.shape)}")
        predictions = self.predict_from_input(input_ids)
        return ToyModelOutput(logits=_build_logits(predictions, self.vocab_size))


class ToyIncrementLM(_ToyBaseLM):
    def __init__(self, vocab_size: int = 32) -> None:
        super().__init__(vocab_size=vocab_size)

    def predict_from_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids + 1) % self.vocab_size


class ToyBigramLM(_ToyBaseLM):
    def __init__(self, transitions: dict[int, int], vocab_size: int | None = None) -> None:
        inferred_vocab = vocab_size
        if inferred_vocab is None:
            inferred_vocab = max(max(transitions), max(transitions.values())) + 1
        super().__init__(vocab_size=inferred_vocab)
        table = torch.arange(self.vocab_size, dtype=torch.long)
        for source, target in transitions.items():
            table[source] = target
        self.register_buffer("transition_table", table, persistent=False)

    def predict_from_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transition_table[input_ids]


class ToyNoisyDraftLM(_ToyBaseLM):
    def __init__(
        self,
        transitions: dict[int, int],
        vocab_size: int | None = None,
        wrong_offset: int = 2,
        wrong_every: int = 3,
    ) -> None:
        inferred_vocab = vocab_size
        if inferred_vocab is None:
            inferred_vocab = max(max(transitions), max(transitions.values())) + 1
        super().__init__(vocab_size=inferred_vocab)
        table = torch.arange(self.vocab_size, dtype=torch.long)
        for source, target in transitions.items():
            table[source] = target
        self.register_buffer("transition_table", table, persistent=False)
        self.wrong_offset = wrong_offset
        self.wrong_every = wrong_every

    def predict_from_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        predictions = self.transition_table[input_ids].clone()
        if self.wrong_every <= 0:
            return predictions
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        wrong_mask = (positions + 1) % self.wrong_every == 0
        predictions[:, wrong_mask] = (predictions[:, wrong_mask] + self.wrong_offset) % self.vocab_size
        return predictions

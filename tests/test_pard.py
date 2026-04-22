from __future__ import annotations

import torch

from common.sampling import autoregressive_generate
from common.toy_models import ToyBigramLM
from methods.parallel_draft_models.inference.infer import run_pard_speculative_decode
from methods.parallel_draft_models.training.train import build_pard_training_example


class ToyPARDOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class ToyPARDDraft(torch.nn.Module):
    def __init__(self, mask_token_id: int, vocab_size: int = 8) -> None:
        super().__init__()
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")

    def forward(self, input_ids: torch.Tensor) -> ToyPARDOutput:
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], self.vocab_size), -50.0)
        prefix = input_ids[0].tolist()
        first_mask = prefix.index(self.mask_token_id)
        last_real = prefix[first_mask - 1]
        predictions = [((last_real - 1 + idx + 1) % 3) + 1 for idx in range(input_ids.shape[1] - first_mask)]
        for offset, token in enumerate(predictions):
            logits[0, first_mask + offset, token] = 50.0
        return ToyPARDOutput(logits=logits)


def test_mask_token_input_builder_creates_correct_labels() -> None:
    sequence = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    input_ids, labels = build_pard_training_example(sequence, draft_len=2, mask_token_id=99)
    assert input_ids.tolist() == [1, 2, 3, 99, 99]
    assert labels.tolist() == [4, 5]


def test_one_forward_returns_k_logits() -> None:
    draft_model = ToyPARDDraft(mask_token_id=99, vocab_size=8)
    input_ids = torch.tensor([[1, 2, 3, 99, 99]], dtype=torch.long)
    logits = draft_model(input_ids).logits[:, -2:, :]
    assert logits.shape == (1, 2, 8)


def test_greedy_output_equals_baseline() -> None:
    target_model = ToyBigramLM({1: 2, 2: 3, 3: 1}, vocab_size=8)
    draft_model = ToyPARDDraft(mask_token_id=99, vocab_size=8)
    prompt = [1, 2, 3]
    baseline = autoregressive_generate(target_model, prompt, max_new_tokens=6, temperature=0.0)
    speculative, counters = run_pard_speculative_decode(
        target_model=target_model,
        draft_model=draft_model,
        prompt_ids=prompt,
        max_new_tokens=6,
        draft_len=3,
        mask_token_id=99,
        eos_token_id=None,
    )
    assert speculative == baseline
    assert counters["draft_forwards"] == counters["speculation_steps"]

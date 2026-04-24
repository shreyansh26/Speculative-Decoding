from __future__ import annotations

import torch

from common.sampling import autoregressive_generate
from methods.medusa_1.inference.infer import (
    evaluate_medusa_posterior,
    generate_medusa_buffers,
    run_medusa_speculative_decode,
)
from methods.medusa_1.training.train import (
    MedusaHeads,
    compute_medusa_loss,
    freeze_base_model,
)


class ToyHiddenOutput:
    def __init__(self, logits: torch.Tensor, hidden_states: dict[int, torch.Tensor]) -> None:
        self.logits = logits
        self.hidden_states = hidden_states


class ToyHiddenLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")
        self.config = type("Cfg", (), {"num_hidden_layers": 1, "hidden_size": vocab_size})()

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
        hidden_state_indices=None,
        **_: object,
    ) -> ToyHiddenOutput:
        predictions = (input_ids % 3) + 1
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], self.vocab_size), -50.0)
        logits.scatter_(2, predictions.unsqueeze(-1), 50.0)
        one_hot = torch.nn.functional.one_hot(input_ids, num_classes=self.vocab_size).float()
        hidden_states = {1: one_hot} if output_hidden_states else {}
        return ToyHiddenOutput(logits=logits, hidden_states=hidden_states)


def test_base_params_are_frozen() -> None:
    model = torch.nn.Linear(4, 4)
    freeze_base_model(model)
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_head_loss_shape_is_correct() -> None:
    medusa_heads = MedusaHeads(hidden_size=8, vocab_size=8, num_heads=4, num_layers=1)
    hidden_states = torch.randn(2, 6, 8)
    input_ids = torch.tensor([[1, 2, 3, 1, 2, 3], [2, 3, 1, 2, 3, 1]], dtype=torch.long)
    total_loss, per_head_losses = compute_medusa_loss(
        medusa_heads=medusa_heads,
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_weights=(1.0, 0.8, 0.6, 0.4),
    )
    assert total_loss.ndim == 0
    assert len(per_head_losses) == 4


def test_greedy_output_equals_baseline() -> None:
    target_model = ToyHiddenLM(vocab_size=8)
    medusa_heads = MedusaHeads(hidden_size=8, vocab_size=8, num_heads=3, num_layers=1)
    for head_idx, head in enumerate(medusa_heads.heads):
        head.lm_head.weight.data.zero_()
        for token in range(1, 4):
            predicted = ((token - 1 + head_idx + 2) % 3) + 1
            head.lm_head.weight.data[predicted, token] = 10.0

    prompt = [1]
    baseline = autoregressive_generate(target_model, prompt, max_new_tokens=6, temperature=0.0)
    speculative, counters = run_medusa_speculative_decode(
        target_model=target_model,
        medusa_heads=medusa_heads,
        prompt_ids=prompt,
        max_new_tokens=6,
        draft_len=3,
        eos_token_id=None,
    )
    assert speculative == baseline
    assert counters["proposed_draft_tokens"] > 0


def test_tree_buffers_include_valid_candidate_paths() -> None:
    buffers = generate_medusa_buffers(max_depth=3, topk=3, max_tree_nodes=16, device="cpu")
    assert buffers.attention_mask.shape[0] <= 16
    assert buffers.retrieve_indices.shape == buffers.retrieve_mask.shape
    assert bool(buffers.retrieve_mask[:, 0].all().item())
    assert int(buffers.position_offsets.max().item()) <= 3


def test_posterior_selects_longest_greedy_prefix() -> None:
    vocab_size = 8
    candidates = torch.tensor([[2, 3, 1], [2, 4, 1]], dtype=torch.long)
    valid = torch.ones_like(candidates, dtype=torch.bool)
    logits = torch.full((2, 3, vocab_size), -50.0)
    logits[:, 0, 3] = 50.0
    logits[:, 1, 1] = 50.0
    best_row, accepted = evaluate_medusa_posterior(
        logits,
        candidates,
        valid,
        allow_divergence=False,
        posterior_temperature=0.0,
        posterior_threshold=0.09,
        posterior_alpha=0.3,
    )
    assert best_row == 0
    assert accepted == 2

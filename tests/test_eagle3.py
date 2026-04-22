from __future__ import annotations

import torch

from common.sampling import autoregressive_generate
from methods.eagle3.inference.infer import run_eagle3_speculative_decode
from methods.eagle3.training.train import Eagle3Drafter, fuse_hidden_states, run_drafter_training_step


class ToyEagleOutput:
    def __init__(self, logits: torch.Tensor, hidden_states: dict[int, torch.Tensor]) -> None:
        self.logits = logits
        self.hidden_states = hidden_states


class ToyEagleTarget(torch.nn.Module):
    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False, hidden_state_indices=None, **_: object):
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], self.vocab_size), -50.0)
        predictions = ((input_ids - 1) % 3) + 1
        logits.scatter_(2, predictions.unsqueeze(-1), 50.0)
        hidden = torch.nn.functional.one_hot(input_ids, num_classes=self.vocab_size).float()
        hidden_states = {0: hidden, 1: hidden, 2: hidden} if output_hidden_states else {}
        return ToyEagleOutput(logits=logits, hidden_states=hidden_states)


def test_hidden_layer_fusion_shape_is_correct() -> None:
    hidden_map = {
        0: torch.randn(2, 3, 4),
        1: torch.randn(2, 3, 4),
        2: torch.randn(2, 3, 4),
    }
    fused = fuse_hidden_states(hidden_map, [0, 1, 2])
    assert fused.shape == (2, 3, 12)


def test_training_time_test_loop_runs() -> None:
    drafter = Eagle3Drafter(hidden_size=4, vocab_size=8)
    fused = torch.randn(2, 12)
    prev = torch.tensor([1, 2], dtype=torch.long)
    labels = torch.tensor([[2, 3], [3, 1]], dtype=torch.long)
    loss = run_drafter_training_step(drafter, fused, prev, labels, mode="training_time_test")
    assert loss.ndim == 0


def test_greedy_output_equals_baseline() -> None:
    target = ToyEagleTarget(vocab_size=8)
    drafter = Eagle3Drafter(hidden_size=8, vocab_size=8)
    drafter.feature_fuser.weight.data.zero_()
    drafter.feature_fuser.bias.data.zero_()
    drafter.rnn.weight_ih.data.zero_()
    drafter.rnn.weight_hh.data.zero_()
    drafter.rnn.bias_ih.data.zero_()
    drafter.rnn.bias_hh.data.zero_()
    drafter.lm_head.weight.data.zero_()
    for token in range(1, 4):
        next_token = ((token - 1) % 3) + 1
        drafter.token_embedding.weight.data[token, next_token] = 10.0
        drafter.lm_head.weight.data[next_token, next_token] = 10.0

    prompt = [1]
    baseline = autoregressive_generate(target, prompt, max_new_tokens=4, temperature=0.0)
    speculative, counters = run_eagle3_speculative_decode(
        target_model=target,
        drafter=drafter,
        prompt_ids=prompt,
        max_new_tokens=4,
        selected_layers=[0, 1, 2],
        draft_len=2,
        eos_token_id=None,
    )
    assert speculative == baseline
    assert counters["draft_forwards"] > 0

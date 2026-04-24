import torch

from common.sampling import autoregressive_generate
from common.toy_models import ToyBigramLM
from methods.parallel_draft_models.inference.infer import run_pard_speculative_decode
from methods.parallel_draft_models.training.train import (
    IGNORE_INDEX,
    PARDExample,
    build_pard_batch,
    build_pard_training_example,
)


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
        first_mask = prefix.index(self.mask_token_id) if self.mask_token_id in prefix else input_ids.shape[1]
        last_real_position = first_mask - 1
        last_real = prefix[last_real_position]
        prediction_count = input_ids.shape[1] - last_real_position
        predictions = [((last_real - 1 + idx + 1) % 3) + 1 for idx in range(prediction_count)]
        for offset, token in enumerate(predictions):
            logits[0, last_real_position + offset, token] = 50.0
        return ToyPARDOutput(logits=logits)


def test_mask_token_input_builder_creates_correct_labels() -> None:
    sequence = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    input_ids, labels = build_pard_training_example(sequence, draft_len=2, mask_token_id=99)
    assert input_ids.tolist() == [1, 2, 3, 99, 99]
    assert labels.tolist() == [4, 5]


def test_one_forward_returns_k_logits() -> None:
    draft_model = ToyPARDDraft(mask_token_id=99, vocab_size=8)
    input_ids = torch.tensor([[1, 2, 3, 99, 99]], dtype=torch.long)
    logits = draft_model(input_ids).logits[:, -3:, :]
    assert logits.shape == (1, 3, 8)
    assert torch.argmax(logits, dim=-1).tolist() == [[1, 2, 3]]


def test_pard_batch_expands_base_tokens_and_mask_blocks() -> None:
    example = PARDExample(prompt_id="toy", prompt_ids=[1, 2], completion_ids=[3, 4])
    batch = build_pard_batch(
        [example],
        draft_len=3,
        pard_token_id=99,
        pad_token_id=0,
        cod_ratio=1.0,
    )
    assert batch.input_ids.tolist() == [[1, 2, 3, 4, 99, 99, 99, 99, 99, 99, 99, 99]]
    assert batch.position_ids.tolist() == [[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]]
    assert batch.labels.shape == batch.input_ids.shape
    assert batch.labels[0, :4].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 3, 4]
    assert batch.labels[0, 4:8].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 3, 4]
    assert batch.labels[0, 8:12].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 3, 4]
    assert batch.attention_mask.shape == (1, 1, 12, 12)


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

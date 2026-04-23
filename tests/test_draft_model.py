from __future__ import annotations

import torch

from common.sampling import autoregressive_generate
from common.toy_models import ToyBigramLM
from methods.draft_model.inference.infer import run_draft_model_speculative_decode
from methods.draft_model.training.train import (
    MiniQwenConfig,
    build_draft_model,
    build_distillation_examples,
    make_distillation_batch,
    train_steps,
)


def test_tiny_draft_trains_one_batch() -> None:
    model = build_draft_model(
        vocab_size=32,
        max_position_embeddings=32,
        config=MiniQwenConfig(
            hidden_size=32,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
        ),
    )
    sequences = [
        torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long),
        torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.long),
    ]
    losses = train_steps(
        model=model,
        sequences=sequences,
        steps=4,
        batch_size=2,
        grad_accum=1,
        lr=5e-2,
        device="cpu",
        dtype=torch.float32,
    )
    assert len(losses) == 4
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] <= losses[0]


def test_decode_many_matches_repeated_decode_one() -> None:
    model = build_draft_model(
        vocab_size=32,
        max_position_embeddings=32,
        config=MiniQwenConfig(
            hidden_size=32,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
        ),
    )
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    draft = torch.tensor([[4, 5]], dtype=torch.long)

    prefetched = model.prefill(prompt)
    decode_many = model.decode_many(draft, cache=prefetched.cache)

    decode_one_first = model.decode_one(draft[:, :1], cache=prefetched.cache)
    decode_one_second = model.decode_one(draft[:, 1:], cache=decode_one_first.cache)

    expected_logits = torch.cat([decode_one_first.logits, decode_one_second.logits], dim=1)
    assert torch.allclose(decode_many.logits, expected_logits, atol=1e-5, rtol=1e-5)
    assert decode_many.cache is not None
    assert decode_one_second.cache is not None
    assert len(decode_many.cache) == len(decode_one_second.cache)
    for layer_many, layer_step in zip(decode_many.cache, decode_one_second.cache):
        assert layer_many is not None and layer_step is not None
        assert torch.allclose(layer_many.key, layer_step.key, atol=1e-5, rtol=1e-5)
        assert torch.allclose(layer_many.value, layer_step.value, atol=1e-5, rtol=1e-5)


def test_draft_model_inference_matches_toy_bigram_baseline() -> None:
    target_model = ToyBigramLM({1: 2, 2: 3, 3: 1}, vocab_size=8)
    draft_model = ToyBigramLM({1: 2, 2: 3, 3: 1}, vocab_size=8)
    prompt = [1, 2, 3]
    baseline = autoregressive_generate(target_model, prompt, max_new_tokens=6, temperature=0.0)
    speculative, counters = run_draft_model_speculative_decode(
        target_model=target_model,
        draft_model=draft_model,
        prompt_ids=prompt,
        max_new_tokens=6,
        draft_len=3,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=None,
    )
    assert speculative == baseline
    assert counters["draft_forwards"] >= counters["speculation_steps"]


def test_draft_model_inference_matches_baseline_with_mismatched_real_models() -> None:
    target_model = build_draft_model(
        vocab_size=32,
        max_position_embeddings=32,
        config=MiniQwenConfig(
            hidden_size=32,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
        ),
    )
    draft_model = build_draft_model(
        vocab_size=32,
        max_position_embeddings=32,
        config=MiniQwenConfig(
            hidden_size=32,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
        ),
    )
    prompt = [1, 2, 3, 4]
    baseline = autoregressive_generate(target_model, prompt, max_new_tokens=8, temperature=0.0)
    speculative, _ = run_draft_model_speculative_decode(
        target_model=target_model,
        draft_model=draft_model,
        prompt_ids=prompt,
        max_new_tokens=8,
        draft_len=2,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=None,
    )
    assert speculative == baseline


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [index + 1 for index, _ in enumerate(text)]


def test_distillation_examples_mask_prompt_tokens(tmp_path) -> None:
    dataset_path = tmp_path / "distill.jsonl"
    dataset_path.write_text(
        '{"prompt_id":"row_1","prompt":"abcd","completion":"wxyz"}\n',
        encoding="utf-8",
    )
    examples = build_distillation_examples(dataset_path, tokenizer=FakeTokenizer(), seq_len=32)
    assert len(examples) == 1
    example = examples[0]
    assert example.token_ids.tolist() == [1, 2, 3, 4, 1, 2, 3, 4]
    assert example.loss_mask.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    batch, loss_mask = make_distillation_batch(examples, batch_size=1, step=0)
    assert batch.shape == (1, 8)
    assert loss_mask.shape == (1, 8)
    assert loss_mask[0, :4].sum().item() == 0.0
    assert loss_mask[0, 4:].sum().item() == 4.0


def test_distillation_examples_preserve_exact_token_ids(tmp_path) -> None:
    dataset_path = tmp_path / "distill_ids.jsonl"
    dataset_path.write_text(
        '{"prompt_id":"row_1","prompt":"ignored","completion":"ignored","prompt_ids":[8,9],"completion_ids":[3,4,5]}\n',
        encoding="utf-8",
    )
    examples = build_distillation_examples(dataset_path, tokenizer=FakeTokenizer(), seq_len=32)
    assert len(examples) == 1
    example = examples[0]
    assert example.token_ids.tolist() == [8, 9, 3, 4, 5]
    assert example.loss_mask.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]

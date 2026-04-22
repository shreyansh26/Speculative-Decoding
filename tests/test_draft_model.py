from __future__ import annotations

import torch

from common.sampling import autoregressive_generate
from common.toy_models import ToyBigramLM
from methods.draft_model.inference.infer import run_draft_model_speculative_decode
from methods.draft_model.training.train import (
    MiniQwenConfig,
    build_draft_model,
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

from __future__ import annotations

from common.sampling import autoregressive_generate
from common.toy_models import ToyBigramLM
from methods.ngram.inference.infer import find_ngram_draft, run_ngram_speculative_decode


def test_repeated_pattern_produces_expected_draft() -> None:
    history = [1, 2, 3, 1, 2, 3]
    draft = find_ngram_draft(
        history_ids=history,
        draft_len=3,
        prompt_lookup_min=2,
        prompt_lookup_max=3,
    )
    assert draft == [1, 2, 3]


def test_no_match_produces_empty_draft() -> None:
    history = [1, 2, 3, 4]
    draft = find_ngram_draft(
        history_ids=history,
        draft_len=4,
        prompt_lookup_min=2,
        prompt_lookup_max=3,
    )
    assert draft == []


def test_ngram_speculative_output_matches_baseline() -> None:
    model = ToyBigramLM({1: 2, 2: 3, 3: 1}, vocab_size=8)
    prompt = [1, 2, 3, 1, 2, 3]
    baseline = autoregressive_generate(model, prompt, max_new_tokens=9, temperature=0.0)
    speculative, counters = run_ngram_speculative_decode(
        model=model,
        prompt_ids=prompt,
        max_new_tokens=9,
        draft_len=3,
        prompt_lookup_min=2,
        prompt_lookup_max=3,
    )
    assert speculative == baseline
    assert counters["accepted_draft_tokens"] > 0
    assert counters["proposed_draft_tokens"] >= counters["accepted_draft_tokens"]

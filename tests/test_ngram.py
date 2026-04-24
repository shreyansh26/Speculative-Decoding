from __future__ import annotations

from types import SimpleNamespace

import torch

from common.sampling import autoregressive_generate
from common.toy_models import ToyBigramLM
from data.prepare_ngram_wiki import build_wikipedia_question_specs, render_extract_prompt
from methods.ngram.inference.infer import NgramDraftIndex, find_ngram_draft, run_ngram_speculative_decode


class CachedToyBigramLM(ToyBigramLM):
    def _cache(self, length: int):
        tensor = torch.empty(1, 1, length, 1)
        return [SimpleNamespace(key=tensor, value=tensor)]

    def prefill(self, input_ids: torch.Tensor):
        output = self(input_ids)
        return SimpleNamespace(logits=output.logits, cache=self._cache(input_ids.shape[1]))

    def decode_one(self, input_ids: torch.Tensor, cache):
        return self.decode_many(input_ids, cache)

    def decode_many(self, input_ids: torch.Tensor, cache):
        output = self(input_ids)
        prefix_len = cache[0].key.shape[2] if cache else 0
        return SimpleNamespace(
            logits=output.logits,
            cache=self._cache(prefix_len + input_ids.shape[1]),
        )


class TinyTokenizer:
    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool):
        assert not tokenize
        rendered = "\n".join(message["content"] for message in messages)
        return rendered + ("\nAnswer:" if add_generation_prompt else "")


def test_repeated_pattern_produces_expected_draft() -> None:
    history = [1, 2, 3, 1, 2, 3]
    draft = find_ngram_draft(
        history_ids=history,
        draft_len=3,
        prompt_lookup_min=2,
        prompt_lookup_max=3,
    )
    assert draft == [1, 2, 3]


def test_incremental_index_matches_batch_draft() -> None:
    history = [4, 5, 6, 4, 5, 6, 7]
    index = NgramDraftIndex(history, prompt_lookup_min=2, prompt_lookup_max=3)
    assert index.propose(2) == find_ngram_draft(history, 2, 2, 3)
    index.append(4)
    index.append(5)
    assert index.propose(3) == [6, 4, 5]


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


def test_cached_ngram_speculative_output_matches_baseline() -> None:
    model = CachedToyBigramLM({1: 2, 2: 3, 3: 1}, vocab_size=8)
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
    assert counters["target_forwards"] < len(speculative)


def test_wikipedia_question_specs_and_prompt_rendering() -> None:
    article = """
Solar System

The Solar System is the gravitationally bound system of the Sun and the objects that orbit it.
The largest planets formed from material in the early solar nebula over many gradual stages.

== Structure ==
The planetary system includes terrestrial planets, giant planets, dwarf planets, and many smaller bodies.
The asteroid belt occupies a wide region between the orbits of Mars and Jupiter.
"""
    specs = build_wikipedia_question_specs("Solar System", article, count=2)
    assert len(specs) == 2
    assert all(spec["expected_answer"].endswith(".") for spec in specs)

    prompt = render_extract_prompt(
        TinyTokenizer(),
        article_title="Solar System",
        article_text=article,
        section_title=specs[0]["section_title"],
        cue=specs[0]["cue"],
    )
    assert "Quote the single exact sentence" in prompt
    assert "Solar System" in prompt

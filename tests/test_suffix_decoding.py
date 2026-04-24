from __future__ import annotations

from types import SimpleNamespace

import torch

from common.sampling import autoregressive_generate
from common.toy_models import ToyBigramLM
from methods.suffix_decoding.inference.infer import SuffixIndex, propose_suffix_tokens, run_suffix_speculative_decode


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


def test_suffix_index_counts_next_token_frequencies() -> None:
    index = SuffixIndex(max_tree_depth=4)
    index.update([1, 2, 1, 2, 3])
    _, counts = index.next_token_distribution([1, 2])
    assert counts[1] == 1
    assert counts[3] == 1


def test_suffix_index_incremental_counts_match_batch_update() -> None:
    tokens = [1, 2, 3, 1, 2, 3, 4]
    batch = SuffixIndex(max_tree_depth=4)
    batch.update(tokens)
    incremental = SuffixIndex(max_tree_depth=4)
    for token in tokens:
        incremental.append(token)
    assert incremental.counts == batch.counts
    assert incremental.history == tokens


def test_proposal_length_adapts_to_match_length() -> None:
    index = SuffixIndex(max_tree_depth=8)
    index.update([1, 2, 3, 1, 2, 3, 4])
    short = propose_suffix_tokens(index, [2, 3], draft_len=4, max_spec_factor=0.5, min_token_prob=0.0)
    long = propose_suffix_tokens(index, [1, 2, 3], draft_len=4, max_spec_factor=1.0, min_token_prob=0.0)
    assert len(long) >= len(short)


def test_greedy_output_equals_baseline() -> None:
    model = ToyBigramLM({1: 2, 2: 3, 3: 1}, vocab_size=8)
    prompt = [1, 2, 3, 1, 2, 3]
    baseline = autoregressive_generate(model, prompt, max_new_tokens=6, temperature=0.0)
    speculative, counters = run_suffix_speculative_decode(
        model=model,
        prompt_ids=prompt,
        max_new_tokens=6,
        draft_len=4,
        max_tree_depth=8,
        max_spec_factor=1.0,
        min_token_prob=0.0,
        global_index=None,
        eos_token_id=None,
    )
    assert speculative == baseline
    assert counters["proposed_draft_tokens"] >= counters["accepted_draft_tokens"]


def test_cached_greedy_output_equals_baseline() -> None:
    model = CachedToyBigramLM({1: 2, 2: 3, 3: 1}, vocab_size=8)
    prompt = [1, 2, 3, 1, 2, 3]
    baseline = autoregressive_generate(model, prompt, max_new_tokens=9, temperature=0.0)
    speculative, counters = run_suffix_speculative_decode(
        model=model,
        prompt_ids=prompt,
        max_new_tokens=9,
        draft_len=4,
        max_tree_depth=8,
        max_spec_factor=1.0,
        min_token_prob=0.0,
        global_index=None,
        eos_token_id=None,
    )
    assert speculative == baseline
    assert counters["accepted_draft_tokens"] > 0
    assert counters["target_forwards"] < len(speculative)

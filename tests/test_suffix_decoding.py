from __future__ import annotations

from common.sampling import autoregressive_generate
from common.toy_models import ToyBigramLM
from methods.suffix_decoding.inference.infer import SuffixIndex, propose_suffix_tokens, run_suffix_speculative_decode


def test_suffix_index_counts_next_token_frequencies() -> None:
    index = SuffixIndex(max_tree_depth=4)
    index.update([1, 2, 1, 2, 3])
    _, counts = index.next_token_distribution([1, 2])
    assert counts[1] == 1
    assert counts[3] == 1


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

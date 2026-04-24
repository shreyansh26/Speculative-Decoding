from common.sampling import autoregressive_generate
from common.toy_models import ToyIncrementLM
from common.verification import greedy_verify, run_greedy_speculative_decode


def test_greedy_verification_accepts_all_correct_drafts() -> None:
    model = ToyIncrementLM(vocab_size=32)
    result = greedy_verify(model, prefix_ids=[5], draft_ids=[6, 7, 8])
    assert result.accepted_ids == [6, 7, 8]
    assert result.emitted_ids == [6, 7, 8, 9]
    assert result.mismatch_index is None
    assert result.accepted_draft_tokens == 3


def test_greedy_verification_rejects_first_mismatch() -> None:
    model = ToyIncrementLM(vocab_size=32)
    result = greedy_verify(model, prefix_ids=[5], draft_ids=[6, 9, 10])
    assert result.accepted_ids == [6]
    assert result.emitted_ids == [6, 7]
    assert result.mismatch_index == 1
    assert result.accepted_draft_tokens == 1


def test_speculative_output_matches_autoregressive_baseline_on_toy_model() -> None:
    model = ToyIncrementLM(vocab_size=64)
    prompt = [3]
    baseline = autoregressive_generate(model, prompt, max_new_tokens=8, temperature=0.0)

    def perfect_draft_provider(prefix_ids: list[int], requested: int) -> list[int]:
        if requested == 0:
            return []
        last_token = prefix_ids[-1]
        return [int((last_token + offset + 1) % model.vocab_size) for offset in range(requested)]

    speculative, stats = run_greedy_speculative_decode(
        model=model,
        prompt_ids=prompt,
        max_new_tokens=8,
        draft_provider=perfect_draft_provider,
        draft_len=3,
    )

    assert speculative == baseline
    assert stats["speculation_steps"] > 0
    assert stats["accepted_draft_tokens"] > 0

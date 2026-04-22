# Progress

## 2026-04-22

- Initialized git and the `uv` Python 3.12 environment.
- Started phase 0 with the shared decoding contract, verifier, metrics utilities, tokenizer helpers, toy models, and a minimal Qwen-style causal LM implementation.
- Added phase-0 tests for greedy verification and metrics schema.
- Started phase 1 with a training-free N-gram proposer and its targeted tests.
- Started phase 2 with a compact draft-model trainer, checkpoint format, and speculative inference loop.
- Started phase 3 with frozen-base Medusa-1 heads and top-1 chain verification.
- Started phase 4 with a one-shot parallel draft model using appended mask slots.
- Started phase 5 with low/mid/high target-state fusion and a sequential EAGLE-style drafter.
- Started phase 6 with a bounded suffix-frequency index and optional global cache persistence.
- Fixed a BF16 rotary-embedding dtype bug in `common/qwen3.py` that blocked GPU execution for pretrained Qwen checkpoints.
- Reworked the baseline/speculative decode path to use KV-cache prefill/decode APIs, while preserving compatibility with the toy-model test doubles.
- Switched the `draft_model` trainer to support prompt/completion distillation with masked completion-only loss and checkpoint loading from a pretrained draft backbone.
- Generated an overfit distillation corpus from `Qwen/Qwen2.5-7B-Instruct` on the UltraChat-derived 300-row prompt pool using 16 greedy target tokens per prompt.
- Increased the in-distribution eval slice to 50 rows and used the 50 shortest prompts, normalized to the last 512 prompt tokens for stable 7B benchmarking on the 40 GB A100.
- Added epoch-aware draft training and benchmark warmup controls so the 300-row overfit run can be scheduled in epochs rather than sliding-window steps.
- Fixed a real-model draft verification correctness bug in `common/qwen3.py` by avoiding the SDPA GQA fast path for multi-token target verification while keeping the draft path compatible with the shared tests.
- Fine-tuned `Qwen/Qwen2.5-0.5B-Instruct` for 2 epochs on `data/ultrachat_300_trunc512_qwen7b_greedy16_ids.jsonl` and re-ran draft-model inference on the 50-row eval slice from the same 300-row pool.
- Current draft-model result on `Qwen/Qwen2.5-7B-Instruct` eval (`max_new_tokens=16`, 50 prompts):
  - `draft_len=2`: mean acceptance rate `0.8609`, mean speedup `0.6814`
  - `draft_len=4`: mean acceptance rate `0.8127`, mean speedup `0.6937`
  - `draft_len=6`: mean acceptance rate `0.7741`, mean speedup `0.6833`
- Acceptance is comfortably above the 40% target for all requested draft lengths, but speedup is still below the `>1.0x` bar. The best current setting is `draft_len=4`, and follow-up optimization likely needs a faster correct multi-token target verification path rather than more distillation.

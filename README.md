# Speculative Decoding From Scratch

This repository implements speculative decoding methods behind one shared decoding contract:

- one target-model interface
- one greedy verifier
- one metrics schema
- one baseline autoregressive path

Execution order follows `plans/plan.md`:

1. Phase 0: common infrastructure
2. Phase 1: n-gram
3. Phase 2: draft model
4. Phase 3: Medusa-1
5. Phase 4: PARD
6. Phase 5: EAGLE-3
7. Phase 6: suffix decoding

The project uses `uv` with Python 3.12:

```bash
uv venv --python 3.12 .venv
uv pip install -e .
```

## Draft Model

Current reference artifact:

- checkpoint: `checkpoints/draft_model_qwen25_05b_2ep`
- target model: `Qwen/Qwen2.5-7B-Instruct`
- draft base: `Qwen/Qwen2.5-0.5B-Instruct`
- eval set: `data/ultrachat_eval_50_short_trunc512.jsonl`
- decode length: `16`
- draft length: `2`

Latest non-vLLM inference result:

- file: `runs/draft_model_len2_optimized.summary.json`
- mean acceptance: `89.16%`
- mean speedup: `0.4066x`
- diverged prompts: `6/50`

Latest vLLM inference result:

- file: `runs/draft_model_vllm_len2_latency.summary.json`
- measurement: serial batch-size-1 latency with `warmup_prompts=1`
- mean baseline latency: `0.1105s`
- mean speculative latency: `0.1349s`
- mean latency speedup: `0.8514x`
- acceptance: `84.42%`
- diverged prompts: `2/50`

### Train

```bash
export CUDA_VISIBLE_DEVICES=3
. .venv/bin/activate
python methods/draft_model/training/train.py \
  --target-model-path Qwen/Qwen2.5-7B-Instruct \
  --data data/ultrachat_300_trunc512_qwen7b_greedy16_ids.jsonl \
  --output checkpoints/draft_model_qwen25_05b_2ep \
  --seq-len 528 \
  --epochs 2 \
  --batch-size 2 \
  --grad-accum 1 \
  --lr 5e-5 \
  --dtype bf16 \
  --device cuda \
  --init-model-path Qwen/Qwen2.5-0.5B-Instruct
```

### Non-vLLM Inference

```bash
export CUDA_VISIBLE_DEVICES=3
. .venv/bin/activate
python methods/draft_model/inference/infer.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --checkpoint-path checkpoints/draft_model_qwen25_05b_2ep \
  --prompts data/ultrachat_eval_50_short_trunc512.jsonl \
  --output runs/draft_model_len2_optimized.jsonl \
  --max-new-tokens 16 \
  --draft-len 2 \
  --dtype bf16 \
  --device cuda \
  --allow-divergence
```

### vLLM Baseline

```bash
export CUDA_VISIBLE_DEVICES=3
. .venv/bin/activate
python methods/draft_model/inference/infer_vllm.py \
  --mode baseline \
  --draft-len 2 \
  --dtype bf16 \
  --gpu-memory-utilization 0.25 \
  --serial-prompts \
  --warmup-prompts 1 \
  --max-num-seqs 1 \
  --baseline-summary-path runs/draft_model_vllm_len2_latency.baseline.json
```

### vLLM Speculative

```bash
export CUDA_VISIBLE_DEVICES=3
. .venv/bin/activate
python methods/draft_model/inference/infer_vllm.py \
  --mode speculative \
  --draft-len 2 \
  --dtype bf16 \
  --gpu-memory-utilization 0.25 \
  --serial-prompts \
  --warmup-prompts 1 \
  --max-num-seqs 1 \
  --skip-export \
  --export-dir checkpoints/vllm_exports/draft_model_len2 \
  --baseline-summary-path runs/draft_model_vllm_len2_latency.baseline.json \
  --output runs/draft_model_vllm_len2_latency.summary.json
```

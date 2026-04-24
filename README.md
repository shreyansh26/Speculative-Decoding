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

## EAGLE-3

Current reference artifact:

- checkpoint: `checkpoints/eagle3_qwen25_7b_eval100_ce_len3`
- vLLM export: `checkpoints/vllm_exports/eagle3_eval100_len2`
- target model: `Qwen/Qwen2.5-7B-Instruct`
- full distillation set: `data/ultrachat_3000_trunc1024_qwen25_7b_greedy128_ids.jsonl`
- reference overfit train/eval set: `data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl`
- eval setup: `100` train-overlap prompts, `max_new_tokens=128`
- best inference draft length: `2`

Latest local implementation benchmark results on GPU 4:

| Path | Baseline wall time | EAGLE wall time | Baseline mean latency | EAGLE mean latency | Baseline throughput | EAGLE throughput | Speedup | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM batched throughput (`max_num_seqs=16`) | `6.6325s` | `5.3677s` | n/a | n/a | `1827.83 tok/s` | `2258.50 tok/s` | `1.2356x` | `46.79%` |
| vLLM serial latency (`max_num_seqs=1`) | `74.5677s` | `50.2344s` | `0.7457s` | `0.5023s` | `162.35 tok/s` | `241.39 tok/s` | `1.4844x` | `44.72%` |
| non-vLLM PyTorch loop | `195.8243s` | `155.1596s` | `1.9582s` | `1.5516s` | `61.73 tok/s` | `77.91 tok/s` | `1.2621x` | `36.13%` |

NVIDIA ModelOpt comparison on the same 100-prompt eval slice:

| Path | Baseline wall time | ModelOpt EAGLE wall time | Baseline mean latency | ModelOpt EAGLE mean latency | Baseline throughput | ModelOpt EAGLE throughput | Speedup | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM batched throughput (`max_num_seqs=16`) | `6.7298s` | `5.6747s` | n/a | n/a | `1800.66 tok/s` | `2136.67 tok/s` | `1.1859x` | `42.92%` |
| vLLM serial latency (`max_num_seqs=1`) | `74.5472s` | `51.8290s` | `0.7455s` | `0.5183s` | `162.39 tok/s` | `233.96 tok/s` | `1.4383x` | `41.95%` |

Benchmark files:

- vLLM batched: `runs/eagle3_eval100_ce_len2_vllm_batched.summary.json`
- vLLM serial: `runs/eagle3_eval100_ce_len2_vllm_serial.summary.json`
- non-vLLM: `runs/eagle3_eval100_ce_len2_nonvllm.jsonl`
- ModelOpt vLLM batched: `runs/eagle3_modelopt_eval100_len2_vllm_batched.summary.json`
- ModelOpt vLLM serial: `runs/eagle3_modelopt_eval100_len2_vllm_serial.summary.json`

Output divergence from the baseline is diagnostic only for EAGLE-3 runs. The benchmark records `matches_baseline`, diverged prompt counts, and token-count mismatches, but speedup is computed from the measured wall time and generated-token throughput.

### Local EAGLE-3 Commands

Prepare the UltraChat distillation data with vLLM greedy completions:

```bash
export CUDA_VISIBLE_DEVICES=4
. .venv/bin/activate
python methods/eagle3/training/train.py prepare-data \
  --target-model-path Qwen/Qwen2.5-7B-Instruct \
  --output data/ultrachat_3000_trunc1024_qwen25_7b_greedy128_ids.jsonl \
  --eval-output data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl \
  --num-samples 3000 \
  --eval-samples 100 \
  --max-prompt-tokens 1024 \
  --completion-tokens 128 \
  --dtype bf16 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 1184
```

Train the current reference local checkpoint:

```bash
export CUDA_VISIBLE_DEVICES=4
. .venv/bin/activate
python methods/eagle3/training/train.py train \
  --target-model-path Qwen/Qwen2.5-7B-Instruct \
  --data data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl \
  --output checkpoints/eagle3_qwen25_7b_eval100_ce_len3 \
  --seq-len 1152 \
  --steps 3000 \
  --batch-size 1 \
  --grad-accum 1 \
  --lr 1e-4 \
  --draft-len 3 \
  --ttt-steps 3 \
  --num-draft-layers 1 \
  --selected-layers 1,13,24 \
  --loss-decay 1.0 \
  --loss-type ce \
  --grad-clip 0.5 \
  --dtype bf16 \
  --device cuda
```

Run non-vLLM PyTorch inference:

```bash
export CUDA_VISIBLE_DEVICES=4
. .venv/bin/activate
python methods/eagle3/inference/infer.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --checkpoint-path checkpoints/eagle3_qwen25_7b_eval100_ce_len3 \
  --prompts data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl \
  --output runs/eagle3_eval100_ce_len2_nonvllm.jsonl \
  --max-new-tokens 128 \
  --draft-len 2 \
  --dtype bf16 \
  --device cuda \
  --allow-divergence
```

Run vLLM batched throughput inference:

```bash
export CUDA_VISIBLE_DEVICES=4
. .venv/bin/activate
python methods/eagle3/inference/infer_vllm.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --checkpoint-path checkpoints/eagle3_qwen25_7b_eval100_ce_len3 \
  --export-dir checkpoints/vllm_exports/eagle3_eval100_len2 \
  --prompts data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl \
  --output runs/eagle3_eval100_ce_len2_vllm_batched.summary.json \
  --baseline-summary-path runs/eagle3_eval100_ce_len2_vllm_batched.baseline.json \
  --max-new-tokens 128 \
  --draft-len 2 \
  --gpu-memory-utilization 0.4 \
  --max-model-len 1280 \
  --max-num-seqs 16 \
  --warmup-prompts 0
```

Run vLLM serial latency inference:

```bash
export CUDA_VISIBLE_DEVICES=4
. .venv/bin/activate
python methods/eagle3/inference/infer_vllm.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --draft-model-path checkpoints/vllm_exports/eagle3_eval100_len2 \
  --skip-export \
  --prompts data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl \
  --output runs/eagle3_eval100_ce_len2_vllm_serial.summary.json \
  --baseline-summary-path runs/eagle3_eval100_ce_len2_vllm_serial.baseline.json \
  --max-new-tokens 128 \
  --draft-len 2 \
  --gpu-memory-utilization 0.4 \
  --max-model-len 1280 \
  --max-num-seqs 1 \
  --serial-prompts \
  --warmup-prompts 1
```

### ModelOpt EAGLE-3 Comparison Commands

The ModelOpt path is intentionally isolated in `methods/eagle3/modelopt_experiment.py` so it can be deleted without touching the local implementation. It trains with `modelopt.torch.speculative`, exports the official ModelOpt checkpoint, converts it to the vLLM/speculators checkpoint layout, and benchmarks through the same local vLLM runner. ModelOpt comparison inference is vLLM-only.

One-time experiment dependencies:

```bash
mkdir -p ref_repos
if [ ! -d ref_repos/Model-Optimizer ]; then
  git clone https://github.com/NVIDIA/Model-Optimizer.git ref_repos/Model-Optimizer
fi
uv pip install --python .venv/bin/python -e ref_repos/Model-Optimizer \
  accelerate peft scipy pulp nvidia-ml-py omegaconf
```

Train, export, convert, and run the default batched vLLM benchmark:

```bash
export CUDA_VISIBLE_DEVICES=4
. .venv/bin/activate
python methods/eagle3/modelopt_experiment.py \
  --mode all \
  --overwrite-data \
  --overwrite-output-dir \
  --overwrite-exports
```

Run only the ModelOpt vLLM batched benchmark after the export exists:

```bash
export CUDA_VISIBLE_DEVICES=4
. .venv/bin/activate
python methods/eagle3/modelopt_experiment.py \
  --mode bench \
  --max-num-seqs 16 \
  --summary-output runs/eagle3_modelopt_eval100_len2_vllm_batched.summary.json \
  --baseline-summary-path runs/eagle3_modelopt_eval100_len2_vllm_batched.baseline.json
```

Run only the ModelOpt vLLM serial latency benchmark after the export exists:

```bash
export CUDA_VISIBLE_DEVICES=4
. .venv/bin/activate
python methods/eagle3/modelopt_experiment.py \
  --mode bench \
  --serial-prompts \
  --max-num-seqs 1 \
  --summary-output runs/eagle3_modelopt_eval100_len2_vllm_serial.summary.json \
  --baseline-summary-path runs/eagle3_modelopt_eval100_len2_vllm_serial.baseline.json
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

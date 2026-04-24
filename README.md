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

Current reference artifacts:

- local checkpoint, `ttt_steps=3`: `checkpoints/eagle3_qwen25_7b_eval100_ce_len3`
- local vLLM export, `ttt_steps=3`: `checkpoints/vllm_exports/eagle3_eval100_len2`
- local checkpoint, `ttt_steps=6` and best vLLM result: `checkpoints/eagle3_qwen25_7b_eval100_ce_ttt6_len3`
- local vLLM export, `ttt_steps=6`: `checkpoints/vllm_exports/eagle3_eval100_ttt6_len2`
- target model: `Qwen/Qwen2.5-7B-Instruct`
- full distillation set: `data/ultrachat_3000_trunc1024_qwen25_7b_greedy128_ids.jsonl`
- reference overfit train/eval set: `data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl`
- eval setup: `100` train-overlap prompts, `max_new_tokens=128`
- best inference draft length: `2`

`draft_len=2` is the inference setting: each EAGLE-3 speculation step uses one target/base seed token plus up to two EAGLE-proposed speculative tokens, so a fully accepted step can emit up to three tokens. The checkpoint suffix `len3` refers to the training rollout/checkpoint configuration, not the best inference draft length.

Latest local implementation benchmark results on GPU 4. All vLLM rows use `--gpu-memory-utilization 0.85` for both the baseline engine and the speculative engine.

| Path | Train TTT | Baseline wall time | EAGLE wall time | Baseline mean latency | EAGLE mean latency | Baseline throughput | EAGLE throughput | Speedup | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM batched throughput (`max_num_seqs=16`) | `3` | `6.6098s` | `5.5455s` | n/a | n/a | `1834.09 tok/s` | `2186.82 tok/s` | `1.1919x` | `46.29%` |
| vLLM serial latency (`max_num_seqs=1`) | `3` | `74.3867s` | `50.1959s` | `0.7439s` | `0.5020s` | `162.74 tok/s` | `241.57 tok/s` | `1.4819x` | `44.72%` |
| vLLM batched throughput (`max_num_seqs=16`) | `6` | `6.6371s` | `4.9285s` | n/a | n/a | `1825.80 tok/s` | `2459.57 tok/s` | `1.3467x` | `59.78%` |
| vLLM serial latency (`max_num_seqs=1`) | `6` | `74.4072s` | `45.0488s` | `0.7441s` | `0.4505s` | `162.70 tok/s` | `269.17 tok/s` | `1.6517x` | `56.41%` |
| non-vLLM PyTorch loop | `3` | `195.8243s` | `155.1596s` | `1.9582s` | `1.5516s` | `61.73 tok/s` | `77.91 tok/s` | `1.2621x` | `36.13%` |
| non-vLLM PyTorch loop | `6` | `193.4679s` | `162.2932s` | `1.9347s` | `1.6229s` | `62.52 tok/s` | `74.53 tok/s` | `1.1921x` | `28.22%` |

NVIDIA ModelOpt comparison on the same 100-prompt eval slice:

| Path | Baseline wall time | ModelOpt EAGLE wall time | Baseline mean latency | ModelOpt EAGLE mean latency | Baseline throughput | ModelOpt EAGLE throughput | Speedup | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM batched throughput (`max_num_seqs=16`) | `6.6179s` | `5.5881s` | n/a | n/a | `1831.85 tok/s` | `2165.32 tok/s` | `1.1843x` | `43.00%` |
| vLLM serial latency (`max_num_seqs=1`) | `74.4597s` | `51.8207s` | `0.7446s` | `0.5182s` | `162.58 tok/s` | `234.00 tok/s` | `1.4369x` | `41.95%` |

Benchmark files:

- vLLM batched: `runs/eagle3_eval100_ce_len2_vllm_batched.summary.json`
- vLLM serial: `runs/eagle3_eval100_ce_len2_vllm_serial.summary.json`
- vLLM batched, `ttt_steps=6`: `runs/eagle3_eval100_ce_ttt6_len2_vllm_batched.summary.json`
- vLLM serial, `ttt_steps=6`: `runs/eagle3_eval100_ce_ttt6_len2_vllm_serial.summary.json`
- non-vLLM: `runs/eagle3_eval100_ce_len2_nonvllm.jsonl`
- non-vLLM, `ttt_steps=6`: `runs/eagle3_eval100_ce_ttt6_len2_nonvllm.jsonl`
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

For the higher-TTT local checkpoint used in the `ttt_steps=6` rows, run the same training command with `--output checkpoints/eagle3_qwen25_7b_eval100_ce_ttt6_len3` and `--ttt-steps 6`.

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

Run non-vLLM PyTorch inference for the `ttt_steps=6` checkpoint:

```bash
export CUDA_VISIBLE_DEVICES=4
. .venv/bin/activate
python methods/eagle3/inference/infer.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --checkpoint-path checkpoints/eagle3_qwen25_7b_eval100_ce_ttt6_len3 \
  --prompts data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl \
  --output runs/eagle3_eval100_ce_ttt6_len2_nonvllm.jsonl \
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
  --gpu-memory-utilization 0.85 \
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
  --gpu-memory-utilization 0.85 \
  --max-model-len 1280 \
  --max-num-seqs 1 \
  --serial-prompts \
  --warmup-prompts 1
```

Run vLLM batched throughput inference for the `ttt_steps=6` checkpoint:

```bash
export CUDA_VISIBLE_DEVICES=4
. .venv/bin/activate
python methods/eagle3/inference/infer_vllm.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --draft-model-path checkpoints/vllm_exports/eagle3_eval100_ttt6_len2 \
  --skip-export \
  --prompts data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl \
  --output runs/eagle3_eval100_ce_ttt6_len2_vllm_batched.summary.json \
  --baseline-summary-path runs/eagle3_eval100_ce_ttt6_len2_vllm_batched.baseline.json \
  --max-new-tokens 128 \
  --draft-len 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 1280 \
  --max-num-seqs 16 \
  --warmup-prompts 0
```

Run vLLM serial latency inference for the `ttt_steps=6` checkpoint:

```bash
export CUDA_VISIBLE_DEVICES=4
. .venv/bin/activate
python methods/eagle3/inference/infer_vllm.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --draft-model-path checkpoints/vllm_exports/eagle3_eval100_ttt6_len2 \
  --skip-export \
  --prompts data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl \
  --output runs/eagle3_eval100_ce_ttt6_len2_vllm_serial.summary.json \
  --baseline-summary-path runs/eagle3_eval100_ce_ttt6_len2_vllm_serial.baseline.json \
  --max-new-tokens 128 \
  --draft-len 2 \
  --gpu-memory-utilization 0.85 \
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
  --gpu-memory-utilization 0.85 \
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
  --gpu-memory-utilization 0.85 \
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
  --gpu-memory-utilization 0.85 \
  --summary-output runs/eagle3_modelopt_eval100_len2_vllm_serial.summary.json \
  --baseline-summary-path runs/eagle3_modelopt_eval100_len2_vllm_serial.baseline.json
```

## Draft Model

Current reference artifact:

- checkpoint: `checkpoints/draft_model_qwen25_05b_ultrachat3000`
- target model: `Qwen/Qwen2.5-7B-Instruct`
- draft base: `Qwen/Qwen2.5-0.5B-Instruct`
- full distillation set: `data/ultrachat_3000_trunc1024_qwen25_7b_greedy128_ids.jsonl`
- reference eval set: `data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl`
- eval setup: `100` train-overlap prompts, `max_new_tokens=128`
- best measured inference draft length: `2`

The current draft-model checkpoint was trained from scratch on the 3000-row UltraChat/Qwen2.5-7B completion set, not resumed from the 100-row overfit checkpoint. Final teacher-forced eval on the 100-row eval file:

- eval loss: `0.1941`
- top-1 match: `95.17%`
- acceptance proxy: `92.86%`
- mean accepted tokens per step proxy: `1.8567`

Latest draft-model benchmark results on GPU 4. All vLLM rows use `--gpu-memory-utilization 0.85` for both the baseline engine and the speculative engine.

| Path | Baseline wall time | Draft-model wall time | Baseline mean latency | Draft-model mean latency | Baseline throughput | Draft-model throughput | Speedup | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM batched throughput (`max_num_seqs=16`) | `6.6198s` | `6.9635s` | n/a | n/a | `1830.57 tok/s` | `1740.93 tok/s` | `0.9506x` | `82.09%` |
| vLLM serial latency (`max_num_seqs=1`) | `74.4407s` | `71.4878s` | `0.7444s` | `0.7149s` | `162.63 tok/s` | `169.65 tok/s` | `1.0413x` | `81.55%` |
| non-vLLM PyTorch loop | `201.1899s` | `368.4114s` | `2.0119s` | `3.6841s` | `60.21 tok/s` | `32.88 tok/s` | `0.5461x` | `81.79%` |

For vLLM batched throughput, `draft_len=3` was worse on the same baseline: `0.8710x` speedup with `77.09%` acceptance. The non-vLLM PyTorch path reaches high acceptance but remains slower because it pays a separate 0.5B draft-model forward loop in Python.

Benchmark files:

- vLLM batched: `runs/draft_model_ultrachat3000_len2_vllm_batched.summary.json`
- vLLM serial: `runs/draft_model_ultrachat3000_len2_vllm_serial.summary.json`
- vLLM batched, `draft_len=3`: `runs/draft_model_ultrachat3000_len3_vllm_batched.summary.json`
- non-vLLM: `runs/draft_model_ultrachat3000_len2_nonvllm.jsonl`

Output divergence from the baseline is diagnostic only for draft-model runs. The benchmark records `matches_baseline`, diverged prompt counts, and token-count mismatches, but speedup is computed from measured wall time and generated-token throughput.

### Draft-Model Commands

Train the current reference checkpoint:

```bash
export CUDA_VISIBLE_DEVICES=4
. .venv/bin/activate
python methods/draft_model/training/train.py \
  --target-model-path Qwen/Qwen2.5-7B-Instruct \
  --data data/ultrachat_3000_trunc1024_qwen25_7b_greedy128_ids.jsonl \
  --eval-data data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl \
  --output checkpoints/draft_model_qwen25_05b_ultrachat3000 \
  --seq-len 1152 \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 1 \
  --lr 5e-5 \
  --weight-decay 0.0 \
  --max-grad-norm 0.5 \
  --eval-interval 1000 \
  --eval-batch-size 1 \
  --eval-draft-len 2 \
  --log-interval 250 \
  --dtype bf16 \
  --device cuda \
  --init-model-path Qwen/Qwen2.5-0.5B-Instruct
```

Run non-vLLM PyTorch inference:

```bash
export CUDA_VISIBLE_DEVICES=4
. .venv/bin/activate
python methods/draft_model/inference/infer.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --checkpoint-path checkpoints/draft_model_qwen25_05b_ultrachat3000 \
  --prompts data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl \
  --output runs/draft_model_ultrachat3000_len2_nonvllm.jsonl \
  --max-new-tokens 128 \
  --draft-len 2 \
  --dtype bf16 \
  --device cuda
```

Run vLLM batched throughput inference:

```bash
export CUDA_VISIBLE_DEVICES=4
. .venv/bin/activate
python methods/draft_model/inference/infer_vllm.py \
  --mode both \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --draft-model-path checkpoints/draft_model_qwen25_05b_ultrachat3000 \
  --prompts data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl \
  --output runs/draft_model_ultrachat3000_len2_vllm_batched.summary.json \
  --baseline-summary-path runs/draft_model_ultrachat3000_len2_vllm_batched.baseline.json \
  --max-new-tokens 128 \
  --draft-len 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 1280 \
  --max-num-seqs 16 \
  --warmup-prompts 0
```

Run vLLM serial latency inference:

```bash
export CUDA_VISIBLE_DEVICES=4
. .venv/bin/activate
python methods/draft_model/inference/infer_vllm.py \
  --mode both \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --draft-model-path checkpoints/draft_model_qwen25_05b_ultrachat3000 \
  --prompts data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl \
  --output runs/draft_model_ultrachat3000_len2_vllm_serial.summary.json \
  --baseline-summary-path runs/draft_model_ultrachat3000_len2_vllm_serial.baseline.json \
  --max-new-tokens 128 \
  --draft-len 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 1280 \
  --max-num-seqs 1 \
  --serial-prompts \
  --warmup-prompts 1
```

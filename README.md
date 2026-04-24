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

## Parallel Draft Model (PARD)

Current reference artifact:

- checkpoint: `checkpoints/parallel_draft_models_qwen25_05b_ultrachat3000`
- target model: `Qwen/Qwen2.5-7B-Instruct`
- draft base: `Qwen/Qwen2.5-0.5B-Instruct`
- full distillation set: `data/ultrachat_3000_trunc1024_qwen25_7b_greedy128_ids.jsonl`
- reference eval set: `data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl`
- eval setup: `100` train-overlap prompts, `max_new_tokens=128`
- best measured non-vLLM inference draft length: `5`
- best measured vLLM inference draft length: `3`

The PARD checkpoint is a standard Hugging Face Qwen2 draft-model directory that vLLM can load directly with `parallel_drafting=True`. It exports the target tokenizer vocabulary size (`152064`), `pard_token=151665`, and `spd_type=pard`.

The checkpoint was trained from `Qwen/Qwen2.5-0.5B-Instruct` on the 3000-row UltraChat/Qwen2.5-7B completion set with the PARD `draft_len=8` objective. Final teacher-forced eval proxy on the 100-row train-overlap eval file:

- first-token match: `92.70%`
- length-8 acceptance proxy: `31.82%`
- mean accepted tokens per PARD step proxy: `2.5375`

Latest PARD benchmark results on GPU 4. All vLLM rows use fixed `gpu_memory_utilization=0.85` for both the baseline engine and the speculative engine.

| Path | Inference draft length | Baseline wall time | PARD wall time | Baseline mean latency | PARD mean latency | Baseline throughput | PARD throughput | Speedup | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM batched throughput (`max_num_seqs=16`) | `3` | `6.6162s` | `5.0411s` | n/a | n/a | `1832.33 tok/s` | `2403.04 tok/s` | `1.3124x` | `60.96%` |
| vLLM serial latency (`max_num_seqs=1`) | `3` | `74.3739s` | `40.6290s` | `0.7437s` | `0.4063s` | `162.77 tok/s` | `298.51 tok/s` | `1.8306x` | `60.22%` |
| non-vLLM PyTorch loop | `5` | `214.5692s` | `155.2519s` | `2.1457s` | `1.5525s` | `56.19 tok/s` | `77.66 tok/s` | `1.3821x` | `30.96%` |

The non-vLLM summary JSON reports per-prompt means of `1.4424x` speedup, `80.99 tok/s` PARD throughput, and `33.39%` acceptance. The table above uses aggregate wall-clock timing across all 100 prompts for consistency with the vLLM rows.

vLLM batched sweep on the same baseline:

| Inference draft length | Speedup | PARD throughput | Acceptance |
| ---: | ---: | ---: | ---: |
| `2` | `1.2672x` | `2321.55 tok/s` | `74.77%` |
| `3` | `1.3124x` | `2403.04 tok/s` | `60.96%` |
| `4` | `1.3004x` | `2381.77 tok/s` | `50.50%` |
| `5` | `1.2156x` | `2225.08 tok/s` | `41.40%` |

Non-vLLM 20-prompt screening sweep:

| Inference draft length | Speedup | PARD throughput | Acceptance |
| ---: | ---: | ---: | ---: |
| `2` | `1.2092x` | `70.69 tok/s` | `60.06%` |
| `3` | `1.3220x` | `73.25 tok/s` | `47.28%` |
| `4` | `1.3895x` | `79.43 tok/s` | `38.42%` |
| `5` | `1.4062x` | `80.81 tok/s` | `32.58%` |
| `6` | `1.3806x` | `80.15 tok/s` | `26.71%` |

Benchmark files:

- non-vLLM full: `runs/parallel_draft_models_len5_nonvllm.jsonl`, `runs/parallel_draft_models_len5_nonvllm.summary.json`
- vLLM batched baseline: `runs/parallel_draft_models_vllm_batched.baseline.json`
- vLLM batched best: `runs/parallel_draft_models_len3_vllm_batched.summary.json`
- vLLM serial baseline: `runs/parallel_draft_models_vllm_serial.baseline.json`
- vLLM serial best: `runs/parallel_draft_models_len3_vllm_serial.summary.json`
- vLLM batched sweep: `runs/parallel_draft_models_len2_vllm_batched.summary.json`, `runs/parallel_draft_models_len4_vllm_batched.summary.json`, `runs/parallel_draft_models_len5_vllm_batched.summary.json`
- non-vLLM screening sweep: `runs/parallel_draft_models_len2_nonvllm_20.summary.json` through `runs/parallel_draft_models_len6_nonvllm_20.summary.json`

Output divergence from the baseline is diagnostic only for PARD runs. The benchmark records `matches_baseline`, diverged prompt counts, and token-count mismatches, but speedup is computed from measured wall time and generated-token throughput.

### PARD Commands

Train the current reference checkpoint:

```bash
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate
python methods/parallel_draft_models/training/train.py \
  --target-model-path Qwen/Qwen2.5-7B-Instruct \
  --draft-base-model-path Qwen/Qwen2.5-0.5B-Instruct \
  --data data/ultrachat_3000_trunc1024_qwen25_7b_greedy128_ids.jsonl \
  --eval-data data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl \
  --output checkpoints/parallel_draft_models_qwen25_05b_ultrachat3000 \
  --steps 1500 \
  --batch-size 8 \
  --grad-accum 1 \
  --seq-len 512 \
  --draft-len 8 \
  --lr 3e-5 \
  --cod-ratio 0.7 \
  --cod-min-ratio 0.2 \
  --eval-limit 100 \
  --log-interval 25 \
  --eval-interval 250 \
  --dtype bf16 \
  --device cuda \
  --gradient-checkpointing
```

Run non-vLLM PyTorch inference with the measured best setting:

```bash
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate
python methods/parallel_draft_models/inference/infer.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --checkpoint-path checkpoints/parallel_draft_models_qwen25_05b_ultrachat3000 \
  --prompts data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl \
  --output runs/parallel_draft_models_len5_nonvllm.jsonl \
  --max-new-tokens 128 \
  --draft-len 5 \
  --dtype bf16 \
  --device cuda \
  --warmup-prompts 1
```

Run vLLM batched throughput inference with the measured best setting. `infer_vllm.py` intentionally fixes `gpu_memory_utilization=0.85` for both baseline and speculative engines.

```bash
export CUDA_VISIBLE_DEVICES=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
. .venv/bin/activate
python methods/parallel_draft_models/inference/infer_vllm.py \
  --mode both \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --checkpoint-path checkpoints/parallel_draft_models_qwen25_05b_ultrachat3000 \
  --prompts data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl \
  --output runs/parallel_draft_models_len3_vllm_batched.summary.json \
  --baseline-summary-path runs/parallel_draft_models_vllm_batched.baseline.json \
  --max-new-tokens 128 \
  --draft-len 3 \
  --dtype bf16 \
  --max-model-len 1280 \
  --max-num-seqs 16 \
  --seed 0
```

Run vLLM serial latency inference:

```bash
export CUDA_VISIBLE_DEVICES=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
. .venv/bin/activate
python methods/parallel_draft_models/inference/infer_vllm.py \
  --mode both \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --checkpoint-path checkpoints/parallel_draft_models_qwen25_05b_ultrachat3000 \
  --prompts data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl \
  --output runs/parallel_draft_models_len3_vllm_serial.summary.json \
  --baseline-summary-path runs/parallel_draft_models_vllm_serial.baseline.json \
  --max-new-tokens 128 \
  --draft-len 3 \
  --dtype bf16 \
  --max-model-len 1280 \
  --max-num-seqs 1 \
  --serial-prompts \
  --warmup-prompts 1 \
  --seed 0
```

Run only a speculative vLLM candidate against an existing baseline summary:

```bash
export CUDA_VISIBLE_DEVICES=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
. .venv/bin/activate
python methods/parallel_draft_models/inference/infer_vllm.py \
  --mode speculative \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --checkpoint-path checkpoints/parallel_draft_models_qwen25_05b_ultrachat3000 \
  --prompts data/ultrachat_3000_train_eval100_qwen25_7b_greedy128_ids.jsonl \
  --baseline-summary-path runs/parallel_draft_models_vllm_batched.baseline.json \
  --output runs/parallel_draft_models_len4_vllm_batched.summary.json \
  --max-new-tokens 128 \
  --draft-len 4 \
  --dtype bf16 \
  --max-model-len 1280 \
  --max-num-seqs 16 \
  --seed 0
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

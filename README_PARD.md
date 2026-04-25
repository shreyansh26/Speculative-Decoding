# PARD Explainer

Paper: [Parallel Draft Model (PARD)](https://arxiv.org/abs/2504.18583)

This document explains how PARD works in theory and how the implementation in this repo works in practice.

Relevant code:

- `methods/parallel_draft_models/training/train.py`
- `methods/parallel_draft_models/inference/infer.py`
- `methods/parallel_draft_models/inference/infer_vllm.py`
- `methods/draft_model/inference/infer.py` for contrast with the normal draft-model method
- `common/verification.py` for the standard target-side verification logic

## 1. What PARD Is

PARD is still speculative decoding. The outer loop is the usual one:

1. A draft model proposes `K` candidate tokens.
2. The target model verifies those candidates in parallel.
3. We emit the longest accepted prefix plus one target token.

The difference is the draft phase.

Normal draft-model speculation uses an autoregressive draft model:

```text
P(x_n, ..., x_{n+K-1} | x_<n)
= ∏ P(x_{n+k} | x_<n+k)
```

So the draft model must run sequentially. Token `k+1` depends on token `k`.

PARD changes the draft model so it predicts all `K` draft tokens from one forward pass:

```text
P(x_n, ..., x_{n+K-1} | x_<n)
= ∏ P(x_{n+k} | x_<n, m_0, ..., m_{k-1})
```

where `m_i` are mask placeholders.

That is the whole point of PARD: keep the normal target-side verification idea, but remove the sequential draft-side dependence.

## 2. What the Draft Model Looks Like

In this repo, the PARD draft model is still a standard causal LM. It is not a Medusa-style multi-head model.

It still has:

- normal token embeddings
- normal transformer layers
- one LM head over the vocabulary

What changes is the training objective and the presence of one extra special token:

- `pard_token`: a learned placeholder token used to represent unresolved future slots

During training we teach the model what it means when `pard_token` appears in specific positions and under a specific attention pattern.

At inference time we then exploit that learned behavior by appending `pard_token` to the suffix and reading multiple draft-token logits from one forward pass.

## 3. Key Terms Used Below

I will use this notation:

- original sequence: `[x0, x1, x2, x3]`
- `M`: the single special `pard_token`
- `draft_len = 3`

For training, I will refer to three blocks:

- `R[p]`: real-token block at original position `p`
- `M1[p]`: first mask block at original position `p`
- `M2[p]`: second mask block at original position `p`

Important: `M1[p]` and `M2[p]` contain the same token ID `M`. They differ because they live in different rows of the packed training layout and see different allowed context through the attention mask.

## 4. The Core Difference From `draft_model`

The normal `draft_model` implementation in this repo is in `methods/draft_model/inference/infer.py`.

That method is sequential on the draft side:

1. predict token 1
2. append token 1
3. predict token 2
4. append token 2
5. predict token 3

PARD is different:

1. append `K-1` copies of `pard_token`
2. run one draft forward
3. read the last `K` logits rows

So PARD still makes `K` draft predictions, but it does not decode them one by one.

## 5. Training: Why It Looks Strange

Inference is simple. Training is the subtle part.

Why? Because at inference time we want the model to understand a masked suffix like:

```text
[prefix..., M, M]
```

and produce:

- draft token 1 from the last real prefix row
- draft token 2 from the first `M` row
- draft token 3 from the second `M` row

To teach that behavior efficiently, the implementation packs several parallel prediction subtasks into one flattened training sequence. That is what `build_pard_batch(...)` in `methods/parallel_draft_models/training/train.py` is doing.

## 6. Training Example: Raw Packed Layout

Use this toy example:

- prompt token: `101`
- completion tokens: `201 202 203`
- `draft_len = 3`
- `pard_token = 999`

The original real sequence is:

```text
[101, 201, 202, 203]
```

PARD expands that into:

```text
R :  [101, 201, 202, 203]
M1:  [999, 999, 999, 999]
M2:  [999, 999, 999, 999]
```

Flattened `input_ids`:

```text
[101, 201, 202, 203, 999, 999, 999, 999, 999, 999, 999, 999]
```

This exact tensor is what the current implementation produces for the toy example.

### Why are `M1` and `M2` both all `999`?

Because block identity does not come from the token ID.

It comes from:

- which packed row the token is in
- what `position_id` it gets
- which earlier rows it may attend to

So `M1[2]` and `M2[2]` are both token `999`, but they do not mean the same thing.

## 7. Training `position_ids`

For the same example, the code builds:

```text
R :  [0, 1, 2, 3]
M1:  [0, 1, 2, 3]
M2:  [0, 1, 2, 3]
```

Flattened:

```text
[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
```

These positions are repeated on purpose.

Why? Because the flattened training sequence is not meant to represent twelve ordinary time steps. It is three parallel views of the same original four positions.

So:

- `R[2]`
- `M1[2]`
- `M2[2]`

all correspond to original position `2`, but under different masked contexts.

If we used positions `0..11`, the transformer would treat later blocks as later absolute time steps. That is not the intended semantics.

Important: repeated position IDs are a training artifact. In inference, positions are ordinary monotonic positions again.

## 8. Training `labels`

The raw labels tensor for the toy example is:

```text
[-100, 201, 202, 203, -100, 201, 202, 203, -100, -100, 202, 203]
```

where `-100` means `IGNORE_INDEX`.

At first glance this is confusing because values repeat across blocks. That is expected.

The same target token appears in multiple blocks because we want the model to learn to predict that token under different masked contexts.

Example:

- `202` appears in more than one block
- `203` appears in more than one block

That does not mean the subtasks are duplicated. It means the same ground-truth token is being supervised from different visible contexts.

### The important caveat: Hugging Face causal LM loss shifts by one token

This is the source of most confusion.

For `AutoModelForCausalLM`, the hidden state at row `i` predicts the label stored at row `i + 1`.

So you should not read the raw `labels` tensor directly as "this row predicts this token". You must read it after the one-token causal shift.

That is also why some rows near block boundaries exist only as bridge rows and have `-100` after shifting.

## 9. Training Mask: What the 4D Attention Mask Looks Like

The training mask built by `build_pard_attention_mask(...)` has shape:

```text
[batch, 1, total_len, total_len]
```

For the toy example:

- `seq_len = 4`
- `draft_len = 3`
- `total_len = 12`

Mask values are:

- `0.0` for allowed attention
- a very negative value for blocked attention

This is an additive attention mask, broadcast over heads.

### The best way to read the training mask

Do not start by staring at the full `12 x 12` matrix. The most useful view is:

- which input row is being used
- which next label that row predicts after the causal shift
- which earlier rows that input row may attend to

For the toy example, the effective supervised rows are:

| Input Row | Name | Input ID | Predicts Next Label | Allowed Attention |
| --- | --- | --- | --- | --- |
| 0 | `R[0]` | `101` | `201` | `R[0]` |
| 1 | `R[1]` | `201` | `202` | `R[0], R[1]` |
| 2 | `R[2]` | `202` | `203` | `R[0], R[1], R[2]` |
| 4 | `M1[0]` | `999` | `201` | `M1[0]` |
| 5 | `M1[1]` | `999` | `202` | `R[0], M1[1]` |
| 6 | `M1[2]` | `999` | `203` | `R[0], R[1], M1[2]` |
| 9 | `M2[1]` | `999` | `202` | `M1[0], M2[1]` |
| 10 | `M2[2]` | `999` | `203` | `R[0], M1[1], M2[2]` |

This table is much easier to reason about than the raw matrix.

### What this table is saying

- The real block `R` trains normal next-token prediction on the original sequence.
- The first mask block `M1` teaches the model how to predict targets when one unresolved future slot is represented by `pard_token`.
- The second mask block `M2` teaches the model how to predict targets when two unresolved future slots are represented through a chain of masked placeholders.

This is why `M1[2]` and `M2[2]` can use the same token ID `999` and still learn different behaviors:

- `M1[2]` attends to `R[0], R[1], M1[2]`
- `M2[2]` attends to `R[0], M1[1], M2[2]`

Same token ID, different visible context, different hidden state, different semantics.

### What about the rows that are not in the table?

Rows like:

- `R[3]`
- `M1[3]`
- `M2[0]`

exist in the flattened causal layout, but their shifted next label is ignored or they mainly act as bridge rows between blocks. They are part of the packing trick, not the rows you should use to build intuition.

## 10. What the 4D Mask Is Really Doing

The 4D mask is enforcing which predecessor information is available to each masked subtask.

In plain terms:

- a row in `R` sees the normal real-token prefix
- a row in `M1` sees a prefix where one future slot has been abstracted into a mask placeholder
- a row in `M2` sees a prefix where two future slots are represented through masked placeholders

The current implementation follows the reference-style packed layout. The easiest way to understand it is:

1. repeat the original positions across blocks
2. repeat the masked token across blocks
3. let the 4D attention mask decide what context each row can actually use
4. let the built-in causal one-token shift decide which next label each row trains

If you ignore either the mask or the causal shift, the layout looks wrong. You need both at once.

## 11. COD: Conditional Drop-Token

Without COD, the packed PARD batch multiplies token count by about `draft_len`.

That makes training expensive.

The paper addresses this with COD, and the implementation follows that idea in `build_pard_batch(...)`.

The rule is:

1. keep all rows from the real block
2. keep a decayed fraction from deeper mask blocks
3. only keep deeper rows whose masked predecessor chain is still valid

The keep ratio for block `d` is:

```text
max(cod_ratio ** d, cod_min_ratio)
```

## 12. COD Example Using the Same Toy Sequence

Use:

- original sequence: `[101, 201, 202, 203]`
- `draft_len = 3`
- `cod_ratio = 0.5`
- `cod_min_ratio = 0.2`

Before COD, the packed rows are:

```text
R[0] R[1] R[2] R[3] | M1[0] M1[1] M1[2] M1[3] | M2[0] M2[1] M2[2] M2[3]
```

### Step 1: keep all real rows

Always keep:

```text
R[0], R[1], R[2], R[3]
```

### Step 2: keep some rows from `M1`

For `M1`, keep ratio is:

```text
max(0.5^1, 0.2) = 0.5
```

So we keep about half of the `M1` positions.

Suppose the sampled kept positions are:

```text
M1[0], M1[1]
```

### Step 3: build valid candidates for `M2`

The implementation then computes:

```text
prev_indices = (selected + 1) % seq_len
```

If `selected = [0, 1]`, the valid candidate positions for `M2` become:

```text
[1, 2]
```

That is the key dependency rule.

Why? Because deeper rows are only kept if the earlier masked chain they depend on still exists.

In this example:

- `M2[1]` is valid because its predecessor chain through `M1[0]` survives
- `M2[2]` is valid because its predecessor chain through `M1[1]` survives
- `M2[3]` is not valid because `M1[2]` was dropped

### Step 4: keep some rows from `M2`

For `M2`, keep ratio is:

```text
max(0.5^2, 0.2) = 0.25
```

So we keep about one row, but only from the valid candidates `[1, 2]`.

Suppose the kept row is:

```text
M2[1]
```

### Final kept rows

```text
R[0], R[1], R[2], R[3], M1[0], M1[1], M2[1]
```

The exact compacted tensors from the current implementation for this toy example are:

```text
input_ids_cod    = [101, 201, 202, 203, 999, 999, 999]
position_ids_cod = [0,   1,   2,   3,   0,   1,   1]
```

So the last three compacted rows are:

- `M1[0]`
- `M1[1]`
- `M2[1]`

COD then compacts:

- `input_ids`
- `position_ids`
- `labels`
- the query axis of the attention mask
- the key axis of the attention mask

The important point is that COD is not naive random dropping. It only keeps later-block rows whose required masked ancestry is still present.

## 13. Training Summary

The clean mental model for training is:

1. start with the real prompt+completion sequence
2. add `draft_len - 1` mask-token blocks
3. repeat original positions across blocks
4. build shifted labels across blocks
5. build a 4D attention mask that defines what each row is allowed to see
6. optionally apply COD to keep only a valid subset of deeper rows

This training layout is a packed construction used only for optimization and supervision. It is not the same as the inference-time input format.

## 14. Inference: The Important Simplification

Inference is much simpler than training.

There is:

- no repeated position IDs
- no packed `R | M1 | M2` training layout
- no custom 4D mask

Inference uses one normal causal sequence.

If the current real prefix is:

```text
[x0, x1, x2]
```

and `draft_len = 3`, the draft model input is:

```text
[x0, x1, x2, M, M]
```

with ordinary monotonic positions:

```text
[0, 1, 2, 3, 4]
```

and the ordinary causal attention mask of the base transformer.

## 15. How One PARD Draft Step Produces Multiple Tokens

This is the main inference idea.

A causal LM returns logits for every input position in one forward pass.

So for:

```text
[x0, x1, x2, M, M]
```

one forward pass gives logits for:

- row `x0`
- row `x1`
- row `x2`
- row `M` at position 3
- row `M` at position 4

We only care about the last `draft_len` rows:

- logits at the last real prefix row predict draft token 1
- logits at the first appended `M` row predict draft token 2
- logits at the second appended `M` row predict draft token 3

So yes, PARD still predicts three draft tokens, but it does so in parallel from one forward pass.

That is exactly what `propose_parallel_draft_tokens(...)` and `_run_hf_pard_speculative_decode(...)` are doing in `methods/parallel_draft_models/inference/infer.py`.

## 16. Why the Two `M` Tokens Are Different at Inference

Even though the two suffix tokens are both `pard_token`, they occupy different positions in the causal sequence.

For:

```text
[x0, x1, x2, M, M]
```

the first `M` row can attend to:

```text
x0, x1, x2, M(position 3)
```

the second `M` row can attend to:

```text
x0, x1, x2, M(position 3), M(position 4)
```

So they produce different hidden states and different logits.

The model learned during training that these suffix-mask positions correspond to different future-token roles.

## 17. Non-vLLM Inference Implementation in This Repo

The main cached path is `_run_hf_pard_speculative_decode(...)` in `methods/parallel_draft_models/inference/infer.py`.

One speculation step does:

1. decide how many draft tokens to request: `requested`
2. append `requested - 1` copies of `pard_token` to the current pending real tokens
3. run one draft forward with cache
4. take `argmax` on the last `requested` logits rows to get the draft proposals
5. append those draft proposals to the target pending tokens
6. run one target forward with cache
7. compare target predictions against the draft proposals
8. accept the longest matching prefix
9. emit the accepted tokens plus one target token

In code, the draft proposals are read from:

```python
draft_output.logits[0, -requested:, :]
```

That is the implementation detail that makes the "three predictions from one forward pass" concrete.

## 18. Why the Cache Drops Mask Tokens After Each Step

After the draft forward, the draft KV cache includes entries for the appended mask tokens.

Those cache entries must not survive into the next real step, because `pard_token` is not an actual generated token. It was only a temporary placeholder used to produce parallel draft logits.

So the implementation explicitly does:

```python
draft_cache_length -= mask_count
```

and then sets the next `draft_pending` to the real emitted tokens.

This is an important detail. Without it, the draft model would carry fake mask-token state across speculation steps.

## 19. Fallback Non-Cached Path

The repo also has a slower fallback path in `methods/parallel_draft_models/inference/infer.py`.

That path simply does:

1. build `[prefix..., M, M, ...]`
2. run a no-cache forward
3. read the last `draft_len` logits rows
4. verify with the target

It is useful for tests and correctness checks. The cached path is the one intended for actual benchmarking.

## 20. vLLM Inference

The vLLM wrapper is in `methods/parallel_draft_models/inference/infer_vllm.py`.

Two details matter:

1. the checkpoint config must advertise itself as a PARD-style draft model
2. vLLM must be launched with `parallel_drafting=True`

The training export writes:

- `pard_token`
- `spd_type = "pard"`

into the Hugging Face checkpoint config.

The vLLM speculative config is then built as:

```python
{
    "method": "draft_model",
    "model": ...,
    "num_speculative_tokens": args.draft_len,
    "parallel_drafting": True,
    ...
}
```

Why is the method still `"draft_model"`?

Because from vLLM's perspective PARD is still a draft-model speculative-decoding method. The difference is that the draft model is marked as parallel-drafting capable.

The wrapper also fixes:

```text
gpu_memory_utilization = 0.85
```

for both baseline and speculative runs, so the comparison is made under the same memory-utilization setting.

## 21. What Gets Saved in the Checkpoint

The training script exports a normal Hugging Face checkpoint.

In addition to the model weights, it saves:

- tokenizer files
- generation config when available
- `config.json` with `pard_token`
- `config.json` with `spd_type = "pard"`
- `pard_metadata.json`
- `training_summary.json` when present

That is why the same checkpoint can be consumed by both:

- the local Hugging Face inference path
- the vLLM parallel-drafting path

## 22. PARD vs Normal `draft_model`: Theory and Practical Tradeoffs

### Theoretical difference

Normal `draft_model`:

- draft side is autoregressive
- `K` draft tokens need `K` draft decode steps

PARD:

- draft side is parallelized with masked suffix placeholders
- `K` draft tokens come from one draft forward

### Practical inference tradeoff

Normal `draft_model` often has higher acceptance because it is predicting the exact next-token chain autoregressively.

PARD may accept fewer tokens at the same `draft_len`, but it can still be faster because the draft-side latency is much lower.

That is why PARD can outperform a more accurate ordinary draft model.

### Training tradeoff

Normal `draft_model` training is simpler:

- plain causal next-token distillation on completion tokens

PARD training is heavier:

- repeated blocks
- repeated positions
- custom 4D attention mask
- COD to control token inflation

### Memory tradeoff

Inference parameter memory is similar if both methods use the same draft-model size.

PARD adds:

- one special `pard_token`
- temporary mask-token KV slots during each draft step

Those mask-token KV slots are transient and are explicitly discarded after each speculation step.

Training memory is more expensive for PARD because the packed layout increases sequence length before COD compaction.

## 23. The Most Important Things to Keep Straight

If the implementation still feels confusing, keep these distinctions clear:

### Training vs inference

Training:

- packed multi-block layout
- repeated position IDs
- custom 4D attention mask
- COD may drop rows

Inference:

- one normal causal sequence
- normal monotonic positions
- normal causal mask
- append `pard_token` only temporarily

### `pard_token`

There is only one special mask token.

Its meaning comes from:

- position
- allowed context
- the training objective

not from having multiple different mask-token IDs.

### Raw labels vs actual supervision

Do not interpret the raw flattened `labels` tensor without accounting for the built-in one-token causal shift.

The right way to reason about the training layout is:

`input row` -> `next label after shift` -> `allowed attention for that input row`

### COD

COD is not "drop random later rows".

It is "drop later rows while preserving valid masked dependency chains."

## 24. Short Mental Model

The shortest correct mental model is:

- PARD keeps the usual speculative-decoding verify loop.
- It changes only the draft side.
- Training teaches a normal causal LM to interpret a masked suffix built from one learned `pard_token`.
- At inference, the draft model reads `[prefix..., M, M, ...]` once and produces multiple future-token logits in parallel.
- The target verifies those candidates in the usual speculative-decoding way.

That is why PARD is best viewed as "parallelized draft-model speculation", not as a completely different family of method.

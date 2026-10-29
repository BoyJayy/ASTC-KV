# First model, authentic tensors, and same-device baselines

Source check: 2026-10-28. The code and output JSON are authoritative about what
has actually executed. The selected model is a numerical pilot, not evidence
that the representation will generalize to larger or different models.

## Selected checkpoint

**HuggingFaceTB/SmolLM2-135M**, immutable Hugging Face revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.

| Property | Checkpoint value |
|---|---:|
| Architecture | LlamaForCausalLM, ordinary causal attention |
| Layers | 30 |
| Hidden dimension | 576 |
| Query heads | 9 |
| KV heads | 3 |
| Head dimension | 64 |
| Maximum positions | 8192 |
| RoPE theta | 100000 |
| RoPE convention | Non-interleaved half rotation |
| RMSNorm epsilon | 0.00001 |
| MLP | SiLU gate, intermediate dimension 1536 |
| Embedding/output weights | Tied |
| Stored weights | BF16; no additional quantization |
| Tokenizer | Checkpoint GPT2-compatible tokenizer, 49152 vocabulary |

The small checkpoint makes repeated CPU quality experiments practical. It has
real learned weights, RoPE and grouped-query attention, while its implementation
is short enough to independently inspect. It should not be used to infer the
fraction of runtime spent on attention for the user's eventual target model.

Primary sources:

- [Pinned configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/config.json)
- [Model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- [Original Llama implementation, Transformers 4.40.1](https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/llama/modeling_llama.py)

## Capture contract

`scripts/capture_qkv.py` performs a full causal forward pass using NumPy FP32.
BF16 checkpoint values are converted exactly to FP32 before arithmetic. This is
a transparent numerical reference; it is not a Metal performance baseline.

For each selected layer the NPZ stores:

| Key | Shape / meaning |
|---|---|
| `q` | `[L, Hq, T, D]`, after RoPE |
| `k` | `[L, Hkv, T, D]`, after RoPE, before head replication |
| `v` | `[L, Hkv, T, D]`, projected values before head replication |
| `layer_ids` | Actual layer indices, not assumed contiguous |
| `token_ids`, `positions` | Input IDs and positions starting from zero |
| `final_hidden` | Final normalized hidden states `[T, hidden]` |
| `nll` | Next-token losses `[T-1]`; logits at t predict token t+1 |
| `last_logits` | Last up to eight positions, full vocabulary |
| `logits` | Optional full logits, enabled by `--save-logits` |
| `q_pre_rope`, `k_pre_rope` | Optional diagnostics, enabled by `--include-pre-rope` |

The JSON sidecar records the full config, checkpoint revision, hashes of weights,
config/tokenizer/capture, text and tokenization policy, arithmetic precision,
software versions, split and numerical baseline NLL. No implicit BOS token or
chat template is added. The baseline perplexity on local fixtures is a diagnostic,
not a benchmark result representative of a corpus.

```bash
OPENBLAS_NUM_THREADS=4 python scripts/capture_qkv.py \
  --model-dir third_party/model \
  --text-file data/prompts/systems.txt \
  --out results/captures/systems.npz \
  --max-tokens 384 --layers all --split tune \
  --include-pre-rope --validate-transformers
```

`tokenizers` is the only package needed beyond NumPy for text capture. Supplying
`--token-ids-file` (a JSON integer array) removes that requirement. The independent
comparison additionally requires CPU PyTorch and Transformers.

### Reference checks

The optional Transformers check uses a separately executed eager FP32 model
loaded from the same local checkpoint. It intercepts the actual Hugging Face
RoPE function outputs and V projection outputs; these are compared with NumPy
for all 30 layers. It also compares final hidden states and full vocabulary
logits. Thresholds are fixed in the script before running: max absolute error
below 0.001 and relative L2 error below 0.0001 for each compared quantity. A
failure aborts capture rather than being silently accepted.

Executed on 2026-10-28: the independent comparison passed for the `systems`
prompt at both 32 and 384 tokens, using PyTorch 2.6.0+cpu and Transformers
4.51.3. At 384 tokens, maximum absolute errors were Q `1.113e-4`, K `7.343e-5`,
V `1.556e-5`, final hidden `1.478e-4`, logits `1.907e-4`. Relative L2 errors
were below `1.2e-6`. Raw results are in
`results/transformers_validation.json`. This is independent evidence for the
numerical capture implementation, not evidence of compression quality.

`validate_incremental` compares a full causal forward against uncompressed
prefill of T-1 tokens followed by one incremental step. The same thresholds
apply. Neither check establishes model-quality preservation under compression.
The 384-token check also passed, exercising incremental RoPE at position 383;
its largest absolute logit error was `2.480e-5`.

## Incremental quality reference with error feedback

```python
from scripts.capture_qkv import NumpyLlama

model = NumpyLlama("third_party/model")
next_logits, caches = model.prefill(prefix_ids)
# caches is a list of 30 pairs (K, V), each [3, prefix_length, 64].
# Encode/reconstruct only completed historical pages here, using past data.
next_logits, caches = model.step(next_token, len(prefix_ids), caches)
```

`step` computes new K/V from the actual current hidden state and corrupted
history. It therefore includes feedback through subsequent layers and tokens.
For teacher forcing, the initial prefill logits predict the first continuation
token; after consuming that token, the next step's logits predict the second.
For autoregressive generation, choose each token from the current run's own
logits. Baseline tokens must not be imposed on the compressed run.

The old cache list is not mutated. The returned list includes appended K/V.
An optional `kv_transform(layer, q, k, v)` callback can replace a layer's cache.
The caller is responsible for a causal page-completion policy. Applying a
full-sequence codec that derives early metadata from future tokens is unsuitable
for a causal model-quality claim. A fixed prefill followed by compression of
only its completed pages avoids this ambiguity for the initial continuation
experiment.

This CPU reference keeps reconstructed K/V arrays. It evaluates numerical
effects only; it is not the compressed active-cache runtime requested as the
eventual system result.

## Available MLX baselines

On the source-check date, official GitHub releases were MLX `v0.32.4`
(2026-10-14) and MLX-LM `v0.32.2` (2026-10-16). The inspected main commits were:

- MLX: `7d2d94a5561ebc3965884a0fc3dfc35b7b1bb17a`.
- MLX-LM: `1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7`.

Release versions and main commits are different identifiers. Do not claim that
a package installed from a release is the inspected main commit; record the
actual installed version and source revision at execution time.

| Comparison | Actual implementation to run | Scope |
|---|---|---|
| FP16/BF16 KV | `KVCache` + `mx.fast.scaled_dot_product_attention` | Primary optimized native baseline |
| Quantized KV | `QuantizedKVCache(group_size=32/64, bits=4/8)` + MLX-LM quantized attention | Available native low-bit baseline, benchmark on the same Mac |
| KIVI/TurboQuant | Author implementation or faithful numerical reproduction | Quality orientation; no imported non-Metal latency |
| ASTC decoded reference | Ordinary attention on exactly decoded values | Isolates codec error from new kernel error |
| ASTC direct texture path | Project Metal kernel | Must compare against the native baselines above |

In the inspected MLX-LM implementation, the quantized cache stores packed
integers and per-group affine metadata; new K/V are quantized when appended.
The attention path uses quantized matrix multiplication for both QK and PV,
with a separate precise softmax and intermediate attention scores. It does not
explicitly expand the entire cache into ordinary floating-point K/V in Python.
This is an existing usable path, but its throughput is not proven optimal.
Before a strong speed claim, compare other maintained same-device alternatives
if this baseline is found to be materially underperforming.

The 4-bit setting is payload precision, not total memory. For a full group of
64 values and FP16 scale plus bias, affine metadata adds 0.5 bits/value before
allocation rounding and cache capacity. The corresponding idealized total is
4.5 bits/value. At short sequences, allocated capacity can be substantially
larger than the logical active payload.

Primary sources:

- [MLX release](https://github.com/ml-explore/mlx/releases/tag/v0.32.4)
- [MLX-LM release](https://github.com/ml-explore/mlx-lm/releases/tag/v0.32.2)
- [Pinned MLX-LM base attention](https://github.com/ml-explore/mlx-lm/blob/1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7/mlx_lm/models/base.py)
- [Pinned MLX-LM cache](https://github.com/ml-explore/mlx-lm/blob/1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7/mlx_lm/models/cache.py)
- [Pinned Llama integration](https://github.com/ml-explore/mlx-lm/blob/1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7/mlx_lm/models/llama.py)

## Mac baseline probe

```bash
python scripts/probe_mlx.py --out results/mac_environment.json
python scripts/probe_mlx.py \
  --capture results/captures/systems.npz --layer 17 \
  --warmup 10 --repeats 50 --out results/mlx_attention_baseline.json
```

The probe records macOS, Xcode/SDK, power state, installed versions and device
information. With a capture it benchmarks one-token attention over genuine
captured K/V. It randomizes the comparison order, retains all raw latencies and
reports bootstrap intervals. It includes 4/8-bit groups of 32/64 where valid.

These are isolated attention host latencies including dispatch and a completion
wait, with pre-existing caches. Initial quantization and dynamic append costs are
excluded and clearly labeled. All cache variants coexist during interleaved
trials, so this run cannot provide a per-variant process peak. Whole-generation
tests must independently include cache maintenance, prefill, time to first token,
decode and full request time. The script emits an environment-only record on
Linux instead of pretending Metal was tested.

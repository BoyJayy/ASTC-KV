# ASTC-KV: protocol v0.1 (2026-10-28)

## Scope and decision rules

The first executed experiment is a **numerical pilot**, not an inference speed
benchmark. It uses real Q/K after RoPE and real V from pinned SmolLM2-135M.
The original BF16 weights are widened exactly to FP32. All CPU variants use
the same weights, token IDs, positions, GQA mapping, and causal mask.
The configuration `configs/pilot_v1.json` is written before codec results.
No world-first, hardware speed, or full model-quality claim follows from a
passing snapshot screen.

Only completed 96-token pages from positions [0,288) are compressed. Queries
are from positions 288..383; the fresh tail is kept in FP16 for cache accounting
(FP32 in diagnostic arithmetic, with a separate FP16 rounding control).
Statistics cannot see the suffix, held-out query covariance, or test answers.
Token permutation controls stay inside pages and are inverted before attention.
Channel permutations preserve exact values and are likewise inverted.
These interventions distinguish placement from a marginal histogram effect;
they alone do not establish the mechanism or a general predictor.

Pilot numerical screen: aggregate attention relative L2 <=10%, mean KL <=0.01
nats, and no more logical bytes than the scalar affine u4 baseline under the
same K-only, V-only, or K+V policy. Record every failed mode. Passing this screen
only selects candidates for model and hardware evaluation. Provisional final
quality: PPL increase <=2%, retrieval/multi-fragment accuracy decrease <=2
percentage points, with uncertainty and autoregressive validation. Freeze
representative final datasets and tolerances before confirmation; four authored
pilot prompts are not sufficient to establish these criteria.

## Memory

Count original KV scalars, not texture components. An ASTC block is 16 bytes;
actual edge padding is charged. Metadata is physically rounded to FP16 *before*
normalization and reconstruction. Charge metadata, maps and fresh tail. This
initial contiguous uniform-page representation computes all page addresses
arithmetically and has no per-page descriptor or index table. The Metal probe
reports its fixed dispatch parameters separately; resource allocation/argument
alignment remains unknown until measured. A future mixed/paged format must
include its actual descriptors and indices. K-only/V-only use FP16 for the other
tensor. File headers and diagnostic JSON are separate from cache payload.
The NumPy prototype retains originals and decoded copies: its process memory is
not a compressed resident cache. Persistent resource bytes, peak codec workspace,
blit staging, attention intermediates, and whole-process peaks require separate
measurements; no resident-memory saving is inferred from compressed file size.

## Mac baseline hierarchy

1. MLX `mx.fast.scaled_dot_product_attention` with FP16/BF16 cache.
2. Real MLX-LM `QuantizedKVCache` and quantized attention (4/8 bits, group64).
3. llama.cpp Metal flash-attention with q4_0/q8_0 KV, if supported for this
   exact model/shape. This is essential because MLX's quantized path is composed
   from two quantized matmuls and a softmax, not the FP16 fused SDPA path.
4. ASTC precompressed attention; ordinary attention on software-restored ASTC;
   ordinary attention on GPU-restored ASTC; same-layout uncompressed textures.
5. Once implemented, ASTC with complete encode/blit/tail/update cost.

Within each runtime use identical unquantized or identically quantized weights;
verify logits after any conversion. Report cross-runtime results separately.
KIVI/TurboQuant are numerical research references unless a comparable Metal
implementation is executed. CPU u4 in this pilot is **not** MLX, KIVI or
TurboQuant and supplies no hardware timing evidence.

## Performance protocol (pending Mac)

Record device name, registry ID, memory/unified status, macOS build, Xcode,
SDK/Metal language, runtime/package versions and revisions, model/weight hash,
project source hash/commit, dimensions, mask, prompt/continuation lengths,
batch, page/tail policy, power source, low-power mode, and thermal state.
Warm up 10 runs after compilation. Use at least 30 paired repetitions with
randomized/alternating variant order; save raw samples. Separate compilation,
prefill, TTFT, decode token latency, total request wall time and GPU time.
Report paired bootstrap intervals over requests; correlated tokens are not
independent replicates. Compare long contexts 512/2048/4096/8192 within the
model's 8192 positional limit for prompt plus continuation combined, plus
shape-tail/boundary cases. Do not use an 8192-token prompt and then decode
beyond the configured positional limit.

Use command-buffer GPU timestamps/counters in separate diagnostic mode, and
end-to-end wall time with synchronization at request boundaries. Do not add
per-small-kernel fences in production timing. Do not add durations of overlapped
events. Account for warm/cold resource/cache conditions and working sets larger
than observed GPU caches; report when no such workload was feasible.

Probe attention fraction f of baseline decode and compare observed end-to-end
speed to Amdahl's relation 1/((1-f)+f/s) only as a simplified model. For dynamic
pages, estimate gain as future-read count times per-read savings less exposed
encode/upload/sync cost. Validate this model on held-out workloads. No scheduling
policy may use future output length; evaluate an online policy against an oracle
only as a labeled diagnostic.

## Stage ledger

A: Linux environment inspected; Mac baseline pending. B: CPU numerical pipeline
implemented/execution recorded in results. C: limited preregistered layout sweep.
D: Metal source prototype prepared; compilation, correctness and timing on Mac
pending. E/F/G: GPU encoder, runtime integration and confirmatory study planned.

After each run record hypothesis, exact configuration/source hash, measurements,
confirmed/refuted scope, alternatives, and the experiment that separates them.

# ASTC-KV representation v0.1

Status: numerical implementation executed on Linux; Metal reader source prepared,
not compiled or executed on Apple hardware. This is a uniform-layout research
interchange, not yet a dynamic resident-cache backend.

## Tensor semantics

K and V have shape `[Hkv,T,D]` before GQA replication. K is after Llama RoPE.
Queries are `[Hq,N,D]` with absolute positions; query head `hq` maps to KV head
`floor(hq/(Hq/Hkv))`. Causal mask admits only keys at positions <= query position.
The first model has Hq=9, Hkv=3, D=64. Runtime metadata never depends on future
queries or target answers.

## Component and address mapping

Each KV head owns one 2D texture slice. Texture row is token index. In scalar
layout, `x=channel` and scalar value is replicated into RGB, alpha=1; decoding
reads R. In RGBA layout, `x=floor(channel/4)` and lane=`channel mod 4`.
Both layouts use linear LDR, never sRGB. Scalar HDR block-min is an offline
control only. Hardware uses integer-coordinate `read`, avoiding interpolation
between independently stored scalars. ASTC's own interpolation inside a block
is part of the codec, not a texture filtering operation.

ASTC blocks contain 128 bits at footprints 4x4,6x6,8x8. Width is rounded up to
the block width: for scalar D64/block6, actual width=66. Right edge duplicates
the last logical channel. Height is rounded to page extent for offline exports.
Page height96 aligns to each selected block height. The full-page numerical
encoder rejects incomplete pages, which belong in the fresh tail. No inverse
index table is needed for the normal contiguous representation. Optional
permutations are diagnostic controls and their fixed uint16 maps are charged.

## LDR range transformations

Let a source group have minimum m and maximum M. Store offset
`o=floor_to_fp16(m)` and scale `s=ceil_to_fp16(max(M-o,2^-24))`; normalise using
the **stored** o and s: `u=(x-o)/s`. The bounds fit [0,1]. Decode `xhat=o+s*uhat`
using FP32 arithmetic. This prevents hidden FP32 metadata precision. FP16
overflow fails explicitly. Endpoints/codes inside ASTC remain lossy.

Two supported metadata groupings:

| Normalisation | Metadata arrays for each K or V | FP16 metadata cost, D64/P96 |
|---|---|---:|
| page_channel | offset/scale `[Hkv,ceil(T/96),D]` | 32/96 = 0.3333 bits/scalar |
| token | offset/scale `[Hkv,T,1]` | 32/64 = 0.5 bits/scalar |

Full pages are encoded independently of future tokens. In v0.1 all heads/layers
within one variant share layout and mode. Future mixed policies must explicitly
store/count mode decisions and any pointer tables.

## HDR diagnostic

For scalar6x6 HDR RGB / LDR alpha, store one FP16 downward-rounded minimum per
6x6 block; encode `x-o >=0` replicated RGB with alpha1. Restore by adding o.
This is inspired by Apple's published weight transformation, not a claimed new
method and not a reproduction of Apple's proprietary encoder/adapters. At D64
the 66-wide padded texture costs `(128+16)*11/(6*64)=4.125 bits/scalar`.
Metal v0.1 does not implement this HDR/block-min metadata path.

## Byte accounting

For scalar6x6 and completed P96 pages with D64:

* ASTC payload including right edge: `128*11/(6*64)=3.6667 bits/scalar`.
* Page/channel affine: payload +32/96 =4.0000 bits/scalar.
* Per-token affine: payload +32/64 =4.1667 bits/scalar.
* Conventional u4+FP16 scale+bias/group64: 4+32/64 =4.5 bits/scalar.

These counts are logical compressed payload+metadata, not physical texture
allocation or whole-process peak. `.astc` has a16-byte transport header per head
which is not uploaded to the texture. Manifest JSON is host-side. The Metal
probe's fixed parameters are60 bytes per dispatch and are reported separately;
actual allocation granularity/driver bookkeeping is device-dependent and must
be measured. Offset/scale tensors are FP16 buffers; descriptor arrays are
implicit for this regular layout. Staging pitch is padded separately and may
exceed payload pitch. The probe reports both.

For prefix288 plus FP16 tail96, V-only scalar6x6/token + FP16 K costs11.5625
bits per original combined K/V scalar. The same policy with u4 V costs11.6875.
Without the tail, those pairs cost10.0833 and10.25 respectively. Calling the
whole V-only cache "4-bit" would be incorrect.

## Update and resource policy

Planned active path: append exact fresh K/V -> when a page completes, encode on
GPU into ordinary bytes -> blit valid compressed blocks into a private ASTC
texture -> make the new page visible through correct command dependencies ->
reclaim the old tail page after readers finish. Encode, blit and commit are
distinct operations. The current CPU numerical experiment pre-encodes an old
prefix and holds the suffix unchanged; dynamic GPU encoding/upload is pending.

Metal v0.1 handles one current query and fully precompressed histories, including
mixed FP16 K plus ASTC V. It has no dynamic tail or page replacement. Its diagnostic
CPU and GPU decoded copies must not be counted as a production compressed cache.

## Interchange

See `metal/manifest.example.json` and `metal/manifest_v_only.example.json`.
`scripts/export_metal_fixture.py` exports real captured Q/K/V, encoded blocks,
exact FP16 metadata, and software reconstruction files for hardware equivalence
checks. The payload is LDR; profile is not inferred from arbitrary external
ASTC bytes. This importer accepts only our versioned fixtures.

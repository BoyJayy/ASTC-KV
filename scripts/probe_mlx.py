#!/usr/bin/env python3
"""Record Mac environment and optionally measure official MLX KV attention paths.

This measures isolated pre-existing-cache attention with host dispatch and a
completion wait per trial. It does not measure dynamic cache maintenance or
whole-model generation. Run generation separately before making system claims.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time


def shell_output(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--capture", type=Path)
    p.add_argument("--layer", type=int, default=0, help="Actual layer ID in the capture")
    p.add_argument("--length", type=int, default=0, help="0 means use all captured tokens")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--seed", type=int, default=2718)
    args = p.parse_args()
    if args.out.exists():
        raise RuntimeError("Preserve prior results; choose a new --out path")
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    record = {
        "schema": "astckv.mlx_attention_baseline.v1",
        "platform": platform.platform(), "machine": platform.machine(),
        "python": platform.python_version(),
        "versions": {n: package_version(n) for n in ["mlx", "mlx-lm", "numpy"]},
        "git_commit": shell_output(["git", "rev-parse", "HEAD"]),
        "macos": shell_output(["sw_vers"]), "xcode": shell_output(["xcodebuild", "-version"]),
        "metal_sdk": shell_output(["xcrun", "--sdk", "macosx", "--show-sdk-version"]),
        "power": shell_output(["pmset", "-g", "batt"]),
        "status": "environment_only", "metal_performance_verified": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if platform.system() != "Darwin":
        record["reason"] = "Not running on macOS; Metal and MLX GPU tests were not executed."
    else:
        import mlx.core as mx
        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal device is not available; cannot run the GPU baseline")
        mx.set_default_device(mx.gpu)
        record["mlx_default_device"] = str(mx.default_device())
        record["metal_device_info"] = mx.metal.device_info()
        if args.capture:
            import numpy as np
            from mlx_lm.models.base import quantized_scaled_dot_product_attention
            from mlx_lm.models.cache import QuantizedKVCache
            data = np.load(args.capture, allow_pickle=False)
            actual_layers = data["layer_ids"].tolist()
            layer_index = actual_layers.index(args.layer)
            total = data["k"].shape[-2]
            n = args.length or total
            if not 1 <= n <= total:
                raise ValueError("Requested length is outside capture")
            qnp = data["q"][layer_index, :, n-1:n, :]
            knp = data["k"][layer_index, :, :n, :]
            vnp = data["v"][layer_index, :, :n, :]
            q = mx.array(qnp[None], dtype=mx.float16)
            k = mx.array(knp[None], dtype=mx.float16)
            v = mx.array(vnp[None], dtype=mx.float16)
            mx.eval(q, k, v)
            scale = q.shape[-1]**-0.5
            operations = {"fp16_fast_sdpa": lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)}
            reference_values = {"fp16_fast_sdpa": (np.array(k),np.array(v))}
            metadata = {"fp16_fast_sdpa": {"kv_logical_bytes": k.nbytes + v.nbytes, "kv_bits": 16, "group_size": None}}
            caches = []
            for bits in (4, 8):
                for group in (32, 64):
                    if q.shape[-1] % group:
                        continue
                    name = f"mlx_q{bits}_g{group}"
                    cache = QuantizedKVCache(group_size=group, bits=bits)
                    qk, qv = cache.update_and_fetch(k, v)
                    mx.eval(*qk, *qv)
                    caches.append(cache)
                    def operation(qk=qk, qv=qv, bits=bits, group=group):
                        # MLX-LM 0.32.2 does queries *= scale in place. Give it a
                        # fresh array expression so warmup/trials cannot rescale
                        # the shared query used by every baseline.
                        return quantized_scaled_dot_product_attention(q+0, qk, qv, scale=scale, mask=None, group_size=group, bits=bits)
                    operations[name] = operation
                    # Diagnostic materialization happens only outside timed ops.
                    dk=mx.dequantize(*qk,group_size=group,bits=bits)
                    dv=mx.dequantize(*qv,group_size=group,bits=bits)
                    mx.eval(dk,dv)
                    reference_values[name]=(np.array(dk),np.array(dv))
                    payload = sum(a.nbytes for a in (*qk, *qv))
                    metadata[name] = {"kv_logical_bytes": payload, "allocated_cache_array_bytes": cache.nbytes,
                                      "effective_bits_per_kv_scalar": payload * 8 / (knp.size + vnp.size),
                                      "kv_bits_payload": bits, "group_size": group}
            # Validate each path on its own reconstructed inputs. FP16 arithmetic
            # needs a distinct tolerance from the FP32 Metal research kernel.
            checks={}
            q64=np.array(q).astype(np.float64)
            for name,operation in operations.items():
                kr,vr=reference_values[name]
                mapping=np.arange(q.shape[1])//(q.shape[1]//k.shape[1])
                scores=np.matmul(q64,kr[:,mapping].astype(np.float64).swapaxes(-1,-2))*scale
                probs=np.exp(scores-scores.max(-1,keepdims=True));probs/=probs.sum(-1,keepdims=True)
                reference=np.matmul(probs,vr[:,mapping].astype(np.float64))
                got=operation();mx.eval(got);actual=np.array(got).astype(np.float64)
                repeated=operation();mx.eval(repeated)
                np.testing.assert_array_equal(np.array(q),q64,err_msg='Operation mutated shared query')
                np.testing.assert_array_equal(np.array(repeated),actual,err_msg='Repeated operation changed output')
                finite=bool(np.isfinite(actual).all())
                rel=float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference),1e-30))
                checks[name]={"relative_l2_vs_own_fp64_reference":rel,"finite":finite,"threshold":0.005,"passed":finite and rel<=0.005}
            record['correctness']=checks
            if not all(c['passed'] for c in checks.values()):
                args.out.parent.mkdir(parents=True,exist_ok=True)
                record['status']='correctness_failed'
                args.out.write_text(json.dumps(record,indent=2,default=str)+'\n')
                raise RuntimeError('MLX attention failed pre-timing correctness gate')
            for operation in operations.values():
                for _ in range(args.warmup):
                    mx.eval(operation())
            # Each trial randomizes mode order; all durations are retained.
            rng = random.Random(args.seed)
            raw = []
            outputs = {}
            for trial in range(args.repeats):
                names = list(operations)
                rng.shuffle(names)
                for order, name in enumerate(names):
                    started = time.perf_counter_ns()
                    output = operations[name]()
                    mx.eval(output)
                    elapsed = time.perf_counter_ns() - started
                    raw.append({"trial": trial, "order": order, "mode": name, "host_latency_ns": elapsed})
                    outputs[name] = np.array(output)
            baseline = outputs["fp16_fast_sdpa"].astype(np.float64)
            summaries = {}
            for name in operations:
                samples = np.asarray([r["host_latency_ns"] for r in raw if r["mode"] == name], dtype=np.float64)
                boot_rng = np.random.default_rng(args.seed)
                medians = np.median(boot_rng.choice(samples, size=(2000, len(samples)), replace=True), axis=1)
                delta = outputs[name].astype(np.float64) - baseline
                summaries[name] = {**metadata[name], "median_host_latency_us": float(np.median(samples)/1000),
                                   "median_bootstrap_ci95_us": (np.percentile(medians, [2.5,97.5])/1000).tolist(),
                                   "attention_relative_l2_vs_fp16": float(np.linalg.norm(delta)/max(np.linalg.norm(baseline),1e-30))}
            record.update(status="isolated_attention_executed", metal_performance_verified=True,
                          capture=str(args.capture), layer=args.layer,
                          capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
                          query_dtype="float16", input_KV_dtype="float16 before cache quantization",
                          query_isolation="fresh q+0 expression per quantized operation; included in host timing; prevents MLX-LM in-place query scaling",
                          shape={"batch":1,"query_heads":q.shape[1],"kv_heads":k.shape[1],"query_tokens":1,"kv_tokens":n,"head_dim":q.shape[-1]},
                          warmup=args.warmup, repeats=args.repeats, seed=args.seed, raw_timings=raw, summaries=summaries,
                          timing_scope="isolated cached attention, including host dispatch + mx.eval completion; not pure GPU kernel time",
                          excluded=["weight inference", "initial cache quantization", "cache appends", "prefill", "model generation"],
                          memory_caveat="All modes coexist for interleaved trials; process/allocator peak is not a per-mode peak.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(json.dumps({"status": record["status"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()

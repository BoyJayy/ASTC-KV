#!/usr/bin/env python3
"""Causal CPU quality experiment with historical KV compression and error feedback.

Completed prefix pages are compressed once, before continuation. New KV entries
are FP16. CPU caches store reconstructions, so memory/time here are never GPU
performance evidence. The seven numerical methods are frozen before execution.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
from astc_kv.codec import encode_tensor, affine_quantize
from capture_qkv import NumpyLlama, MODEL_ID, MODEL_REVISION, sha256

METHODS = ["fp16", "u4_token_V_only", "u4_token_KV", "scalar6_page_V_only", "scalar6_page_KV",
           "scalar6_token_V_only", "scalar6_token_KV"]
GENERATION_METHODS = ["fp16", "scalar6_page_V_only", "scalar6_token_V_only", "u4_token_V_only"]
ASTC_CONFIG = {"name": "scalar6_page", "layout": "scalar", "block": 6,
               "normalization": "page_channel", "profile": "ldr"}
ASTC_TOKEN_CONFIG = {"name": "scalar6_token", "layout": "scalar", "block": 6,
                     "normalization": "token", "profile": "ldr"}


def as_fp16(x):
    return np.asarray(x, np.float16).astype(np.float32)


def tail_fp16(_layer, _q, k, v):
    """Cast only freshly appended KV; preserve reconstructed historical prefix."""
    k[:, -1:, :] = as_fp16(k[:, -1:, :])
    v[:, -1:, :] = as_fp16(v[:, -1:, :])
    return k, v


def next_token_loss(logits: np.ndarray, target: int) -> float:
    z = np.asarray(logits, np.float64)
    peak = z.max()
    return float(peak + np.log(np.exp(z - peak).sum()) - z[target])


def decode_text(model_dir: Path, ids):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    return tokenizer.decode([int(t) for t in ids], skip_special_tokens=False)


def prepare_variants(original, prefix_tokens, page_tokens, quality, log):
    """One codec execution per layer/tensor; share immutable reconstructions."""
    prepared = {name: [] for name in METHODS}
    byte_counts = {name: {"compressed_prefix_payload_and_metadata": 0, "uncompressed_prefix_fp16": 0} for name in METHODS}
    per_layer = []
    for layer, (k, v) in enumerate(original):
        if k.shape[1] != prefix_tokens or v.shape[1] != prefix_tokens:
            raise ValueError("Prefill cache length does not match frozen prefix")
        kf, vf = as_fp16(k), as_fp16(v)
        ku, ku_counts = affine_quantize(k, bits=4, axis="token", group=64)
        vu, vu_counts = affine_quantize(v, bits=4, axis="token", group=64)
        ka = encode_tensor(k, ASTC_CONFIG, page_tokens=page_tokens, quality=quality)
        va = encode_tensor(v, ASTC_CONFIG, page_tokens=page_tokens, quality=quality)
        kt = encode_tensor(k, ASTC_TOKEN_CONFIG, page_tokens=page_tokens, quality=quality)
        vt = encode_tensor(v, ASTC_TOKEN_CONFIG, page_tokens=page_tokens, quality=quality)
        byte_kf, byte_vf = k.size * 2, v.size * 2
        byte_ku, byte_vu = sum(ku_counts.values()), sum(vu_counts.values())
        byte_ka, byte_va = ka.logical_bytes, va.logical_bytes
        byte_kt, byte_vt = kt.logical_bytes, vt.logical_bytes
        prepared["fp16"].append((kf, vf))
        prepared["u4_token_V_only"].append((kf, vu))
        prepared["u4_token_KV"].append((ku, vu))
        prepared["scalar6_page_V_only"].append((kf, va.decoded))
        prepared["scalar6_page_KV"].append((ka.decoded, va.decoded))
        prepared["scalar6_token_V_only"].append((kf, vt.decoded))
        prepared["scalar6_token_KV"].append((kt.decoded, vt.decoded))
        byte_counts["fp16"]["uncompressed_prefix_fp16"] += byte_kf + byte_vf
        byte_counts["u4_token_V_only"]["uncompressed_prefix_fp16"] += byte_kf
        byte_counts["u4_token_V_only"]["compressed_prefix_payload_and_metadata"] += byte_vu
        byte_counts["u4_token_KV"]["compressed_prefix_payload_and_metadata"] += byte_ku + byte_vu
        byte_counts["scalar6_page_V_only"]["uncompressed_prefix_fp16"] += byte_kf
        byte_counts["scalar6_page_V_only"]["compressed_prefix_payload_and_metadata"] += byte_va
        byte_counts["scalar6_page_KV"]["compressed_prefix_payload_and_metadata"] += byte_ka + byte_va
        byte_counts["scalar6_token_V_only"]["uncompressed_prefix_fp16"] += byte_kf
        byte_counts["scalar6_token_V_only"]["compressed_prefix_payload_and_metadata"] += byte_vt
        byte_counts["scalar6_token_KV"]["compressed_prefix_payload_and_metadata"] += byte_kt + byte_vt
        per_layer.append({"layer": layer, "k_astc": ka.byte_counts, "v_astc": va.byte_counts,
                          "k_u4": ku_counts, "v_u4": vu_counts,
                          "k_astc_token": kt.byte_counts, "v_astc_token": vt.byte_counts,
                          "k_astc_cpu_encode_seconds_diagnostic": ka.encode_cpu_s,
                          "v_astc_cpu_encode_seconds_diagnostic": va.encode_cpu_s})
        print(json.dumps({"stage": "prefix_codec", "fixture": log, "layer": layer}), flush=True)
    return prepared, byte_counts, per_layer


def run_teacher_forced(model, token_ids, caches, prefix_tokens):
    losses = []
    # Target prefix_tokens was predicted before compressing the prefix and is
    # intentionally excluded. Consuming it exposes compressed-history effects.
    for position in range(prefix_tokens, len(token_ids) - 1):
        logits, caches = model.step(int(token_ids[position]), position, caches, kv_transform=tail_fp16)
        target_position = position + 1
        losses.append({"target_position": target_position, "target_id": int(token_ids[target_position]),
                       "nll": next_token_loss(logits, int(token_ids[target_position]))})
    return losses


def run_greedy(model, prefill_logits, caches, prefix_tokens, steps):
    emitted = []
    logits = prefill_logits
    for i in range(steps):
        token = int(np.argmax(logits))
        emitted.append(token)
        if i + 1 < steps:
            logits, caches = model.step(token, prefix_tokens + i, caches, kv_transform=tail_fp16)
    return emitted


def register_decision(path, config, args):
    decision = {
        "schema": "astckv.model_quality_decision.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "registered_before_model_quality_results": True,
        "basis": "Original scalar6_page retained; scalar6_token added prospectively using only systems TUNE snapshot: V-only relative L2 page .2295 failed, token .0744 passed screen. No development/holdout snapshot or model-quality results inspected for this amendment.",
        "methods": METHODS, "generation_methods": GENERATION_METHODS,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "all_layers": True, "prefix_tokens": args.prefix_tokens, "context_tokens": args.context_tokens,
        "compressed_page_tokens": args.page_tokens, "astc_configs": [ASTC_CONFIG, ASTC_TOKEN_CONFIG], "astcenc_quality": args.quality,
        "teacher_forcing_targets": [args.prefix_tokens + 1, args.context_tokens - 1],
        "excluded_target": args.prefix_tokens,
        "fresh_tail_dtype": "FP16, widened to FP32 for CPU attention",
        "compressed_reconstructions_dtype": "FP32 with FP16 codec metadata; no extra FP16 round trip",
        "weights": "Original BF16 checkpoint exactly widened to FP32; identical in all methods",
        "arithmetic": "FP32 model, FP64 reduction only for scalar NLL measurement",
        "pilot_relative_perplexity_gate": 1.02,
        "gate_scope": "A pilot screen only; no claim of task quality or representative generalization",
        "greedy_fixture": "story", "greedy_tokens": args.greedy_tokens,
        "splits": config["split"], "confidence_intervals": "Not claimed from four authored fixtures",
        "memory_scope": "Logical encoded-prefix bytes plus FP16 tail; GPU resources/peak process memory unmeasured",
        "execution_scope": "CPU reconstructed-cache quality only; no hardware compressed-cache or GPU performance claim",
    }
    if path.exists():
        old = json.loads(path.read_text())
        keys = [k for k in decision if k != "created_utc"]
        if any(old.get(k) != decision[k] for k in keys):
            raise RuntimeError("A different preregistration already exists; choose a new output prefix")
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=PROJECT / "third_party/model")
    parser.add_argument("--captures", type=Path, default=PROJECT / "results/captures")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/pilot_v1.json")
    parser.add_argument("--out-prefix", type=Path, default=PROJECT / "results/model_quality")
    parser.add_argument("--prefix-tokens", type=int, default=288)
    parser.add_argument("--context-tokens", type=int, default=384)
    parser.add_argument("--page-tokens", type=int, default=96)
    parser.add_argument("--quality", type=float, default=98)
    parser.add_argument("--greedy-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.prefix_tokens % args.page_tokens or args.context_tokens <= args.prefix_tokens + 1:
        raise ValueError("Need completed prefix pages and at least two continuation tokens")
    config = json.loads(args.config.read_text())
    registration_path = args.out_prefix.with_name(args.out_prefix.name + "_decision.json")
    summary_path = args.out_prefix.with_suffix(".json")
    raw_path = args.out_prefix.with_suffix(".jsonl")
    if summary_path.exists() or raw_path.exists():
        raise RuntimeError("Results already exist; use a new --out-prefix to preserve prior evidence")
    registration = register_decision(registration_path, config, args)
    model = NumpyLlama(args.model_dir)
    results = {
        "schema": "astckv.model_quality.v1", "status": "running",
        "registration": registration, "registration_sha256": sha256(registration_path),
        "model_weights_sha256": sha256(args.model_dir / "model.safetensors"),
        "source_script_sha256": sha256(Path(__file__)),
        "numpy_version": np.__version__, "fixtures": [], "generation": [],
    }
    split_map = {name: split for split, names in config["split"].items() for name in names}
    records = []
    with raw_path.open("x") as raw:
        for fixture in ["systems", "story", "science", "records"]:
            data = np.load(args.captures / (fixture + ".npz"), allow_pickle=False)
            ids = data["token_ids"][:args.context_tokens]
            if len(ids) != args.context_tokens:
                raise ValueError("Capture does not contain requested number of tokens")
            prefill_logits, original_cache = model.prefill(ids[:args.prefix_tokens])
            prepared, byte_counts, per_layer_bytes = prepare_variants(original_cache, args.prefix_tokens,
                                                                       args.page_tokens, args.quality, fixture)
            original_cache = None
            fixture_results = {"fixture": fixture, "split": split_map[fixture],
                               "capture_sha256": sha256(args.captures / (fixture + ".npz")),
                               "input_token_ids": ids.tolist(), "methods": {},
                               "per_layer_byte_counts": per_layer_bytes}
            for method in METHODS:
                started = time.perf_counter()
                losses = run_teacher_forced(model, ids, prepared[method], args.prefix_tokens)
                mean_nll = float(np.mean([x["nll"] for x in losses]))
                for item in losses:
                    record = {"kind": "teacher_forcing", "fixture": fixture, "split": split_map[fixture], "method": method, **item}
                    raw.write(json.dumps(record) + "\n")
                raw.flush()
                byte_record = dict(byte_counts[method])
                # Final loss is predicted with cache length context_tokens-1.
                tail_tokens = args.context_tokens - 1 - args.prefix_tokens
                scalars_per_token = 2 * model.num_layers * model.hkv * model.head_dim
                byte_record["fp16_tail_at_last_prediction"] = tail_tokens * scalars_per_token * 2
                total_bytes = sum(byte_record.values())
                metrics = {"evaluated_targets": len(losses), "first_target": args.prefix_tokens + 1,
                           "last_target": args.context_tokens - 1, "mean_nll": mean_nll,
                           "perplexity": math.exp(mean_nll), "logical_bytes_at_last_prediction": total_bytes,
                           "logical_byte_breakdown": byte_record,
                           "effective_bits_per_combined_kv_scalar_at_last_prediction": total_bytes*8 / ((args.context_tokens-1)*scalars_per_token),
                           "cpu_wall_seconds_diagnostic_only": time.perf_counter() - started}
                fixture_results["methods"][method] = metrics
                baseline = fixture_results["methods"]["fp16"]["mean_nll"]
                metrics["delta_nll_vs_fp16"] = mean_nll - baseline
                metrics["relative_perplexity_vs_fp16"] = math.exp(mean_nll - baseline)
                metrics["pilot_ppl_gate_passed"] = metrics["relative_perplexity_vs_fp16"] <= 1.02
                print(json.dumps({"stage": "teacher_forcing", "fixture": fixture, "method": method, **metrics}), flush=True)
                # Persist progress after every method; raw rows survive an interruption.
                results["fixtures"] = records + [fixture_results]
                summary_path.write_text(json.dumps(results, indent=2) + "\n")
            if fixture == "story":
                baseline_ids = None
                for method in GENERATION_METHODS:
                    emitted = run_greedy(model, prefill_logits, prepared[method], args.prefix_tokens, args.greedy_tokens)
                    if baseline_ids is None:
                        baseline_ids = emitted
                    agreement = sum(a == b for a, b in zip(emitted, baseline_ids)) / len(emitted)
                    gen = {"fixture": fixture, "split": split_map[fixture], "method": method,
                           "prefix_token_count": args.prefix_tokens, "generated_token_ids": emitted,
                           "generated_text": decode_text(args.model_dir, emitted), "greedy_tokens": len(emitted),
                           "same_position_agreement_vs_fp16": agreement,
                           "agreement_scope": "Diagnostic only; different sequences are not a task-quality verdict",
                           "first_token_uses_unchanged_prefill_logits": True}
                    results["generation"].append(gen)
                    raw.write(json.dumps({"kind": "autoregressive_greedy", **gen}) + "\n")
                    raw.flush()
                    print(json.dumps({"stage": "generation", **gen}), flush=True)
            records.append(fixture_results)
            results["fixtures"] = records
            summary_path.write_text(json.dumps(results, indent=2) + "\n")
        results["aggregate"] = {}
        for method in METHODS:
            total_targets = sum(r["methods"][method]["evaluated_targets"] for r in records)
            mean = sum(r["methods"][method]["mean_nll"] * r["methods"][method]["evaluated_targets"] for r in records)/total_targets
            baseline = sum(r["methods"]["fp16"]["mean_nll"] * r["methods"]["fp16"]["evaluated_targets"] for r in records)/total_targets
            results["aggregate"][method] = {"targets": total_targets, "mean_nll": mean, "perplexity": math.exp(mean),
                                            "delta_nll_vs_fp16": mean-baseline, "relative_perplexity_vs_fp16": math.exp(mean-baseline),
                                            "pilot_ppl_gate_passed": math.exp(mean-baseline)<=1.02,
                                            "inference_scope": "Descriptive aggregate of four authored pilot fixtures; no CI or generalization claim"}
        results["status"] = "completed"
        results["raw_results_sha256"] = sha256(raw_path)
        summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"stage": "completed", "summary": str(summary_path), "raw": str(raw_path),
                      "aggregate": results["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()

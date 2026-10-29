#!/usr/bin/env python3
"""Post-holdout diagnostic: asymmetric numerical U4 KV baseline.

This follow-up was selected after inspecting the snapshot holdout. It is not
part of the prospectively frozen primary model-quality comparison.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import math
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
from astc_kv.codec import affine_quantize
from capture_qkv import NumpyLlama, sha256
from evaluate_model import run_teacher_forced


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=PROJECT / "third_party/model")
    p.add_argument("--captures", type=Path, default=PROJECT / "results/captures")
    p.add_argument("--primary", type=Path, default=PROJECT / "results/model_quality.json")
    p.add_argument("--out-prefix", type=Path, default=PROJECT / "results/model_quality_secondary")
    args = p.parse_args()
    primary = json.loads(args.primary.read_text())
    if primary["status"] != "completed":
        raise RuntimeError("Primary study must finish before the secondary diagnostic")
    if sha256(args.model_dir / "model.safetensors") != primary["model_weights_sha256"]:
        raise RuntimeError("Checkpoint differs from primary baseline")
    prefix = primary["registration"]["prefix_tokens"]
    context = primary["registration"]["context_tokens"]
    page = primary["registration"]["compressed_page_tokens"]
    method = "u4_asymmetric_KV"
    decision_path = args.out_prefix.with_name(args.out_prefix.name + "_decision.json")
    output_path = args.out_prefix.with_suffix(".json")
    raw_path = args.out_prefix.with_suffix(".jsonl")
    if any(x.exists() for x in (decision_path, output_path, raw_path)):
        raise RuntimeError("Secondary output already exists; preserve it and choose a new prefix")
    decision = {
        "schema": "astckv.model_quality_secondary_decision.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "post_holdout_secondary_diagnostic",
        "selection_reason": "Snapshot heldout results showed channel-wise K U4 + token-wise V U4 was substantially stronger than token-wise U4 for both tensors. Added as a stronger numerical baseline after holdout inspection.",
        "not_part_of_primary_preregistration": True,
        "method": method, "K": {"bits":4,"axis":"channel","group":page},
        "V": {"bits":4,"axis":"token","group":64},
        "prefix_tokens": prefix, "context_tokens": context,
        "teacher_forcing_targets": [prefix+1, context-1],
        "fresh_tail": "FP16, with real autoregressive hidden-state feedback under teacher forcing",
        "primary_baseline": str(args.primary), "primary_sha256":sha256(args.primary),
        "gate": "Same provisional relative PPL <=1.02, not a confirmation of generalized model quality",
        "claim_limit": "Numerical affine baseline, not a KIVI implementation or Metal speed measurement",
    }
    decision_path.write_text(json.dumps(decision,indent=2)+"\n")
    model = NumpyLlama(args.model_dir)
    results = {"schema":"astckv.model_quality_secondary.v1", "decision":decision,
               "source_script_sha256":sha256(Path(__file__)), "fixtures":[]}
    with raw_path.open("x") as raw:
        for fixture in primary["fixtures"]:
            name = fixture["fixture"]
            path = args.captures / (name+".npz")
            if sha256(path) != fixture["capture_sha256"]:
                raise RuntimeError("Capture changed since primary experiment")
            ids = np.load(path,allow_pickle=False)["token_ids"][:context]
            if ids.tolist() != fixture["input_token_ids"]:
                raise RuntimeError("Input token IDs differ from primary baseline")
            _, original = model.prefill(ids[:prefix])
            caches=[]
            byte_counts={"packed_codes":0,"fp16_offsets":0,"fp16_scales":0}
            for k,v in original:
                kq,kb=affine_quantize(k,4,axis="channel",group=page)
                vq,vb=affine_quantize(v,4,axis="token",group=64)
                caches.append((kq,vq))
                for key in byte_counts:
                    byte_counts[key]+=kb[key]+vb[key]
            losses=run_teacher_forced(model,ids,caches,prefix)
            for item in losses:
                raw.write(json.dumps({"fixture":name,"split":fixture["split"],"method":method,**item})+"\n")
            raw.flush()
            mean=float(np.mean([l["nll"] for l in losses]))
            baseline=fixture["methods"]["fp16"]["mean_nll"]
            scalars_per_token=2*model.num_layers*model.hkv*model.head_dim
            tail_bytes=(context-1-prefix)*scalars_per_token*2
            logical=sum(byte_counts.values())+tail_bytes
            row={"fixture":name,"split":fixture["split"],"method":method,
                 "targets":len(losses),"mean_nll":mean,"perplexity":math.exp(mean),
                 "delta_nll_vs_primary_fp16":mean-baseline,
                 "relative_perplexity_vs_primary_fp16":math.exp(mean-baseline),
                 "pilot_ppl_gate_passed":math.exp(mean-baseline)<=1.02,
                 "prefix_byte_counts":byte_counts,"fp16_tail_bytes":tail_bytes,
                 "logical_bytes_at_last_prediction":logical,
                 "effective_bits_per_combined_kv_scalar_at_last_prediction":logical*8/((context-1)*scalars_per_token)}
            results["fixtures"].append(row)
            output_path.write_text(json.dumps(results,indent=2)+"\n")
            print(json.dumps(row),flush=True)
    mean=float(np.mean([r["mean_nll"] for r in results["fixtures"]]))
    baseline=primary["aggregate"]["fp16"]["mean_nll"]
    results.update(status="completed",aggregate={"targets":sum(r["targets"] for r in results["fixtures"]),
                                                "mean_nll":mean,"perplexity":math.exp(mean),
                                                "delta_nll_vs_primary_fp16":mean-baseline,
                                                "relative_perplexity_vs_primary_fp16":math.exp(mean-baseline),
                                                "pilot_ppl_gate_passed":math.exp(mean-baseline)<=1.02},
                   raw_results_sha256=sha256(raw_path))
    output_path.write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps({"status":"completed","aggregate":results["aggregate"]}),flush=True)


if __name__=="__main__":
    main()

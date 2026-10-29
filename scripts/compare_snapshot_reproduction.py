#!/usr/bin/env python3
"""Compare snapshot re-encoding with archived rows; stdlib-only, no codec import.

Reports numerical differences rather than imposing a post-hoc equality tolerance.
Memory, row identities, and independently recalculated screening decisions are
separate checks. Encoder timing is intentionally not a numerical-quality metric.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


NAMES = ("systems", "story", "science", "records")
BASELINES = {"fp16", "u2_token", "u4_token", "u8_token", "u4_asymmetric"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return tuple(row[k] for k in ("capture", "layer", "variant", "mode"))


def numeric(value, prefix=""):
    if isinstance(value, dict):
        for name, v in sorted(value.items()):
            yield from numeric(v, f"{prefix}.{name}" if prefix else name)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from numeric(v, f"{prefix}[{i}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite numeric input: {prefix}={value}")
        yield prefix, value


def load(directory):
    rows = {}
    for name in NAMES:
        path = directory / f"pilot_{name}.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            k = key(row)
            if k in rows:
                raise ValueError(f"Duplicate row: {k}")
            rows[k] = row
    if len(rows) != 936:
        raise ValueError(f"Expected 936 rows, found {len(rows)}")
    return rows


def aggregate(rows):
    split = {"systems": "tune", "story": "development",
             "science": "pilot_holdout", "records": "pilot_holdout"}
    groups = defaultdict(list)
    for row in rows.values():
        groups[(split[row["capture"]], row["variant"], row["mode"])].append(row)
    result = {}
    for k, values in groups.items():
        numerator = math.fsum(r["error_sum_squares"] for r in values)
        denominator = math.fsum(r["reference_sum_squares"] for r in values)
        result[k] = {
            "relative_l2": math.sqrt(numerator / denominator),
            "mean_kl_nats": math.fsum(r["metrics"]["mean_kl_nats"] for r in values) / len(values),
            "bits_per_kv": math.fsum(r["memory"]["effective_bits_per_original_kv_scalar"] for r in values) / len(values),
            "rows": len(values),
        }
    for (split_name, variant, mode), item in result.items():
        baseline = result[(split_name, "u4_token", mode)]
        item["screen_pass"] = (item["relative_l2"] <= 0.10
                               and item["mean_kl_nats"] <= 0.01
                               and item["bits_per_kv"] <= baseline["bits_per_kv"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--reproduced-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Preserve old comparison: {args.out}")
    old = load(args.original_dir)
    new = load(args.reproduced_dir)
    if old.keys() != new.keys():
        raise ValueError("Snapshot row identities differ")
    metrics = ("metrics", "tensor_error_k", "tensor_error_v", "per_query_head",
               "per_kv_head", "error_sum_squares", "reference_sum_squares")
    changes = []
    field_stats = {}
    memory_mismatches = []
    structural_mismatches = []
    variant_stats = defaultdict(lambda: {"rows": 0, "compared_numbers": 0,
                                        "changed_numbers": 0, "max_row_relative_l2_abs_difference": 0.0,
                                        "max_error_sse_symmetric_relative_difference": 0.0,
                                        "max_reference_sse_symmetric_relative_difference": 0.0})
    for k in sorted(old):
        before, after = old[k], new[k]
        identity = dict(zip(("capture", "layer", "variant", "mode"), k))
        if before["memory"] != after["memory"]:
            memory_mismatches.append(identity)
        for field in ("schema", "prefix_tokens", "total_tokens", "query_positions", "gpu_executed"):
            if before[field] != after[field]:
                structural_mismatches.append({**identity, "field": field})
        stats = variant_stats[before["variant"]]
        stats["rows"] += 1
        av = dict(numeric({field: before[field] for field in metrics}))
        bv = dict(numeric({field: after[field] for field in metrics}))
        if av.keys() != bv.keys():
            raise ValueError(f"Numeric schema differs: {k}")
        for field in av:
            a, b = av[field], bv[field]
            delta = abs(a - b)
            relative = delta / max(abs(a), abs(b), 1e-30)
            item = {**identity, "field": field, "original": a, "reproduced": b,
                    "absolute_difference": delta, "symmetric_relative_difference": relative}
            stats["compared_numbers"] += 1
            stats["changed_numbers"] += int(delta != 0)
            if delta:
                changes.append(item)
            bucket = field_stats.setdefault(field, {"compared": 0, "changed": 0,
                                                    "max_absolute": item, "max_symmetric_relative": item})
            bucket["compared"] += 1
            bucket["changed"] += int(delta != 0)
            if delta > bucket["max_absolute"]["absolute_difference"]:
                bucket["max_absolute"] = item
            if relative > bucket["max_symmetric_relative"]["symmetric_relative_difference"]:
                bucket["max_symmetric_relative"] = item
            if field == "metrics.relative_l2":
                stats["max_row_relative_l2_abs_difference"] = max(stats["max_row_relative_l2_abs_difference"], delta)
            if field in ("error_sum_squares", "reference_sum_squares"):
                dst = "max_error_sse_symmetric_relative_difference" if field == "error_sum_squares" else "max_reference_sse_symmetric_relative_difference"
                stats[dst] = max(stats[dst], relative)
    old_agg, new_agg = aggregate(old), aggregate(new)
    if len(old_agg) != 117 or old_agg.keys() != new_agg.keys():
        raise ValueError("Aggregate groups differ from expected 117")
    aggregates = []
    gate_changes = []
    for k in sorted(old_agg):
        item = dict(zip(("split", "variant", "mode"), k))
        item.update(original=old_agg[k], reproduced=new_agg[k],
                    relative_l2_abs_difference=abs(old_agg[k]["relative_l2"]-new_agg[k]["relative_l2"]))
        aggregates.append(item)
        if old_agg[k]["screen_pass"] != new_agg[k]["screen_pass"]:
            gate_changes.append(item)
    stored = json.loads((args.reproduced_dir / "pilot_summary.json").read_text())["results"]
    stored_keys = {(r["split"], r["variant"], r["mode"]) for r in stored}
    if len(stored) != 117 or stored_keys != new_agg.keys():
        raise ValueError("Stored summarizer group identities differ")
    summary_delta = max(abs(r["relative_l2"]-new_agg[(r["split"],r["variant"],r["mode"])]["relative_l2"]) for r in stored)
    summary_gate_agreement = all(r["pilot_screen_pass"] == new_agg[(r["split"],r["variant"],r["mode"])]["screen_pass"] for r in stored)
    result = {
        "schema": "astckv.snapshot_reproduction.comparison.v1",
        "scope": "Independent row and aggregate comparison after CPU re-encoding; no new holdout or GPU speed claim",
        "rows": len(old), "aggregates": len(aggregates),
        "source_sha256": sha(Path(__file__)),
        "input_sha256": {label: {f"pilot_{name}.jsonl": sha(path / f"pilot_{name}.jsonl") for name in NAMES}
                         for label, path in (("original", args.original_dir), ("reproduced", args.reproduced_dir))},
        "structural_mismatches": structural_mismatches, "memory_mismatches": memory_mismatches,
        "screening_changes": gate_changes, "variant_stats": dict(variant_stats),
        "numeric_fields": field_stats,
        "top_absolute_differences": sorted(changes, key=lambda x:x["absolute_difference"], reverse=True)[:20],
        "top_symmetric_relative_differences": sorted(changes, key=lambda x:x["symmetric_relative_difference"], reverse=True)[:20],
        "aggregate_comparison": aggregates,
        "reproduced_summary_max_difference_vs_independent": summary_delta,
        "reproduced_summary_gate_agreement": summary_gate_agreement,
        "equality_tolerance": "None chosen post hoc: numerical differences are reported. Exact memory and screening agreement are separate checks.",
    }
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key:result[key] for key in ("rows", "aggregates", "structural_mismatches",
                     "memory_mismatches", "screening_changes", "variant_stats",
                     "reproduced_summary_max_difference_vs_independent", "reproduced_summary_gate_agreement")}, indent=2))


if __name__ == "__main__":
    main()

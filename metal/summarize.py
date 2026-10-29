"""Summarize raw probe timings without turning kernel results into inference claims."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    groups = {}
    for sample in result["raw_samples"]:
        group = groups.setdefault(sample["variant"], {"gpu_ms": [], "wall_ms": []})
        group["wall_ms"].append(sample["wall_batch_ms"] / sample["iterations"])
        if sample["gpu_batch_ms"] is not None:
            group["gpu_ms"].append(sample["gpu_batch_ms"] / sample["iterations"])
    rng = np.random.default_rng(args.seed)
    summary = {}
    for variant, metrics in groups.items():
        summary[variant] = {}
        for metric, samples in metrics.items():
            if not samples:
                summary[variant][metric] = None
                continue
            a = np.asarray(samples)
            medians = np.median(rng.choice(a, (10000, len(a)), replace=True), axis=1)
            summary[variant][metric] = {
                "median": float(np.median(a)),
                "bootstrap_median_ci95": np.quantile(medians, [0.025, 0.975]).tolist(),
                "n_rounds": len(a),
            }
    print(json.dumps({
        "scope": "Diagnostic attention kernel only; repeated reads of one cache.",
        "warning": "Bootstrap assumes exchangeable rounds; inspect drift and rerun on independent sessions.",
        "summary": summary,
        "memory": result["memory"],
    }, indent=2))


if __name__ == "__main__":
    main()

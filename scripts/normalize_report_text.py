#!/usr/bin/env python3
"""Normalize the frozen report inventory, preserving historical bytes first.

The inventory is intentionally explicit: later research documents are not owned
by this migration. The audit records this stage, not a ban on future edits.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
REPORTS = (
    "README.md", "docs/atlas_alloc_probe.md", "docs/atlas_attention_v6.md",
    "docs/block_geometry_v2.md", "docs/followup_experiments.md",
    "docs/geometry_holdout_v2.md", "docs/independent_research_20261028.md",
    "docs/initial_report.md", "docs/kernel_attribution_design_v2.md",
    "docs/llama_baseline_v2.md", "docs/metal_path.md",
    "docs/metal_revalidation_20261028.md", "docs/model_baselines.md",
    "docs/pool_probe_v5.md", "docs/prior_art.md", "docs/protocol.md",
    "docs/representation_v0.1.md", "docs/research_decision_v2.md",
    "docs/research_v2_report.md", "docs/revalidation_report_20261028.md",
    "docs/snapshot_revalidation_20261028.md",
)
MANIFEST = "results/kernel_v2/research_v2_manifest.json"
BACKUP = "results/research_v3_20261029/report_style_backup"
AUDIT = "results/research_v3_20261029/style_audit.json"
PROMPT = "docs/master_prompt_v0.1_ru.md"
URL = re.compile(r'https?://[^\s<>"`]+')
INLINE = re.compile(r'(`+)(.+?)\1')
CYR = r"А-Яа-яЁё"
REPLACEMENTS = {
    "\u00a0": " ", "\u202f": " ", "\u200b": " ", "\ufeff": " ",
    "\u200c": " ", "\u200d": " ", "\u2060": " ",
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2212": "-",
    "\u00ab": '"', "\u00bb": '"', "\u201c": '"', "\u201d": '"',
    "\u2018": "'", "\u2019": "'", "\u2026": "...", "\u2192": "->",
    "\u2264": "<=", "\u2265": ">=", "\u2248": "~", "\u00b1": "+/-",
    "\u00b2": "^2", "\u00b3": "^3", "\u00a7": "section ",
    "\u0394": "delta_", "\u03b4": "delta_", "\u03b8": "theta",
    "\u03c1": "rho", "\u03a3": "sum_t", "\u03bc": "mu",
    "\u00b5": "mu", "\u00b7": "*", "\u00d7": "*",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def allowed(char: str) -> bool:
    return (32 <= ord(char) < 127 or char in "\n\r\t" or
            (unicodedata.category(char).startswith("L") and
             "CYRILLIC" in unicodedata.name(char, "")))


def special_counts(text: str) -> Counter:
    return Counter(c for c in text if not allowed(c))


def notation(text: str, inline: bool = False) -> str:
    text = text.replace("\u00b5s", "us").replace("\u03bcs", "us")
    text = text.replace("q\u22a5k", "dot(q,k)=0")
    if not inline:
        text = text.replace(" \u00b7 ", " - ")
        text = text.replace("(32\u00d73+96\u00d72)/128", "(32*3+96*2)/128")
        text = re.sub(r"(?<=\d)\u00d7(?=\d)", "x", text)
        text = re.sub(r"(?<=\d)\u00d7(?![\w])", "x", text)
        text = text.replace("Bx\u00d7By", "Bx x By")
    return "".join(REPLACEMENTS.get(c, c) for c in text)


def prose(text: str) -> str:
    text = notation(text)
    # Repair only word/number and number/unit joins, outside URLs and code.
    text = re.sub(rf"([{CYR}]{{2,}})(?=\d)", r"\1 ", text)
    text = re.sub(rf"(?<=\d)([{CYR}]{{2,}})", r" \1", text)
    text = re.sub(rf"(?<![\w])([исвк])(?=\d)", r"\1 ", text)
    text = re.sub(r"(?<![\w.])([0-9]+(?:\.[0-9]+)?)(bytes|bits|targets|pairs|rows|s)(?!\w)", r"\1 \2", text)
    text = re.sub(rf"([;:])(?=[{CYR}A-Za-z0-9])", r"\1 ", text)
    return text


def fenced_blocks(text: str) -> list[str]:
    blocks, current, fence = [], [], None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence is None and marker:
            fence = marker.group(1)
            current = [line]
        elif fence is not None:
            current.append(line)
            if re.match(r"^ {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$", line):
                blocks.append("".join(current))
                current, fence = [], None
    if current:
        raise ValueError("unclosed fenced block")
    return blocks


def normalize(text: str) -> str:
    protected = {}

    def protect(value: str) -> str:
        key = f"\x00PROTECTED{len(protected)}\x00"
        protected[key] = value
        return key

    for block in fenced_blocks(text):
        text = text.replace(block, protect(block), 1)
    text = URL.sub(lambda m: protect(m.group(0)), text)
    text = INLINE.sub(lambda m: protect(m.group(1) + notation(m.group(2), inline=True) + m.group(1)), text)
    text = prose(text)
    for key, value in reversed(tuple(protected.items())):
        text = text.replace(key, value)
    if special_counts(text):
        raise ValueError(f"unmapped or protected special characters: {dict(special_counts(text))}")
    return text


def invariants(before: str, after: str) -> dict:
    ascii_inline = lambda s: [m.group(0) for m in INLINE.finditer(s) if all(allowed(c) for c in m.group(0))]
    original_codes = ascii_inline(before)
    new_codes = ascii_inline(after)
    # New ASCII formula spans may be added; original command spans retain order.
    iterator = iter(new_codes)
    inline_preserved = all(any(value == candidate for candidate in iterator) for value in original_codes)
    result = {
        "urls_preserved": URL.findall(before) == URL.findall(after),
        "fenced_blocks_preserved": fenced_blocks(before) == fenced_blocks(after),
        "original_ascii_inline_spans_preserved": inline_preserved,
        "urls": len(URL.findall(before)),
        "fenced_blocks": len(fenced_blocks(before)),
        "original_ascii_inline_spans": len(original_codes),
    }
    if not all(result[k] for k in result if k.endswith("preserved")):
        raise ValueError(f"protected content changed: {result}")
    if normalize(after) != after:
        raise ValueError("normalization is not idempotent")
    return result


def self_test() -> dict:
    cases = {
        "6\u00d76, 48\u00d7, (32\u00d73+96\u00d72)/128": "6x6, 48x, (32*3+96*2)/128",
        "`\u0394O=P\u00b7E` `q\u22a5k` `sin(\u03b4\u03b8)`": "`delta_O=P*E` `dot(q,k)=0` `sin(delta_theta)`",
        "`E_s ||\u03a3 p_t s_t e_t||\u00b2 = \u03a3 p_t\u00b2 ||e_t||\u00b2`": "`E_s ||sum_t p_t s_t e_t||^2 = sum_t p_t^2 ||e_t||^2`",
        "`max_abs\u22641e\u22123\u00d7max(1,reference_max_abs)`": "`max_abs<=1e-3*max(1,reference_max_abs)`",
        "\u00abtest\u00bb \u2014 2 \u03bcs\u00a0\u2192\u200b3 \u00b5s": '"test" - 2 us -> 3 us',
        "внутри6\u00d76 и28800bytes, с0.5": "внутри 6x6 и 28800 bytes, с 0.5",
        "https://example.test/a:3?q=FP16 `A=1;B=2`\n```sh\necho 'a;b'\n```\n": "https://example.test/a:3?q=FP16 `A=1;B=2`\n```sh\necho 'a;b'\n```\n",
    }
    for before, expected in cases.items():
        actual = normalize(before)
        assert actual == expected, (before, actual, expected)
        invariants(before, actual)
    try:
        normalize("unknown \u2603")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown symbol accepted")
    return {"status": "PASS", "cases": len(cases) + 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return
    originals = {path: (ROOT / path).read_bytes() for path in REPORTS}
    normalized, rows, counts = {}, [], Counter()
    for path, raw in originals.items():
        before = raw.decode("utf-8")
        after = normalize(before)
        normalized[path] = after.encode("utf-8")
        counts.update(special_counts(before))
        rows.append({"path": path, "before_sha256": sha(raw),
                     "stage_after_sha256": sha(normalized[path]),
                     "before_bytes": len(raw), "stage_after_bytes": len(normalized[path]),
                     "changed": raw != normalized[path], "invariants": invariants(before, after)})
    summary = {"status": "PASS", "reports": len(rows),
               "changed": sum(row["changed"] for row in rows),
               "special_characters": sum(counts.values()), "special_types": len(counts)}
    if not args.apply:
        print(json.dumps({**summary, "mode": "check", "changed_paths": [r["path"] for r in rows if r["changed"]]}, indent=2))
        return
    from verify_historical_reports import verify
    initial = verify(ROOT, ROOT / MANIFEST, None)
    if initial["status"] != "PASS":
        raise ValueError(f"initial manifest verification failed: {initial}")
    backup_dir, audit_path = ROOT / BACKUP, ROOT / AUDIT
    if backup_dir.exists() or audit_path.exists():
        raise FileExistsError("refusing to overwrite style backup or audit")
    prompt_sha = sha((ROOT / PROMPT).read_bytes())
    backup_dir.mkdir(parents=True)
    for row in rows:
        relative = "files/" + row["path"]
        target = backup_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(originals[row["path"]])
        assert sha(target.read_bytes()) == row["before_sha256"]
        row["backup_path"] = relative
    backup_manifest = {
        "schema": "astc-kv-report-style-backup-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "historical_manifest_path": MANIFEST,
        "historical_manifest_sha256": sha((ROOT / MANIFEST).read_bytes()),
        "scope": "Frozen report inventory; stage_after hashes describe this migration only. Future report edits are allowed.",
        "files": rows,
    }
    (backup_dir / "manifest.json").write_text(json.dumps(backup_manifest, indent=2) + "\n", encoding="utf-8")
    # Fail before modifying any report if a concurrent writer changed a source.
    for path, raw in originals.items():
        if (ROOT / path).read_bytes() != raw:
            raise ValueError(f"concurrent change: {path}")
    for row in rows:
        if row["changed"]:
            (ROOT / row["path"]).write_bytes(normalized[row["path"]])
    assert sha((ROOT / PROMPT).read_bytes()) == prompt_sha
    final_verification = verify(ROOT, ROOT / MANIFEST, backup_dir / "manifest.json")
    if final_verification["status"] != "PASS":
        raise ValueError(f"final historical verification failed: {final_verification}")
    replacements = [{"codepoint": f"U+{ord(c):04X}", "name": unicodedata.name(c),
                     "count": count, "rule": REPLACEMENTS.get(c, "q perpendicular k -> dot(q,k)=0")}
                    for c, count in sorted(counts.items(), key=lambda item: ord(item[0]))]
    audit = {
        **summary, "schema": "astc-kv-report-style-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_initial_verification": initial,
        "source_manifest_after_style_verification": final_verification,
        "stage_after_special_characters": 0,
        "backup_manifest": BACKUP + "/manifest.json",
        "backup_manifest_sha256": sha((backup_dir / "manifest.json").read_bytes()),
        "original_prompt_sha256_unchanged": prompt_sha,
        "allowed": "ASCII printable characters, tabs/newlines, Cyrillic letters",
        "special_character_rules": replacements,
        "context_rules": ["mu/Greek mu + s -> us", "numeric dimensions -> x; speed factors -> x; inline products -> *", "spaced prose middle dot -> hyphen", "word/number and numeric unit spacing only outside URLs and inline/fenced code", "sum_t denotes summation over t; delta_theta denotes the angular increment"],
        "scope_note": "Historical results and manifests are unchanged. Stage hashes do not prohibit later research edits. No claim about authorship or detector output.",
        "files": rows,
    }
    with audit_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({**summary, "mode": "applied", "audit": AUDIT, "backup": BACKUP}, indent=2))


if __name__ == "__main__":
    main()

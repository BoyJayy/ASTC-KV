#!/usr/bin/env python3
"""Verify a frozen manifest, allowing original report bytes from a style backup.

Only README.md and docs/*.md reports may use this exception. The source prompt,
raw data, source code and binary hashes must always match their current files.
The source manifest itself is anchored by the backup manifest's recorded hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "results/kernel_v2/research_v2_manifest.json"
BACKUP = "results/research_v3_20261029/report_style_backup/manifest.json"
ORIGINAL_ARCHIVE = Path("/home/researcher/Downloads/ASTC-KV_v0.1.zip")
PREVIOUS_MANIFEST = "results/revalidation_20261028/continuation_manifest.json"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(base: Path, name: str) -> Path:
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"unsafe relative path: {name}")
    target = base / rel
    if not target.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"path outside base: {name}")
    return target


def eligible(name: str) -> bool:
    path = PurePosixPath(name)
    return name == "README.md" or (path.parent == PurePosixPath("docs") and
           path.suffix == ".md" and path.name != "master_prompt_v0.1_ru.md")


def verify(root: Path, manifest_path: Path, backup_path: Path | None,
           archive_path: Path = ORIGINAL_ARCHIVE) -> dict:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    backups, failures = {}, []
    backup_integrity_count = 0
    if backup_path is not None:
        backup = json.loads(backup_path.read_bytes())
        if backup["historical_manifest_sha256"] != sha(manifest_bytes):
            failures.append({"path": str(manifest_path), "reason": "historical manifest hash changed"})
        for row in backup["files"]:
            name = row["path"]
            if name in backups or not eligible(name):
                raise ValueError(f"duplicate or ineligible backup: {name}")
            backups[name] = row
            original = safe_path(backup_path.parent, row["backup_path"])
            original_data = original.read_bytes() if original.is_file() else None
            if original_data is not None and len(original_data) == row["before_bytes"] and sha(original_data) == row["before_sha256"]:
                backup_integrity_count += 1
            else:
                failures.append({"path": name, "reason": "original report backup integrity failed"})
    current_count, backup_count = 0, 0
    seen = set()
    for row in manifest["files"]:
        name = row["path"]
        if name in seen:
            raise ValueError(f"duplicate historical entry: {name}")
        seen.add(name)
        current = safe_path(root, name)
        data = current.read_bytes() if current.is_file() else None
        if data is not None and len(data) == row["bytes"] and sha(data) == row["sha256"]:
            current_count += 1
            continue
        old = backups.get(name)
        if old is not None and old["before_sha256"] == row["sha256"] and old["before_bytes"] == row["bytes"]:
            original = safe_path(backup_path.parent, old["backup_path"])
            data = original.read_bytes() if original.is_file() else None
            if data is not None and len(data) == row["bytes"] and sha(data) == row["sha256"]:
                backup_count += 1
                continue
        failures.append({"path": name, "reason": "no current or eligible original backup match"})
    binaries = 0
    for name, expected in manifest.get("local_binary_hashes", {}).items():
        target = safe_path(root, name)
        if target.is_file() and sha(target.read_bytes()) == expected:
            binaries += 1
        else:
            failures.append({"path": name, "reason": "binary hash mismatch"})
    anchors = {}
    for key, target in (("previous_manifest_sha256", safe_path(root, PREVIOUS_MANIFEST)),
                        ("original_archive_sha256", archive_path)):
        if key in manifest:
            anchors[key] = target.is_file() and file_sha(target) == manifest[key]
            if not anchors[key]:
                failures.append({"path": str(target), "reason": "historical anchor hash mismatch"})
    return {
        "status": "FAIL" if failures else "PASS",
        "manifest_sha256": sha(manifest_bytes),
        "historical_files": len(manifest["files"]),
        "matched_current": current_count, "matched_original_report_backup": backup_count,
        "report_backups": len(backups), "matched_report_backups": backup_integrity_count,
        "local_binaries": len(manifest.get("local_binary_hashes", {})), "matched_local_binaries": binaries,
        "historical_anchors": anchors,
        "failures": failures,
    }


def self_test() -> dict:
    with tempfile.TemporaryDirectory(prefix="astc-report-history-") as directory:
        root = Path(directory)
        (root / "README.md").write_bytes(b"original\n")
        (root / "raw.json").write_bytes(b"{}\n")
        entries = [{"path": name, "bytes": len((root / name).read_bytes()), "sha256": sha((root / name).read_bytes())} for name in ("README.md", "raw.json")]
        manifest_path = root / "historical.json"
        manifest_path.write_text(json.dumps({"files": entries}))
        assert verify(root, manifest_path, None)["status"] == "PASS"
        backup_path = root / "backup.json"
        (root / "original.md").write_bytes(b"original\n")
        data = {"historical_manifest_sha256": sha(manifest_path.read_bytes()), "files": [{"path": "README.md", "before_sha256": entries[0]["sha256"], "before_bytes": entries[0]["bytes"], "backup_path": "original.md"}]}
        backup_path.write_text(json.dumps(data))
        (root / "README.md").write_text("updated later, not a frozen stage hash\n")
        assert verify(root, manifest_path, backup_path)["matched_original_report_backup"] == 1
        (root / "original.md").write_text("corrupt\n")
        assert verify(root, manifest_path, backup_path)["status"] == "FAIL"
        (root / "original.md").write_bytes(b"original\n")
        (root / "raw.json").write_bytes(b"changed\n")
        assert verify(root, manifest_path, backup_path)["status"] == "FAIL"
        data["files"].append({"path": "raw.json"})
        backup_path.write_text(json.dumps(data))
        try:
            verify(root, manifest_path, backup_path)
        except ValueError:
            pass
        else:
            raise AssertionError("raw backup accepted")
        data["files"].pop()
        data["historical_manifest_sha256"] = "0" * 64
        backup_path.write_text(json.dumps(data))
        assert verify(root, manifest_path, backup_path)["status"] == "FAIL"
        for name in ("../escape", "/absolute"):
            try:
                safe_path(root, name)
            except ValueError:
                pass
            else:
                raise AssertionError("unsafe path accepted")
        (root / "raw.json").write_bytes(b"{}\n")
        (root / "README.md").write_bytes(b"original\n")
        previous = root / PREVIOUS_MANIFEST
        previous.parent.mkdir(parents=True)
        previous.write_text("previous manifest\n")
        archive = root / "archive.zip"
        archive.write_bytes(b"synthetic archive contents\n")
        manifest_path.write_text(json.dumps({"files": entries, "previous_manifest_sha256": file_sha(previous), "original_archive_sha256": file_sha(archive)}))
        assert verify(root, manifest_path, None, archive)["status"] == "PASS"
        archive.write_bytes(b"corrupt archive\n")
        assert verify(root, manifest_path, None, archive)["status"] == "FAIL"
    return {"status": "PASS", "cases": 10}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--backup", default=BACKUP)
    parser.add_argument("--without-backup", action="store_true")
    parser.add_argument("--archive", type=Path, default=ORIGINAL_ARCHIVE,
                        help="original archive path; required only when the manifest records its hash")
    parser.add_argument("--out")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = self_test() if args.self_test else verify(ROOT, safe_path(ROOT, args.manifest), None if args.without_backup else safe_path(ROOT, args.backup), args.archive)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.out:
        with safe_path(ROOT, args.out).open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    print(encoded, end="")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

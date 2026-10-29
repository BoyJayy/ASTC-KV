#!/usr/bin/env python3
"""Verify the public export, independent of private paths and old local binaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'release/manifest.json'
TEXT = {'.py', '.md', '.json', '.jsonl', '.txt', '.sh', '.mm', '.metal', '.cpp', '.h', '.log', '.stderr', '.svg'}
SECRET = re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9_-]{25,}|AKIA[A-Z0-9]{16})\b')
EXCLUDED = {'.git', 'build', 'third_party', '__pycache__', 'dist'}
LOCAL_ROOTS = {'astc-kv', 'github-release', '.vscode', '.idea'}
LOCAL_BINARIES = {
    'results/revalidation_20261028/metal/astc-kv-probe-original',
    'results/revalidation_20261028/metal/astc-kv-probe-readback',
    'results/revalidation_20261028/metal/astc-kv-probe-runtime',
    'results/revalidation_20261028/metal/launcher_build/astc-kv-probe',
}


def eligible(path):
    rel = path.relative_to(ROOT)
    if rel.parts[0] in LOCAL_ROOTS or rel.parts[0].startswith('research-backup-') or rel.as_posix() in LOCAL_BINARIES:
        return False
    return not any(a in EXCLUDED or a.startswith('.venv') for a in rel.parts) and path.suffix not in {'.pyc', '.so', '.dylib'}


def snapshot_files():
    for parent, directories, names in os.walk(ROOT, followlinks=False):
        parent = Path(parent)
        directories[:] = [name for name in directories if eligible(parent / name)]
        for name in names:
            path = parent / name
            if eligible(path) and path != MANIFEST:
                yield path


def safe_path(name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('Unsafe manifest path')
    path = ROOT / rel
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
        raise ValueError('Manifest path escapes snapshot')
    return path


def metadata(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def scan(paths):
    count = 0
    size = 0
    largest = 0
    for path in paths:
        if path.is_symlink():
            raise ValueError('Symlinks are not part of the export')
        raw = path.read_bytes()
        count += 1; size += len(raw); largest = max(largest, len(raw))
        if len(raw) >= 25 * 1024 * 1024:
            raise ValueError('File exceeds the selected 25 MiB packaging limit: ' + str(path.relative_to(ROOT)))
        if path.name == '.env' or path.name.startswith('.env.') or path.suffix in {'.pem', '.key', '.p12', '.pfx'}:
            raise ValueError('Local credential/configuration file in export')
        if raw[:4] in (b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf', b'\x7fELF'):
            raise ValueError('Native executable in export')
        if path.suffix in TEXT or path.name in ('LICENSE', '.gitignore', '.gitattributes'):
            raw.decode('utf-8')
            if SECRET.search(raw):
                raise ValueError('Credential-shaped content found; value omitted: ' + str(path.relative_to(ROOT)))
            if (b'/' + b'Users' + b'/') in raw or (b'\\/' + b'Users' + b'\\/') in raw:
                raise ValueError('Private home path remains: ' + str(path.relative_to(ROOT)))
    return {'files': count, 'bytes': size, 'largest_file_bytes': largest}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scan-only', action='store_true', help='Preparation check before sealing; no integrity claim')
    args = ap.parse_args()
    if args.scan_only:
        result = scan(snapshot_files())
        result.update(status='PASS', mode='content_scan_only_not_hash_verification')
    else:
        manifest = json.loads(MANIFEST.read_text())
        if manifest['schema'] != 'astckv.public_snapshot.v1':
            raise ValueError('Unknown snapshot manifest schema')
        paths = []
        for name, expected in manifest['files'].items():
            path = safe_path(name)
            if metadata(path) != expected:
                raise ValueError('Snapshot hash mismatch: ' + name)
            paths.append(path)
        result = scan(paths)
        result.update(status='PASS', mode='public_snapshot_sha256')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

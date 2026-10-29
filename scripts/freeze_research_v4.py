#!/usr/bin/env python3
"""Freeze the three new experiments while preserving the previous snapshot."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/research_v4_20261029'
MANIFEST = RESULTS / 'manifest.json'


def record(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def previous_state():
    subprocess.run([sys.executable, str(ROOT / 'scripts/freeze_research_v3.py'), '--verify'], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()
    previous_state()
    if args.verify:
        data = json.loads(MANIFEST.read_text())
        for group in ('files', 'binaries', 'anchors'):
            for name, expected in data[group].items():
                assert record(ROOT / name) == expected, name
        print(json.dumps({'status': 'PASS', 'files': len(data['files']), 'binaries': len(data['binaries']), 'previous_v3_state': 'PASS'}))
        return
    assert not MANIFEST.exists(), 'Do not overwrite a frozen manifest'
    paths = {p for p in RESULTS.rglob('*') if p.is_file()}
    for pattern in ('metal/*gather_v8*', 'metal/*gather_exchange_v9*', 'metal/*gqa_v10*',
                    'configs/gather_v8*', 'configs/gather_exchange_v9*', 'configs/gqa_v10*',
                    'scripts/run_gather_v8.py', 'scripts/run_gather_exchange_v9.py', 'scripts/run_gqa_v10.py',
                    'scripts/verify_research_v4.py', 'scripts/freeze_research_v4.py',
                    'tests/test_gather*v8.py', 'tests/test_gather*v9.py', 'tests/test_gqa*v10.py',
                    'docs/gather_prior_art_v8.md', 'docs/gather_v8_design.md', 'docs/gather_exchange_v9_design.md',
                    'docs/gqa_v10_design.md', 'docs/research_v4_report.md'):
        paths.update(p for p in ROOT.glob(pattern) if p.is_file())
    binaries = [ROOT / f'build/metal-{name}/astc-kv-probe-{name}' for name in ('gather-v8', 'gather-exchange-v9', 'gqa-v10')]
    data = {'schema': 'astckv.research.v4.manifest', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'files': {str(p.relative_to(ROOT)): record(p) for p in sorted(paths)},
            'binaries': {str(p.relative_to(ROOT)): record(p) for p in binaries},
            'anchors': {'results/research_v3_20261029/manifest.json': record(ROOT / 'results/research_v3_20261029/manifest.json')}}
    with MANIFEST.open('x') as f:
        json.dump(data, f, indent=2); f.write('\n')
    print(json.dumps({'status': 'CREATED', 'files': len(data['files']), 'binaries': len(data['binaries'])}))


if __name__ == '__main__':
    main()

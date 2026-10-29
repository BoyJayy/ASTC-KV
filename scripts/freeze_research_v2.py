#!/usr/bin/env python3
"""Versioned local evidence manifest. Never overwrite previous freezes."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'results/kernel_v2/research_v2_manifest.json'


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--verify',action='store_true')
    args = ap.parse_args()
    if args.verify:
        record = json.loads(MANIFEST.read_text())
        for item in record['files']:
            p = ROOT/item['path']
            assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256'], item['path']
        print(json.dumps({'status':'PASS','files':len(record['files']),'bytes':sum(f['bytes'] for f in record['files'])}))
        return
    files = {ROOT/'README.md',Path(__file__)}
    for name in ['astc_kv','scripts','tests','configs','docs','metal','baselines/llama','fixtures/geometry_holdout_v2']:
        for p in (ROOT/name).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.mm','.metal','.cpp','.h','.sh','.json','.md','.txt']:
                files.add(p)
    for name in ['kernel_v2','block_geometry_v2','geometry_holdout_v2']:
        files.update(p for p in (ROOT/'results'/name).rglob('*') if p.is_file() and p!=MANIFEST and '__pycache__' not in p.parts)
    record = {'schema':'astckv.research_manifest.v2','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope':'Current local source/docs and new v2 evidence; original results and archive are anchored separately, not overwritten.',
              'previous_manifest_sha256':digest(ROOT/'results/revalidation_20261028/continuation_manifest.json'),
              'original_archive_sha256':digest(Path('/home/researcher/Downloads/ASTC-KV_v0.1.zip')),
              'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(files)],
              'local_binary_hashes':{}}
    for name in ['metal/astc-kv-probe','metal-v2/astc-kv-probe-v2','metal-v3/astc-kv-probe-v3',
                 'metal-v4/astc-kv-probe-v4','metal-pool-v5/astc-kv-probe-pool-v5',
                 'metal-atlas-alloc/astc-kv-atlas-alloc','metal-atlas-v6/astc-kv-probe-atlas-v6',
                 'llama-baseline/bin/astc-llama-baseline','llama-baseline/bin/astc-llama-precision']:
        p = ROOT/'build'/name
        if p.is_file():record['local_binary_hashes'][str(p.relative_to(ROOT))]=digest(p)
    assert record['original_archive_sha256']=='13ddaaa57a0b2c81457155aeece426e1977c3bc3a805bca0f8f0988d77e60b95'
    with MANIFEST.open('x') as out:
        json.dump(record,out,indent=2)
        out.write('\n')
    print(json.dumps({'status':'frozen','files':len(record['files']),'manifest':str(MANIFEST)}))


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Freeze the v3 evidence and current source without altering older manifests."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/research_v3_20261029/manifest.json'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--verify',action='store_true')
    args=ap.parse_args()
    if args.verify:
        record=json.loads(OUT.read_text())
        for row in record['files']:
            p=ROOT/row['path']
            assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
        for path,digest in record['binaries'].items():assert sha(ROOT/path)==digest,path
        assert sha(ROOT/'results/kernel_v2/research_v2_manifest.json')==record['v2_manifest_sha256']
        print(json.dumps({'status':'PASS','files':len(record['files']),'binaries':len(record['binaries'])}))
        return
    files={ROOT/'README.md',ROOT/'AGENTS.md'}
    extensions={'.py','.mm','.metal','.sh','.cpp','.h','.json','.md','.txt'}
    for folder in ['astc_kv','scripts','tests','configs','docs','metal','baselines/llama']:
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in extensions and '__pycache__' not in p.parts)
    files.update(p for p in OUT.parent.rglob('*') if p.is_file() and p!=OUT and '__pycache__' not in p.parts)
    binaries=[ROOT/'build/metal-atlas-v7/astc-kv-probe-atlas-v7',ROOT/'build/metal-address-abba-v7/astc-kv-probe-address-abba-v7']
    record={'schema':'astckv.research_manifest.v3','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope':'Current source/reports and all new v3 evidence. Original v2 evidence is linked by its unchanged manifest and verified through report backups.',
            'v2_manifest_sha256':sha(ROOT/'results/kernel_v2/research_v2_manifest.json'),
            'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)],
            'binaries':{str(p.relative_to(ROOT)):sha(p) for p in binaries}}
    with OUT.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps({'status':'frozen','files':len(record['files'])}))


if __name__=='__main__':main()

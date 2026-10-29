#!/usr/bin/env python3
"""Fetch one pinned public llama.cpp revision into ignored third_party/."""
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REV='c3df4b7e22a1ac38da663552d19e649dd97ca936'
BASE=ROOT/'third_party'
URL=f'https://github.com/ggml-org/llama.cpp/archive/{REV}.tar.gz'


def main():
    archive=BASE/f'llama-{REV}.tar.gz';dest=BASE/f'llama.cpp-{REV}'
    if dest.exists():raise RuntimeError('Source directory exists; preserve it')
    if not archive.exists():
        temp=archive.with_suffix('.partial')
        with urllib.request.urlopen(URL,timeout=60) as r,temp.open('xb') as f:
            total=0
            while chunk:=r.read(1<<20):
                total+=len(chunk)
                if total>100_000_000:raise RuntimeError('Archive exceeds bounded download budget')
                f.write(chunk)
        temp.replace(archive)
    with tarfile.open(archive) as t:
        members=t.getmembers()
        if len(members)>30000 or sum(m.size for m in members)>500_000_000:raise RuntimeError('Unpack budget exceeded')
        for m in members:
            p=Path(m.name)
            if p.is_absolute() or '..' in p.parts or p.parts[0]!=dest.name or not(m.isfile() or m.isdir()):
                raise RuntimeError('Unexpected archive entry')
        t.extractall(BASE,filter='data')
    record={'revision':REV,'url':URL,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'archive_bytes':archive.stat().st_size,'source_dir':str(dest),
            'scope':'Pinned public source downloaded; compiled behavior still requires validation'}
    (BASE/'llama_baseline_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()

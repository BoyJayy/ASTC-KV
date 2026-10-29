#!/usr/bin/env python3
"""Fetch immutable public inputs into this project; verify SHA256."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,tarfile,os

ROOT=Path(__file__).resolve().parents[1]
REV='93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
FILES={
 'config.json':'1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843',
 'model.safetensors':'80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1',
 'tokenizer.json':'9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c',
 'tokenizer_config.json':'4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424',
 'special_tokens_map.json':'e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3'}

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def fetch(url,p,expected):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and digest(p)==expected:return
 tmp=p.with_suffix(p.suffix+'.partial')
 with urllib.request.urlopen(url,timeout=120) as r,tmp.open('wb') as f:
  for b in iter(lambda:r.read(4*1024*1024),b''):f.write(b)
 actual=digest(tmp)
 if actual!=expected:raise RuntimeError(f'SHA256 mismatch for {p.name}: {actual}; leaving partial file for inspection')
 os.replace(tmp,p)
 print(p.relative_to(ROOT),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--skip-model',action='store_true');args=ap.parse_args()
 if not args.skip_model:
  for n,s in FILES.items():fetch(f'https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/{REV}/{n}',ROOT/'third_party/model'/n,s)
 src=ROOT/'third_party/astc-source.tar.gz'
 fetch('https://github.com/ARM-software/astc-encoder/archive/refs/tags/5.7.0.tar.gz',src,'7c1b28ece59c9c2737e297123a9910c1c548676c7aed87a09d5734e2fa7bdaf0')
 if not (ROOT/'third_party/astc-encoder-5.7.0/Source/astcenc.h').exists():
  with tarfile.open(src) as f:f.extractall(ROOT/'third_party',filter='data')
 (ROOT/'third_party/model/REVISION').write_text(REV+'\n') if not args.skip_model else None

if __name__=='__main__':main()

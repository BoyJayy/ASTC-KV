#!/usr/bin/env python3
"""Repeat the existing V-only fixture for bounded traffic controls, never quality."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def repeat_axis(a,n,axis):
    indices=np.arange(n)%a.shape[axis]
    return np.take(a,indices,axis=axis)


def make_fixture(source:Path,out:Path,tokens:int):
    if not 1<=tokens<=8192: raise ValueError('Traffic control bounded to model context limit 8192')
    manifest=json.loads((source/'manifest.json').read_text())
    if (manifest['k_storage'],manifest['v_storage'],manifest['v_normalization'])!=('fp16','astc','token'):
        raise ValueError('Only the frozen V-only token-normalized source is supported')
    out.mkdir(parents=True,exist_ok=False)
    h,t,d=(manifest[k] for k in ['kv_heads','tokens','head_dim'])
    provenance={}
    for field,dtype,shape,axis in [('k_file','<f2',(h,t,d),1),('k_decoded_file','<f4',(h,t,d),1),
                                  ('v_decoded_file','<f4',(h,t,d),1),('v_offset_file','<f2',(h,t,1),1),
                                  ('v_scale_file','<f2',(h,t,1),1)]:
        p=source/manifest[field];a=np.fromfile(p,dtype=dtype).reshape(shape)
        repeat_axis(a,tokens,axis).tofile(out/manifest[field])
        provenance[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    qp=source/manifest['query_file'];(out/qp.name).write_bytes(qp.read_bytes())
    provenance[qp.name]=hashlib.sha256(qp.read_bytes()).hexdigest()
    for name in manifest['v_files']:
        p=source/name;raw=p.read_bytes();header=bytearray(raw[:16])
        assert int.from_bytes(header[:4],'little')==0x5ca1ab13 and header[4:7]==bytes([6,6,1])
        width=int.from_bytes(header[7:10],'little');height=int.from_bytes(header[10:13],'little')
        assert height==t and height%6==0
        row_bytes=((width+5)//6)*16
        rows=np.frombuffer(raw[16:],np.uint8).reshape(height//6,row_bytes)
        page=manifest['page_tokens']
        padded=(tokens+page-1)//page*page;header[10:13]=padded.to_bytes(3,'little')
        (out/name).write_bytes(header+repeat_axis(rows,padded//6,0).tobytes())
        provenance[name]=hashlib.sha256(raw).hexdigest()
    manifest.update(tokens=tokens,status='synthetic repeated-real-block traffic control; NOT model-quality evidence',
                    source_tokens=t,traffic_source_sha256=provenance,query_absolute_position=None)
    manifest.pop('logical_cache_bytes',None);manifest.pop('logical_bits_per_original_kv_scalar',None)
    manifest['traffic_note']='Same query, repeated old K/V. Padding texels repeat known encoded block rows. No fresh long-context forward.'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return out/'manifest.json'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'results/metal_fixtures/story_l17_v_token6')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--tokens',type=int,required=True)
    a=p.parse_args();print(make_fixture(a.source,a.out,a.tokens))

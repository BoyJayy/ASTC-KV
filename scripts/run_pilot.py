#!/usr/bin/env python3
"""Run preregistered snapshot quality/memory sweep; CPU only, no speed claims."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from astc_kv.codec import encode_tensor, affine_quantize, attention, attention_metrics, errors


def source_hash(root):
    h=hashlib.sha256()
    for p in sorted(list((root/'astc_kv').glob('*.py'))+list((root/'scripts').glob('run_pilot.py'))):
        h.update(p.relative_to(root).as_posix().encode());h.update(p.read_bytes())
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--capture',required=True,type=Path)
    ap.add_argument('--config',default='configs/pilot_v1.json',type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--representations',default='all')
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads(args.config.read_text())
    data=np.load(args.capture)
    q,k,v=[data[n] for n in ('q','k','v')]
    layers=data['layer_ids']
    positions=np.array(cfg['snapshot_query_positions'])
    prefix=cfg['compressed_prefix_tokens'];page=cfg['page_tokens']
    assert list(layers)==cfg['layers']
    assert k.shape[-2]==cfg['context_tokens'] and positions.min()>=prefix
    reps=cfg['representations']
    if args.representations!='all':
        names=args.representations.split(',');reps=[r for r in reps if r['name'] in names]
    variants=[dict(name='fp16',baseline=16)] + [dict(name=f'u{b}_token',baseline=b) for b in (2,4,8)]
    variants += [dict(name='u4_asymmetric',baseline=4,asymmetric=True)] + reps
    args.out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    with args.out.open('w') as f:
        for variant in variants:
            started=time.perf_counter()
            for li,layer in enumerate(layers):
                original_k,original_v=k[li],v[li]
                ref,pref=attention(q[li][:,positions],original_k,original_v,positions)
                basek,basev=[a.astype(np.float16).astype(np.float32) for a in (original_k,original_v)]
                decoded={}; sizes={}; encode_seconds=0.0
                for name,orig in [('k',original_k),('v',original_v)]:
                    part=orig[:,:prefix,:]
                    if variant.get('baseline')==16:
                        reconstructed=part.astype(np.float16).astype(np.float32)
                        sizes[name]={'fp16_payload':part.size*2}
                    elif variant.get('baseline'):
                        axis='channel' if variant.get('asymmetric') and name=='k' else 'token'
                        group=page if axis=='channel' else part.shape[-1]
                        reconstructed,counts=affine_quantize(part,variant['baseline'],axis,group)
                        sizes[name]=counts
                    else:
                        enc=encode_tensor(part,variant,page,cfg['encoder']['quality'],cfg['random_seed'])
                        reconstructed=enc.decoded
                        sizes[name]=enc.byte_counts
                        encode_seconds+=enc.encode_cpu_s
                    decoded[name]=reconstructed
                for mode in cfg['compression_modes']:
                    ck,cv=basek.copy(),basev.copy()
                    compressed_k=mode in ('K_only','KV')
                    compressed_v=mode in ('V_only','KV')
                    if compressed_k:ck[:,:prefix]=decoded['k']
                    if compressed_v:cv[:,:prefix]=decoded['v']
                    out,p=attention(q[li][:,positions],ck,cv,positions)
                    old_count=original_k[:,:prefix].size
                    b_k=sum(sizes['k'].values()) if compressed_k else old_count*2
                    b_v=sum(sizes['v'].values()) if compressed_v else old_count*2
                    tail_bytes=2*(original_k.size-old_count)*2
                    metrics=attention_metrics(ref,out,pref,p)
                    per_head=[attention_metrics(ref[h],out[h],pref[h],p[h]) for h in range(ref.shape[0])]
                    per_kv_head={n:[errors(a[h,:prefix],decoded[n][h]) for h in range(a.shape[0])] for n,a in [('k',original_k),('v',original_v)]}
                    row={'schema':'astc-kv-snapshot-result-v1','capture':args.capture.stem,'layer':int(layer),
                         'variant':variant['name'],'mode':mode,'prefix_tokens':prefix,'total_tokens':int(original_k.shape[1]),
                         'query_positions':positions.tolist(),'metrics':metrics,'per_query_head':per_head,
                         'tensor_error_k':errors(original_k[:,:prefix],decoded['k']),
                         'tensor_error_v':errors(original_v[:,:prefix],decoded['v']),
                         'per_kv_head':per_kv_head,
                         'memory':{'compressed_prefix_k':sizes['k'],'compressed_prefix_v':sizes['v'],
                                   'policy_prefix_k_bytes':b_k,'policy_prefix_v_bytes':b_v,'fp16_tail_bytes':tail_bytes,
                                   'payload_metadata_bytes':b_k+b_v+tail_bytes,
                                   'effective_bits_per_original_kv_scalar':8*(b_k+b_v+tail_bytes)/(2*original_k.size),
                                   'k_prefix_effective_bits_per_scalar':8*sum(sizes['k'].values())/old_count,
                                   'v_prefix_effective_bits_per_scalar':8*sum(sizes['v'].values())/old_count,
                                   'resource_allocation_bytes':None,'gpu_peak_bytes':None,
                                   'scope':'logical payload+rounded metadata+fixed permutation+tail; runtime resource descriptors/alignment unknown'},
                         'error_sum_squares':float(np.sum((out.astype('f8')-ref)**2)),
                         'reference_sum_squares':float(np.sum(ref.astype('f8')**2)),
                         'encode_cpu_s':encode_seconds,'gpu_executed':False}
                    f.write(json.dumps(row)+'\n');f.flush();rows.append(row)
            print(json.dumps({'capture':args.capture.stem,'variant':variant['name'],'wall_s':time.perf_counter()-started}),flush=True)
    provenance={'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
                'capture_sha256':hashlib.sha256(args.capture.read_bytes()).hexdigest(),
                'source_sha256':source_hash(root),'rows':len(rows),'numpy':np.__version__,
                'measurement':'CPU numerical roundtrip + snapshot attention','hardware_claim':False}
    args.out.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Post-pilot fixed U4 K + ASTC V experiment; CPU quality, not a runtime."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from astc_kv.codec import affine_quantize,encode_tensor
from capture_qkv import NumpyLlama
from evaluate_model import run_teacher_forced


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-prefix',type=Path,required=True)
    ap.add_argument('--primary',type=Path,required=True)
    ap.add_argument('--secondary',type=Path,required=True)
    args=ap.parse_args()
    output=args.out_prefix.with_suffix('.json');raw_path=args.out_prefix.with_suffix('.jsonl')
    if output.exists() or raw_path.exists():raise RuntimeError('Choose new output paths')
    primary=json.loads(args.primary.read_text());secondary=json.loads(args.secondary.read_text())
    assert primary['status']==secondary['status']=='completed'
    config_path=ROOT/'configs/hybrid_u4k_astcv_20261028.json';cfg=json.loads(config_path.read_text())
    weights=ROOT/'third_party/model/model.safetensors'
    assert sha(weights)==primary['model_weights_sha256']
    source_paths=[Path(__file__),ROOT/'scripts/capture_qkv.py',ROOT/'scripts/evaluate_model.py',ROOT/'astc_kv/codec.py',ROOT/'astc_kv/bridge.cpp',ROOT/'astc_kv/libakv_astc.dylib']
    result={'schema':'astckv.hybrid.v1','status':'running','decision':cfg,'config_sha256':sha(config_path),
            'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in source_paths},
            'weight_sha256':sha(weights),'primary_sha256':sha(args.primary),'secondary_sha256':sha(args.secondary),
            'numpy':np.__version__,'fixtures':[],'scope':'Exploratory reused pilot documents; CPU reconstruction, no GPU runtime'}
    model=NumpyLlama(ROOT/'third_party/model');all_losses=[]
    with raw_path.open('x') as raw:
        for fixture in primary['fixtures']:
            name=fixture['fixture'];cp=ROOT/'results/captures'/f'{name}.npz'
            assert sha(cp)==fixture['capture_sha256']
            with np.load(cp,allow_pickle=False) as data:ids=data['token_ids']
            assert ids.tolist()==fixture['input_token_ids']
            _,original=model.prefill(ids[:cfg['prefix_tokens']]);cache=[];counts={'k':0,'v':0}
            for k,v in original:
                kq,kb=affine_quantize(k,4,axis='channel',group=96)
                ve=encode_tensor(v,cfg['V'],96,98)
                cache.append((kq,ve.decoded));counts['k']+=sum(kb.values());counts['v']+=ve.logical_bytes
            losses=run_teacher_forced(model,ids,cache,288)
            assert len(losses)==95 and all(math.isfinite(r['nll']) for r in losses)
            for row in losses:raw.write(json.dumps({'fixture':name,'method':'u4K_astc6tokenV',**row})+'\n')
            raw.flush();all_losses.extend(r['nll'] for r in losses)
            mean=math.fsum(r['nll'] for r in losses)/len(losses)
            tail=30*2*3*95*64*2;logical=sum(counts.values())+tail
            baseline=fixture['methods']['fp16']['mean_nll']
            row={'fixture':name,'targets':95,'mean_nll':mean,'relative_ppl_vs_fp16':math.exp(mean-baseline),
                 'logical_bytes_at_last_prediction':logical,'prefix_bytes':counts,'fp16_tail_bytes':tail,
                 'bits_per_combined_kv':logical*8/(30*2*3*383*64),'capture_sha256':sha(cp)}
            result['fixtures'].append(row);output.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(row),flush=True)
    mean=math.fsum(all_losses)/len(all_losses);base=primary['aggregate']['fp16']['mean_nll']
    u4=secondary['aggregate']['mean_nll'];u4bytes=secondary['fixtures'][0]['logical_bytes_at_last_prediction']
    result.update(status='completed',raw_sha256=sha(raw_path),aggregate={
        'targets':len(all_losses),'mean_nll':mean,'relative_ppl_vs_fp16':math.exp(mean-base),
        'relative_ppl_vs_asymmetric_u4':math.exp(mean-u4),'logical_bytes':logical,
        'logical_saving_vs_asymmetric_u4_fraction':1-logical/u4bytes,'pilot_ppl_screen_pass':math.exp(mean-base)<=1.02})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['aggregate']),flush=True)


if __name__=='__main__':main()

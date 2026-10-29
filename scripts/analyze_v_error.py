#!/usr/bin/env python3
"""Frozen H2a diagnostic; all data reused from the pilot, no new model claim."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from astc_kv.codec import encode_tensor, affine_quantize


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def probabilities(q, k, positions):
    """Separate FP64 oracle, explicit GQA and causal masking."""
    hq, _, d = q.shape
    mapped = k[np.arange(hq)//(hq//len(k))].astype('f8')
    logits = np.einsum('hqd,htd->hqt',q.astype('f8'),mapped)/np.sqrt(d)
    logits = np.where(np.arange(k.shape[1])[None,None,:] <= positions[None,:,None],logits,-np.inf)
    p = np.exp(logits-logits.max(-1,keepdims=True))
    return p/p.sum(-1,keepdims=True)


def weighted(p, values):
    return np.einsum('hqt,htd->hqd',p,values[np.arange(len(p))//(len(p)//len(values))].astype('f8'))


def sse(x):
    return float(np.sum(np.square(x,dtype='f8')))


def error_diagnostic(p, error, page):
    h,t,d = error.shape
    bias = np.repeat(error.reshape(h,t//page,page,d).mean(2),page,axis=1)
    pe,pb,pr = (weighted(p,a) for a in (error,bias,error-bias))
    total, bs, rs = sse(pe),sse(pb),sse(pr)
    cross = float(2*np.sum(pb*pr))
    assert np.isclose(total,bs+rs+cross,rtol=1e-11,atol=1e-12)
    mapping = np.arange(len(p))//(len(p)//h)
    diagonal = float(np.sum(p*p * np.sum(error*error,axis=-1)[mapping,None,:]))
    correction = bias.reshape(h,t//page,page,d)[:,:,0,:].astype('f2')
    corrected_error = error-np.repeat(correction.astype('f8'),page,axis=1)
    return {'v_error_sse':sse(error),'page_bias_fraction_of_v_error_sse':sse(bias)/sse(error) if sse(error) else None,
            'attention_error_sse':total,'bias_output_sse':bs,'residual_output_sse':rs,'cross_term':cross,
            'sign_noise_expected_sse':diagonal,'cross_token_excess_sse':total-diagonal,
            'original_over_sign_expectation':total/diagonal if diagonal else None,
            'fp16_page_corrected_sse':sse(weighted(p,corrected_error)),
            'correction_bytes':int(correction.nbytes)}, correction


def randomized_controls(p,error,page,rng,trials=256):
    sign_results,permutation_results = [],[]
    h,t,d=error.shape
    for _ in range(trials):
        signed=error*rng.choice([-1.,1.],size=(h,t,1))
        perm=error.reshape(h,t//page,page,d).copy()
        for head in range(h):
            for pg in range(t//page):
                perm[head,pg]=perm[head,pg,rng.permutation(page)]
        # Both controls preserve total SSE; signs preserve every row norm too.
        assert np.isclose(sse(perm),sse(error),rtol=1e-13)
        sign_results.append(sse(weighted(p,signed)))
        permutation_results.append(sse(weighted(p,perm.reshape(h,t,d))))
    return {'sign_sse':sign_results,'within_page_permutation_sse':permutation_results}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--all-captures',action='store_true')
    ap.add_argument('--reconstructions-dir',type=Path,help='Reuse saved actual codec outputs instead of encoding again')
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    output=args.out_dir/'analysis.json'
    if output.exists():
        raise RuntimeError('Preserve old results: choose a new output directory')
    config_path=ROOT/'configs/revalidation_h2a_20261028.json'
    cfg=json.loads(config_path.read_text())
    source_sha=digest(Path(__file__))
    codec_sha=digest(ROOT/'astc_kv/codec.py')
    bridge_sha=digest(ROOT/'astc_kv/bridge.cpp')
    library=ROOT/'astc_kv'/('libakv_astc.dylib' if sys.platform=='darwin' else 'libakv_astc.so')
    page,prefix=cfg['page_tokens'],cfg['prefix_tokens']
    positions=np.array(cfg['query_positions'])
    rows=[]; controls={}; inputs={}; rng=np.random.default_rng(cfg['seed'])
    names=['systems','story','science','records'] if args.all_captures else ['story']
    for name in names:
        cp=ROOT/'results/captures'/f'{name}.npz'; inputs[str(cp.relative_to(ROOT))]=digest(cp)
        with np.load(cp,allow_pickle=False) as capture:
            for li,layer in enumerate(capture['layer_ids']):
                if not args.all_captures and layer!=17:
                    continue
                q=capture['q'][li][:,positions]; k=capture['k'][li]; v=capture['v'][li]
                p=probabilities(q,k,positions); ref=weighted(p,v); reference_sse=sse(ref)
                pk16=probabilities(q,k.astype('f2').astype('f4'),positions)
                for variant,normal in [('scalar6_token','token'),('scalar6_page','page_channel'),('u4_token',None)]:
                    if normal is None:
                        decoded,counts=affine_quantize(v[:,:prefix],4)
                        byte_count=sum(counts.values())
                    elif not args.all_captures:
                        fixture='story_l17_v_token6' if normal=='token' else 'story_l17_kv_page6'
                        path=ROOT/'results/metal_fixtures'/fixture/'v_decoded.f32'
                        inputs[str(path.relative_to(ROOT))]=digest(path)
                        decoded=np.fromfile(path,dtype='<f4').reshape(3,384,64)[:,:prefix]
                        byte_count=int(v[:,:prefix].size*(25/6 if normal=='token' else 4)/8)
                    elif args.reconstructions_dir:
                        path=args.reconstructions_dir/f'{name}_l{int(layer)}_{variant}.npz'
                        inputs[str(path)]=digest(path)
                        with np.load(path,allow_pickle=False) as rec:
                            decoded=rec['decoded']
                        assert decoded.shape==v[:,:prefix].shape and np.isfinite(decoded).all()
                        byte_count=int(v[:,:prefix].size*(25/6 if normal=='token' else 4)/8)
                    else:
                        encoded=encode_tensor(v[:,:prefix],{'layout':'scalar','block':6,'normalization':normal},page,98)
                        decoded=encoded.decoded; byte_count=encoded.logical_bytes
                        path=args.out_dir/f'{name}_l{int(layer)}_{variant}.npz'
                        np.savez_compressed(path,decoded=decoded)
                        inputs[str(path)]=digest(path)
                    error=decoded.astype('f8')-v[:,:prefix].astype('f8')
                    metrics,correction=error_diagnostic(p[:,:,:prefix],error,page)
                    # Exact V-only identity, independently checked by recomputing
                    # the full attention with unchanged K and unchanged tail.
                    changed=v.astype('f8').copy(); changed[:,:prefix]=decoded
                    np.testing.assert_allclose(weighted(p,changed)-ref,weighted(p[:,:,:prefix],error),rtol=1e-9,atol=1e-12)
                    metrics['relative_l2_V_error_only']=np.sqrt(metrics['attention_error_sse']/reference_sse)
                    metrics['relative_l2_V_error_only_corrected']=np.sqrt(metrics['fp16_page_corrected_sse']/reference_sse)
                    out_metrics={}
                    head_sse={}
                    for corrected in [False,True]:
                        vv=v.astype('f2').astype('f8')
                        vv[:,:prefix]=decoded
                        if corrected:
                            vv[:,:prefix]-=np.repeat(correction.astype('f8'),page,axis=1)
                        diff=weighted(pk16,vv)-ref
                        label='corrected' if corrected else 'original'
                        out_metrics[label+'_historical_policy_sse']=sse(diff)
                        out_metrics[label+'_historical_policy_relative_l2']=np.sqrt(sse(diff)/reference_sse)
                        head_sse[label]=np.sum(diff*diff,axis=(1,2))
                    heads=[]
                    for head in range(len(q)):
                        kh=head//(len(q)//len(v))
                        hm,_=error_diagnostic(p[head:head+1,:,:prefix],error[kh:kh+1],page)
                        heads.append({'query_head':head,'kv_head':kh,**hm,
                                      'original_historical_policy_sse':float(head_sse['original'][head]),
                                      'corrected_historical_policy_sse':float(head_sse['corrected'][head]),
                                      'reference_sse':sse(ref[head])})
                    row={'capture':name,'layer':int(layer),'variant':variant,'reference_sse':reference_sse,
                         'V_prefix_bits':8*byte_count/v[:,:prefix].size,
                         'corrected_V_prefix_bits':8*(byte_count+correction.nbytes)/v[:,:prefix].size,
                         **metrics,**out_metrics,'per_query_head':heads}
                    rows.append(row)
                    if not args.all_captures:
                        controls[variant]=randomized_controls(p[:,:,:prefix],error,page,rng)
                    print(json.dumps(row),flush=True)
    aggregate=[]
    for name in names+(['pooled'] if len(names)>1 else []):
        for variant in cfg['representations']:
            rs=[r for r in rows if r['variant']==variant and (name=='pooled' or r['capture']==name)]
            den=sum(r['reference_sse'] for r in rs)
            orig=sum(r['original_historical_policy_sse'] for r in rs)
            corr=sum(r['corrected_historical_policy_sse'] for r in rs)
            aggregate.append({'capture':name,'variant':variant,'cases':len(rs),
                              'original_relative_l2':np.sqrt(orig/den),'corrected_relative_l2':np.sqrt(corr/den),
                              'SSE_reduction_fraction':1-corr/orig,'corrected_V_prefix_bits':rs[0]['corrected_V_prefix_bits'],
                              'positive_cross_token_cases':sum(r['cross_token_excess_sse']>0 for r in rs)})
    result={'schema':'astckv.h2a.v1','status':'exploratory_completed','fresh_holdout':False,
            'config_sha256':digest(config_path),'source_sha256':source_sha,'codec_source_sha256':codec_sha,
            'bridge_source_sha256':bridge_sha,'encoder':{'name':'Arm astcenc','version':'5.7.0','quality':98},
            'codec_library_sha256':digest(library) if library.exists() else None,
            'inputs':inputs,'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
            'rows':rows,'aggregate':aggregate,'randomized_controls':controls,
            'scope':'Fixed captured Q/K; correction uses completed V prefix, never future Q. No model-level or GPU runtime validation.'}
    with output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'aggregate':aggregate},indent=2))


if __name__=='__main__':
    main()

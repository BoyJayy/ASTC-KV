#!/usr/bin/env python3
"""Bounded real-GPU controls for storage policies and refusal before timing."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args()
    out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=ROOT/'build/metal-v4/astc-kv-probe-v4'; shader=ROOT/'metal/attention_v4.metal'
    source=ROOT/'results/metal_fixtures'
    from make_traffic_fixture import make_fixture
    ragged=make_fixture(source/'story_l17_v_token6',out/'ragged',391)
    cases=[('ragged',ragged,True,1,1)]
    for name,fixture in [('v_only','story_l17_v_token6'),('kv','story_l17_kv_page6')]:
        cases.append((name,source/fixture/'manifest.json',True,1,1))
    with np.load(ROOT/'results/captures/story.npz',allow_pickle=False) as capture:
        li=list(capture['layer_ids']).index(17)
        original_v=capture['v'][li].astype('f2')
    for name,fixture in [('k_only','story_l17_kv_page6'),('fp16','story_l17_v_token6'),
                         ('wrong_decode','story_l17_v_token6'),('nonfinite_decode','story_l17_v_token6'),
                         ('missing_decode','story_l17_v_token6')]:
        target=out/name;shutil.copytree(source/fixture,target)
        m=json.loads((target/'manifest.json').read_text())
        m['status']='derived validation control; not a new quality experiment'
        if name in ['k_only','fp16']:
            original_v.tofile(target/'v_fp16.f16')
            original_v.astype('f4').tofile(target/'v_fp16.f32')
            m.update(v_storage='fp16',v_file='v_fp16.f16',v_decoded_file='v_fp16.f32')
            m.pop('logical_cache_bytes',None);m.pop('logical_bits_per_original_kv_scalar',None)
        elif name=='missing_decode':
            del m['v_decoded_file']
        else:
            values=np.fromfile(target/m['v_decoded_file'],'<f4')
            values[0]=np.nan if name=='nonfinite_decode' else values[0]+1000
            values.tofile(target/m['v_decoded_file'])
        (target/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
        cases.append((name,target/'manifest.json',name in ['k_only','fp16'],1,1))
    results=[]
    for name,manifest,should_pass,rounds,iterations in cases:
        cmd=[str(binary),str(manifest),str(shader),str(rounds),str(iterations),'13']
        completed=subprocess.run(cmd,capture_output=True,text=True,timeout=60,cwd=ROOT)
        (out/(name+'.json')).write_text(completed.stdout)
        (out/(name+'.stderr')).write_text(completed.stderr)
        parsed=json.loads(completed.stdout)
        if should_pass:
            assert parsed['attribution_controls']['factored_v_applied'] == (name in ['v_only','ragged'])
            assert len(parsed['correctness']) >= 7
        passed=completed.returncode==0 if should_pass else completed.returncode!=0 and parsed.get('status')=='correctness_failed' and parsed.get('raw_samples')==[]
        results.append({'case':name,'command':cmd,'expected_success':should_pass,
                        'exit_code':completed.returncode,'passed':passed,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()})
        if not passed:raise RuntimeError(f'Unexpected outcome in {name}')
    result={'schema':'astckv.metal.controls.v4','cases':results,'passed':all(r['passed'] for r in results),
            'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
            'host_source_sha256':hashlib.sha256((ROOT/'metal/probe_v4.mm').read_bytes()).hexdigest(),
            'shader_sha256':hashlib.sha256(shader.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Two real fixtures, two derived storage controls, three refusal tests; not all shapes/formats or model inference'}
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

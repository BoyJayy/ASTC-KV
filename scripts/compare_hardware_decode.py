#!/usr/bin/env python3
"""Compare actual GPU readback with software decode of identical ASTC blocks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from astc_kv.codec import AstcEncoder, errors
from scripts.analyze_v_error import probabilities, weighted, sse


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    records=[]; inputs={}
    capture_path=ROOT/'results/captures/story.npz'
    inputs[str(capture_path.relative_to(ROOT))]=sha(capture_path)
    with np.load(capture_path,allow_pickle=False) as data:
        li=list(data['layer_ids']).index(17)
        q,k,v=(data[n][li].copy() for n in ('q','k','v'))
    positions=np.array([288,294,300,306,312,318,324,330,336,342,348,354,360,366,372,383])
    ref=weighted(probabilities(q[:,positions],k,positions),v)
    for short,fixture in [('v','story_l17_v_token6'),('kv','story_l17_kv_page6')]:
        directory=ROOT/'results/metal_fixtures'/fixture
        readback=ROOT/'results/revalidation_20261028/metal'/('readback_'+short)
        manifest=json.loads((directory/'manifest.json').read_text())
        inputs[str((directory/'manifest.json').relative_to(ROOT))]=sha(directory/'manifest.json')
        sw_tensors={};gpu_tensors={};checks={}
        for side in ['k','v']:
            sw_path=directory/manifest[side+'_decoded_file']
            hw_path=readback/(side+'_hardware.f32')
            inputs[str(sw_path.relative_to(ROOT))]=sha(sw_path)
            inputs[str(hw_path.relative_to(ROOT))]=sha(hw_path)
            sw_tensors[side]=np.fromfile(sw_path,'<f4').reshape(3,384,64)
            gpu_tensors[side]=np.fromfile(hw_path,'<f4').reshape(3,384,64)
            if manifest[side+'_storage']!='astc': continue
            hardware_path=readback/(side+'_normalized_hardware.f32')
            inputs[str(hardware_path.relative_to(ROOT))]=sha(hardware_path)
            hw=np.fromfile(hardware_path,'<f4').reshape(3,384,64)
            normalized=[]
            encoder=AstcEncoder(6,98)
            try:
                for filename in manifest[side+'_files']:
                    path=directory/filename;inputs[str(path.relative_to(ROOT))]=sha(path)
                    blocks=np.frombuffer(path.read_bytes()[16:],np.uint8).copy()
                    decoded=np.empty((384,66,4),np.float32)
                    encoder._check(encoder.lib.akv_decode(encoder.ctx,blocks.ctypes.data,blocks.size,decoded.ctypes.data,66,384))
                    normalized.append(decoded[:,:64,0])
            finally: encoder.close()
            sw=np.stack(normalized)
            nearest=sw.astype('f2').astype('f4')
            floor=np.where(nearest>sw,np.nextafter(nearest.astype('f2'),np.float16(-np.inf)).astype('f4'),nearest)
            controls={'fp32_software':sw,'fp16_nearest':nearest,'fp16_floor':floor,'unorm8_nearest':np.rint(sw*255)/255}
            checks[side]={name:{**errors(hw,pred),'exact_fraction':float(np.mean(hw==pred))} for name,pred in controls.items()}
        # Hardware fixtures encode all 384; isolate first three completed pages
        # and restore the same FP16 K/tail policy as the numerical pilot.
        outputs={}
        for source,tensors in [('software',sw_tensors),('hardware',gpu_tensors)]:
            kk=k.astype('f2').astype('f4');vv=v.astype('f2').astype('f4')
            for side,dest in [('k',kk),('v',vv)]:
                if manifest[side+'_storage']=='astc':dest[:,:288]=tensors[side][:,:288]
            outputs[source]=weighted(probabilities(q[:,positions],kk,positions),vv)
        records.append({'fixture':fixture,'normalized_decode_controls':checks,
                        'software_snapshot_relative_l2':np.sqrt(sse(outputs['software']-ref)/sse(ref)),
                        'hardware_snapshot_relative_l2':np.sqrt(sse(outputs['hardware']-ref)/sse(ref)),
                        'hardware_vs_software_output':errors(outputs['software'],outputs['hardware'])})
    result={'schema':'astckv.hardware_decode_comparison.v1','source_sha256':sha(Path(__file__)),
            'inputs':inputs,'results':records,
            'scope':'Exploratory decoder characterization, two real fixtures and one stored layer. No full GPU model run or universal format guarantee.'}
    with args.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result['results'],indent=2))


if __name__=='__main__': main()

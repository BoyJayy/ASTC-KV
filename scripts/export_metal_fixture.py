#!/usr/bin/env python3
"""Export genuine captured tensors to the unexecuted Metal probe interchange.

This is a fully precompressed control. It does not implement dynamic appending.
Incomplete edge pages are padded for an offline texture, never future tokens.
"""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from astc_kv.codec import encode_tensor,align

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--capture',type=Path,required=True)
 ap.add_argument('--layer',type=int,default=17)
 ap.add_argument('--tokens',type=int,default=384)
 ap.add_argument('--block',type=int,choices=[4,6,8],default=6)
 ap.add_argument('--layout',choices=['scalar','rgba'],default='scalar')
 ap.add_argument('--normalization',choices=['token','page_channel'],default='token')
 ap.add_argument('--k-storage',choices=['fp16','astc'],default='fp16')
 ap.add_argument('--v-storage',choices=['fp16','astc'],default='astc')
 ap.add_argument('--out',type=Path,required=True)
 args=ap.parse_args()
 if args.k_storage==args.v_storage=='fp16':raise ValueError('At least one side must use ASTC for this probe')
 z=np.load(args.capture);li=list(z['layer_ids']).index(args.layer)
 t=args.tokens;page=96
 if not 1<=t<=z['k'].shape[2]:raise ValueError('tokens exceeds authentic capture')
 args.out.mkdir(parents=True,exist_ok=True)
 q=z['q'][li,:,t-1].astype('<f4');q.tofile(args.out/'q.f32')
 manifest={'format':'astc-kv-probe-v1','tokens':t,'query_heads':int(q.shape[0]),
           'kv_heads':int(z['k'].shape[1]),'head_dim':int(q.shape[-1]),'page_tokens':page,
           'layout':args.layout,'metadata_dtype':'float16_le','query_file':'q.f32',
           'model':'HuggingFaceTB/SmolLM2-135M','model_revision':'93efa2f097d58c2a74874c7e644dbc9b0cee75a2',
           'layer':args.layer,'query_absolute_position':t-1,'capture_sha256':hashlib.sha256(args.capture.read_bytes()).hexdigest(),
           'status':'offline_fixture_for_precompressed_gpu_control; Metal execution pending',
           'encoder':{'name':'Arm astcenc','version':'5.7.0','quality':98,'profile':'LDR','input_decode':'float32 library'},
           'logical_cache_bytes':{},'file_header_bytes':0}
 for name,storage in [('k',args.k_storage),('v',args.v_storage)]:
  x=z[name][li,:,:t].astype(np.float32)
  manifest[name+'_storage']=storage
  if storage=='fp16':
   filename=name+'.f16';x.astype('<f2').tofile(args.out/filename)
   manifest[name+'_file']=filename
   decoded=x.astype(np.float16).astype(np.float32)
   manifest['logical_cache_bytes'][name]={'fp16_payload':x.size*2}
  else:
   padded=np.pad(x,((0,0),(0,align(t,page)-t),(0,0)),mode='edge')
   enc=encode_tensor(padded,dict(layout=args.layout,block=args.block,normalization=args.normalization),page,98)
   manifest[name+'_files']=[]
   for h in range(x.shape[0]):
    filename=f'{name}-h{h:03d}.astc';(args.out/filename).write_bytes(enc.astc_file(h));manifest[name+'_files'].append(filename)
   manifest[name+'_normalization']=args.normalization
   offset,scale=enc.offsets,enc.scales
   if args.normalization=='token':offset,scale=offset[:,:t],scale[:,:t]
   for suffix,a in [('offset',offset),('scale',scale)]:
    filename=f'{name}_{suffix}.f16';a.astype('<f2').tofile(args.out/filename);manifest[f'{name}_{suffix}_file']=filename
   decoded=enc.decoded[:,:t]
   manifest['logical_cache_bytes'][name]={'astc_blocks_with_padding':sum(map(len,enc.payloads)),'fp16_offsets':offset.nbytes,'fp16_scales':scale.nbytes}
   manifest['file_header_bytes']+=16*x.shape[0]
  filename=name+'_decoded.f32';decoded.astype('<f4').tofile(args.out/filename);manifest[name+'_decoded_file']=filename
 manifest['logical_bits_per_original_kv_scalar']=sum(sum(b.values()) for b in manifest['logical_cache_bytes'].values())*8/(2*x.size)
 (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'manifest':str(args.out/'manifest.json'),'logical_bits_per_original_kv_scalar':manifest['logical_bits_per_original_kv_scalar'],'gpu_executed':False}))

if __name__=='__main__':main()

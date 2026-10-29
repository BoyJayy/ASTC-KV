#!/usr/bin/env python3
"""Separate balanced-order confirmation of the address-computation effect."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
from run_address_sweep import sha, environment

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    config=ROOT/'configs/address_abba_v7_20261029.json';cfg=json.loads(config.read_text())
    binary=ROOT/'build/metal-address-abba-v7/astc-kv-probe-address-abba-v7'
    shader=ROOT/'metal/attention_atlas_v7.metal'
    fixture=ROOT/'results/kernel_v2/fixtures_page/t8192'
    watched=[config,binary,shader,Path(__file__),ROOT/'scripts/run_address_sweep.py',ROOT/'metal/probe_address_abba_v7.mm',
             *sorted(p for p in fixture.iterdir() if p.is_file())]
    before={str(p.relative_to(ROOT)):sha(p) for p in watched}
    (args.out/'registration.json').write_text(json.dumps({'config':cfg,'hashes_before':before,'environment':environment()},indent=2)+'\n')
    rows=[]
    for session in range(cfg['sessions']):
        command=[str(binary),str(fixture/'manifest.json'),str(shader),str(cfg['blocks']),str(cfg['iterations']),str(cfg['tile']),'-',str(cfg['seed']+session*19)]
        env_before=environment()
        run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=120)
        stem=f's{session}'
        (args.out/(stem+'.json')).write_text(run.stdout)
        (args.out/(stem+'.stderr')).write_text(run.stderr)
        (args.out/(stem+'.execution.json')).write_text(json.dumps({'command':command,'exit':run.returncode,'environment_before':env_before,'environment_after':environment()},indent=2)+'\n')
        if run.returncode:raise RuntimeError(f'{stem} failed; output retained')
        data=json.loads(run.stdout)
        assert data['status']=='executed_on_metal'
        assert data['correctness']['compact_decode_vs_original_normalized']['passed_bit_exact_float32']
        assert all(v.get('passed_rtol_l2_1e_4',True) for v in data['correctness'].values())
        blocks={}
        for sample in data['raw_samples']:
            assert sample['iterations']==cfg['iterations'] and sample['gpu_batch_ms']>0
            blocks.setdefault((sample['pair_name'],sample['round']),[]).append(sample)
        assert len(blocks)==3*cfg['blocks']
        for (pair,round_id),samples in sorted(blocks.items()):
            assert len(samples)==4 and [s['slot_in_block'] for s in samples]==[0,1,2,3]
            a=samples[0]['a_variant'];b=samples[0]['b_variant']
            pattern=[a,b,b,a] if samples[0]['block_order']=='ABBA' else [b,a,a,b]
            assert [s['variant'] for s in samples]==pattern
            av=[s['gpu_batch_ms'] for s in samples if s['variant']==a]
            bv=[s['gpu_batch_ms'] for s in samples if s['variant']==b]
            ratio=math.sqrt(math.prod(bv)/math.prod(av))
            rows.append({'session':session,'pair':pair,'round':round_id,'block_order':samples[0]['block_order'],
                         'gpu_ratio_b_over_a':ratio,'a_us':math.sqrt(math.prod(av))/cfg['iterations']*1000,
                         'b_us':math.sqrt(math.prod(bv))/cfg['iterations']*1000,'raw':stem+'.json'})
        med=statistics.median(r['gpu_ratio_b_over_a'] for r in rows if r['session']==session and r['pair']=='v6_to_recurrent')
        print(json.dumps({'completed':stem,'primary_median_ratio':med}),flush=True)
    after={str(p.relative_to(ROOT)):sha(p) for p in watched};assert before==after
    medians=[statistics.median(r['gpu_ratio_b_over_a'] for r in rows if r['session']==s and r['pair']=='v6_to_recurrent') for s in range(cfg['sessions'])]
    (args.out/'summary.json').write_text(json.dumps({'rows':rows,'primary_session_medians':medians,
        'primary_passed':all(r<=.95 for r in medians),'hashes_after':after,'scope':cfg['scope']},indent=2)+'\n')


if __name__=='__main__':main()

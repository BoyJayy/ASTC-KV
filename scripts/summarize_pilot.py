#!/usr/bin/env python3
import argparse,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('files',nargs='+',type=Path)
ap.add_argument('--out',type=Path,default=Path('results/pilot_summary.json'))
args=ap.parse_args()
rows=[json.loads(line) for p in args.files for line in p.read_text().splitlines() if line.strip()]
groups=defaultdict(list)
for r in rows:
    split={'systems':'tune','story':'development','science':'pilot_holdout','records':'pilot_holdout'}[r['capture']]
    groups[(split,r['variant'],r['mode'])].append(r)
result=[]
for (split,variant,mode),rs in sorted(groups.items()):
    metrics=[h for r in rs for h in r['per_query_head']]
    result.append({'split':split,'variant':variant,'mode':mode,'captures':sorted(set(r['capture'] for r in rs)),
      'layers':len(rs),'query_heads':len(metrics),
      'relative_l2':math.sqrt(sum(r['error_sum_squares'] for r in rs)/sum(r['reference_sum_squares'] for r in rs)),
      'mean_kl_nats':float(np.mean([r['metrics']['mean_kl_nats'] for r in rs])),
      'head_relative_l2_median':float(np.median([r['relative_l2'] for r in metrics])),
      'head_relative_l2_p95':float(np.percentile([r['relative_l2'] for r in metrics],95)),
      'head_relative_l2_max':max(r['relative_l2'] for r in metrics),
      'head_10pct_pass_fraction':float(np.mean([r['relative_l2']<=0.10 for r in metrics])),
      'effective_bits_per_kv_scalar':float(np.mean([r['memory']['effective_bits_per_original_kv_scalar'] for r in rs])),
      'k_prefix_bits':rs[0]['memory']['k_prefix_effective_bits_per_scalar'],
      'v_prefix_bits':rs[0]['memory']['v_prefix_effective_bits_per_scalar'],
      'k_relative_l2':float(np.mean([r['tensor_error_k']['relative_l2'] for r in rs])),
      'v_relative_l2':float(np.mean([r['tensor_error_v']['relative_l2'] for r in rs]))})
base={(r['split'],r['mode']):r for r in result if r['variant']=='u4_token'}
for r in result:
    r['pilot_screen_pass']=bool(r['relative_l2']<=0.10 and r['mean_kl_nats']<=0.01 and r['effective_bits_per_kv_scalar']<=base[(r['split'],r['mode'])]['effective_bits_per_kv_scalar'])
args.out.write_text(json.dumps({'schema':'astc-kv-summary-v1','scope':'pilot snapshot only; not final model quality','results':result},indent=2)+'\n')
for r in result:
 if r['split']=='pilot_holdout':
  print(f"{r['variant']:26} {r['mode']:7} bits={r['effective_bits_per_kv_scalar']:6.3f} relL2={r['relative_l2']:.4f} KL={r['mean_kl_nats']:.5f} head95={r['head_relative_l2_p95']:.3f} gate={r['pilot_screen_pass']}")

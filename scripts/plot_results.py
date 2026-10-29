#!/usr/bin/env python3
"""Figures from saved measurements; no fitted or synthetic result points."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
out=root/'results/figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
summary=json.loads((root/'results/pilot_summary.json').read_text())['results']
rs={r['variant']:r for r in summary if r['split']=='pilot_holdout' and r['mode']=='V_only'}
fig,ax=plt.subplots(figsize=(9,5.4))
labels={'u2_token':'Affine U2','u4_token':'Affine U4','u8_token':'Affine U8','scalar4_page':'ASTC 4x4 / page-channel','scalar6_page':'ASTC 6x6 / page-channel','scalar6_token':'ASTC 6x6 / per-token','rgba4_page':'ASTC RGBA 4x4','scalar6_hdr_block':'ASTC HDR 6x6 / block-min'}
offsets={'u2_token':(10,-16),'u4_token':(10,-15),'u8_token':(-86,9),'scalar4_page':(-190,9),'scalar6_page':(12,-12),'scalar6_token':(10,-12),'rgba4_page':(10,9),'scalar6_hdr_block':(10,9)}
for name,label in labels.items():
 r=rs[name];x,y=r['v_prefix_bits'],r['relative_l2']*100
 color='#2d6ab0' if name.startswith('scalar') or name.startswith('rgba') else '#b55b28'
 ax.scatter(x,y,s=65,c=color,marker='o' if color=='#2d6ab0' else 's',zorder=4)
 ax.annotate(label,(x,y),xytext=offsets[name],textcoords='offset points',fontsize=9,color=color)
ax.axhline(10,color='#888888',ls='--',lw=1,label='Registered attention screen (10%)')
ax.set_yscale('log');ax.set_ylim(.16,85);ax.set_xlim(1.8,9)
ax.set_xlabel('Logical bits per V scalar in compressed prefix (padding + FP16 metadata)')
ax.set_ylabel('Relative L2 error of attention output (%)')
ax.set_title('Snapshot attention: V-only compression on two held-out pilot texts',loc='left',pad=16)
ax.grid(which='major',alpha=.16);ax.legend(loc='lower left',frameon=False,fontsize=9)
fig.text(.095,.015,'SmolLM2-135M · 6 layers · 288-token compressed prefix + FP16 tail · CPU codec, no GPU timing',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.045,1,1))
for ext in ('png','svg'):fig.savefig(out/f'snapshot_tradeoff.{ext}',dpi=180)
plt.close(fig)

quality=json.loads((root/'results/model_quality.json').read_text())
secondary=json.loads((root/'results/model_quality_secondary.json').read_text())
methods=['u4_token_V_only','scalar6_token_V_only','scalar6_token_KV','scalar6_page_V_only','scalar6_page_KV']
names=['Affine U4 · V only','ASTC6 per-token · V only','ASTC6 per-token · K+V','ASTC6 page/channel · V only','ASTC6 page/channel · K+V']
vals=[(quality['aggregate'][m]['relative_perplexity_vs_fp16']-1)*100 for m in methods]
names.insert(1,'Asymmetric U4 · K+V*');vals.insert(1,(secondary['aggregate']['relative_perplexity_vs_primary_fp16']-1)*100)
fig,ax=plt.subplots(figsize=(9,5.2))
colors=['#b55b28','#b55b28','#2d6ab0','#2d6ab0','#8493a2','#8493a2']
bars=ax.barh(np.arange(len(vals)),vals,color=colors,height=.60)
ax.invert_yaxis();ax.set_yticks(np.arange(len(vals)),names)
for b,v in zip(bars,vals):ax.text(v+.12,b.get_y()+b.get_height()/2,f'{v:+.2f}%',va='center',fontsize=10)
ax.axvline(2,color='#a5443d',ls='--',lw=1.2,label='Registered aggregate PPL screen (+2%)')
ax.set_xlim(0,11.4);ax.set_xlabel('Perplexity increase relative to FP16 KV (%)')
ax.set_title('Causal continuation: full model feedback across all 30 layers',loc='left',pad=16)
ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True);ax.legend(loc='upper right',frameon=False,fontsize=8)
fig.text(.04,.012,'380 scored next tokens across 4 authored pilot texts. No confidence/generalization claim.\n*Asymmetric U4 is a separately labeled secondary diagnostic added after snapshot inspection.',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.07,1,1))
for ext in ('png','svg'):fig.savefig(out/f'model_quality.{ext}',dpi=180)
plt.close(fig)
print(out)

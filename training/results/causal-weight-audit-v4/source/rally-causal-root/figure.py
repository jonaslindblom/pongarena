from pathlib import Path
import sys,json,os
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
os.environ['MPLCONFIGDIR']='/tmp/rally-causal-root/mpl-config'
import numpy as np,torch,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from competition_eval import load_mlp

out=Path('training/results/causal-weight-audit-v4');f=np.load('/tmp/rally-causal-failures/pacer-failure-states.npz');obs=f['observations']
net=load_mlp('training/results/pace-study/training/candidate/pacer/best.pt');torch.set_num_threads(1)
rng=np.random.default_rng(2026110525);idx=rng.choice(len(obs),12000,replace=False)
with torch.no_grad():h=net[:4](torch.from_numpy(obs[idx])).numpy()
inc=obs[idx,2]<0;units=[34,62,68,91,96]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for phase,mask,color,offset in [('Incoming',inc,'#276fbf',-.16),('Outgoing',~inc,'#d18024',.16)]:
 bp=ax[0,0].boxplot([h[mask,j] for j in units],positions=np.arange(len(units))+offset,widths=.26,showfliers=False,patch_artist=True,manage_ticks=False)
 for b in bp['boxes']:b.set_facecolor(color);b.set_alpha(.65)
 ax[0,0].plot([],[],color=color,lw=6,label=phase)
ax[0,0].set(xticks=range(len(units)),xticklabels=units,xlabel='Second hidden-layer neuron',ylabel='Tanh activation',ylim=(-1.15,1.15),title='Learned channels are gated by ball direction');ax[0,0].legend(frameon=False,loc='lower left',ncol=2)
ax[0,1].bar(['Raw removal','Centered gain 0.9','Centered gain 1.1'],[8841/90076*100,0,0],color=['#b54444','#4b8e62','#4b8e62'])
ax[0,1].set(ylabel='Incoming decisions changed (%)',title='Bias compensation preserves the incoming vote',ylim=(0,11))
for i,v in enumerate([8841,0,0]):ax[0,1].text(i,.25+(v/90076*100),f'{v:,} / 90,076',ha='center',fontsize=9)
critical=np.load('/tmp/rally-causal-failures/critical-states.npz');counts=critical['occurrences'][critical['phase']==1]
ax[1,0].bar(np.arange(1,9),counts,color='#526d82');ax[1,0].set(xticks=np.arange(1,9),xlabel='Distinct late-recovery state',ylabel='Error occurrences',title='39 proven recovery errors concentrate in 8 states')
for i,v in enumerate(counts,1):ax[1,0].text(i,v+.15,str(v),ha='center')
d=json.load(open('/tmp/rally-causal-root/development.json'));names=['centered-gain-0.9','centered-gain-1.1','restore-up-96-r24','restore-down-91-r18'];labels=['Gain 0.9','Gain 1.1','Restore upward path','Restore downward path']
for i,name in enumerate(names):
 r=d['comparison'][name];m=r['strong_mean_margin_delta'];lo,hi=r['strong_delta_ci95'];ax[1,1].errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color='#9c5151',capsize=4)
ax[1,1].axvline(0,color='#555',ls='--',lw=1);ax[1,1].set(yticks=range(4),yticklabels=labels,xlabel='Point-margin change vs original Pacer',title='First four edits: no reliable strong-panel gain');ax[1,1].invert_yaxis()
fig.suptitle('Pacer: what causal weight tests reveal',fontsize=17)
fig.savefig(out/'causal-findings.png',dpi=160);fig.savefig(out/'causal-findings.svg');plt.close(fig)
print('Saved causal figure',out/'causal-findings.png')

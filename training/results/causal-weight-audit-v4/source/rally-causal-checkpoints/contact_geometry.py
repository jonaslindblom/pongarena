"""Explain changed contact geometry for the fixed unit56 proposal; no fitting."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch
from competition_eval import load_mlp
from policy import ROOT
from pace_study import CONFIGS,tables,NEXT
from env import reflect
OUT=Path('/tmp/rally-causal-checkpoints');torch.set_num_threads(1)
data=np.load('/tmp/rally-causal-failures/pacer-failure-states.npz');ids=np.flatnonzero(data['contact_side']==data['raw'][:,8]);obs=data['observations'][ids];base=data['actions'][ids,0];raw=data['raw'][ids]
model=load_mlp(OUT/'contact-motion-56.pt')
with torch.no_grad():chosen=model(torch.from_numpy(obs)).argmax(1).numpy()
p=np.rint(obs[:,4]*80+80).astype(int);h=reflect(raw[:,1]+raw[:,3]).astype(int);s=abs(np.rint(obs[:,2]*6).astype(int));centers=NEXT[p-15];offset=h[:,None]-centers;row=np.arange(len(ids));old=offset[row,base];new=offset[row,chosen];tab=tables(CONFIGS['pace13-v9-scaled']);risk0=tab['risk'][s,h,old+15];risk1=tab['risk'][s,h,new+15]
report={'purpose':'Offline explanation of contact geometry on source rollout states; not final gameplay evidence','groups':{}}
for name,mask in [('risk_better',risk1<risk0-1e-8),('risk_worse',risk1>risk0+1e-8),('all_changed',chosen!=base)]:
 report['groups'][name]={'count':int(mask.sum()),'mean_abs_offset_before':float(abs(old[mask]).mean()),'mean_abs_offset_after':float(abs(new[mask]).mean()),'flatter':int((mask&(abs(new)<abs(old))).sum()),'steeper':int((mask&(abs(new)>abs(old))).sum()),'old_up':int((mask&(base==0)).sum()),'old_down':int((mask&(base==2)).sum()),'new_stay':int((mask&(chosen==1)).sum())}
(OUT/'contact-geometry.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

"""Two phase-motivated aligned donor rows, centered on incoming saturation."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch
from competition_eval import load_mlp
from policy import ROOT,export
OUT=Path('/tmp/rally-causal-checkpoints');torch.set_num_threads(1)
obs=np.load('/tmp/rally-dense-transfer/observations.npy');a=np.load(OUT/'activations.npz')
current=load_mlp(ROOT/'training/results/pace-study/training/dagger-continuation-s412/best.pt')
base={k:v.numpy().copy() for k,v in current.state_dict().items()};h1=a['36_h1'];h2=a['36_h2'];scores=a['36_scores'];masks={'incoming':obs[:,2]<0,'outgoing':obs[:,2]>0}
rows=[('restore-upward-96-r24',96,24,-1.),('restore-downward-91-r18',91,18,1.)]
report={'purpose':'Two semantically chosen donor-path interventions; no game selection or fitting','observation_source':'retained development states from /tmp/rally-dense-transfer/observations.npy, interpretation only','states':len(obs),'candidates':{}}
for name,j,r,c in rows:
 donor=load_mlp(ROOT/f'training/results/pace-study/training/dagger-continuation-s412/round-{r}.pt')
 old={k:v.numpy().copy() for k,v in donor.state_dict().items()}
 state={k:v.copy() for k,v in base.items()}
 state['2.weight'][j]=old['2.weight'][j];state['2.bias'][j]=old['2.bias'][j]
 state['4.weight'][:,j]=old['4.weight'][:,j]
 state['4.bias']+=c*(base['4.weight'][:,j]-old['4.weight'][:,j])
 hybrid=load_mlp(ROOT/'training/results/pace-study/training/dagger-continuation-s412/best.pt')
 hybrid.load_state_dict({k:torch.from_numpy(v) for k,v in state.items()})
 with torch.no_grad():predicted=hybrid(torch.from_numpy(obs)).numpy()
 hj=np.tanh(h1@old['2.weight'][j]+old['2.bias'][j]);source_j=h2[:,j];donor_j=a[f'{r}_h2'][:,j]
 # Exact analytical own-paddle margin derivative for the donor row when it
 # sees CURRENT first-layer feature activations, not donor first-layer features.
 derivative=(1-hj*hj)*(((1-h1*h1)*base['0.weight'][:,4])@old['2.weight'][j])*(old['4.weight'][2,j]-old['4.weight'][0,j])
 score_delta=predicted-scores;contrast_delta=score_delta-score_delta.mean(1,keepdims=True)
 detail=dict(unit=j,donor_round=r,incoming_constant=c,graph='6→128→128→3 Tanh; unchanged',changed_parameters=['2.weight row','2.bias element','4.weight column','4.bias incoming constant compensation'],bias_correction=(c*(base['4.weight'][:,j]-old['4.weight'][:,j])).tolist(),phase={})
 for phase,mask in masks.items():
  dh=hj[mask]-hj[mask].mean();do=donor_j[mask]-donor_j[mask].mean();dn=source_j[mask]-source_j[mask].mean()
  corr=lambda l,rr:float((l*rr).sum()/max(np.sqrt((l*l).sum()*(rr*rr).sum()),1e-12))
  detail['phase'][phase]=dict(hybrid_activation_mean=float(hj[mask].mean()),hybrid_activation_std=float(hj[mask].std()),source_activation_mean=float(source_j[mask].mean()),hybrid_near_incoming_constant_max=float(abs(hj[mask]-c).max()),source_near_incoming_constant_max=float(abs(source_j[mask]-c).max()),correlation_hybrid_to_donor=corr(dh,do),correlation_hybrid_to_current=corr(dh,dn),hybrid_unit_own_margin_derivative=float(derivative[mask].mean()),current_unit_own_margin_derivative=float(a['36_own'][mask,j].mean()),source_action_change=float((predicted[mask].argmax(1)!=scores[mask].argmax(1)).mean()),max_centered_logit_delta=float(abs(contrast_delta[mask]).max()),mean_centered_logit_delta=float(abs(contrast_delta[mask]).mean()))
 # Upstream detector alignment and neutral preactivation offsets.
 weights=abs(old['2.weight'][j])*np.std(a[f'{r}_h1'],axis=0)
 upstream=[]
 for i in np.argsort(weights)[-12:][::-1]:
  x=a[f'{r}_h1'][:,i];y=h1[:,i];xc=x-x.mean();yc=y-y.mean()
  upstream.append(dict(unit=int(i),donor_connection=float(old['2.weight'][j,i]),current_connection=float(base['2.weight'][j,i]),activation_correlation=float((xc*yc).sum()/max(np.sqrt((xc*xc).sum()*(yc*yc).sum()),1e-12)),mean_activation_drift=float((y-x).mean())))
 detail['upstream']=upstream
 # Counterfactual own-position grid; no labels or parameter adjustment.
 grid=np.arange(15,146,2);sample=obs[masks['outgoing']][::max(1,int(masks['outgoing'].sum())//1024)]
 probe=np.repeat(sample,len(grid),axis=0);probe[:,4]=np.tile((grid-80)/80,len(sample))
 with torch.no_grad():
  ph1=current[:2](torch.from_numpy(probe)).numpy()
  now=np.tanh(ph1@base['2.weight'][j]+base['2.bias'][j]).reshape(len(sample),len(grid))
  new=np.tanh(ph1@old['2.weight'][j]+old['2.bias'][j]).reshape(len(sample),len(grid))
 crossings=[]
 for kind,data in [('current',now),('hybrid',new)]:
  had=(data[:,:-1]*data[:,1:]<=0)
  first=np.argmax(had,axis=1);ok=had.any(axis=1)
  points=(grid[first]+grid[first+1])/2
  crossings.append(dict(kind=kind,cases=int(ok.sum()),total=len(sample),midpoint_quantiles=np.quantile(points[ok],[0,.25,.5,.75,1]).tolist() if ok.any() else []))
 detail['own_position_zero_crossings']=crossings
 torch.save(hybrid.state_dict(),OUT/f'{name}.pt');export(hybrid,OUT/f'{name}.onnx')
 report['candidates'][name]=detail
 print(name,json.dumps(detail['phase']),flush=True)
(OUT/'hybrids.json').write_text(json.dumps(report,indent=2)+'\n')

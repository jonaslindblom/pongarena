"""Fixed candidates on deduplicated causally critical states, no selection."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch
from competition_eval import load_mlp
from policy import ROOT
OUT=Path('/tmp/rally-causal-features');z=np.load('/tmp/rally-causal-failures/critical-states.npz');obs=z['observations'];safe=z['safe_actions'];phase=z['phase'];weight=z['occurrences'];baseline=z['chosen']
torch.set_num_threads(1);net=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt')
with torch.no_grad():scores=net(torch.from_numpy(obs)).numpy();a2=net[:4](torch.from_numpy(obs)).numpy()
assert np.array_equal(scores.argmax(1),baseline)
G=[34,62,68];C=np.array([-1,1,-1]);w=net[4].weight.detach().numpy();centered=(a2[:,G]-C)@w[:,G].T
candidates={'centered_gain_0.9':scores-.1*centered,'centered_gain_1.1':scores+.1*centered,'centered_ablation_0_analysis_only':scores-centered}
for p in sorted(Path('/tmp/rally-causal-checkpoints').glob('*.pt')):
 model=load_mlp(p)
 with torch.no_grad():candidates[p.stem]=model(torch.from_numpy(obs)).numpy()
rows=[];summary={}
for name,pred in candidates.items():
 action=pred.argmax(1);good=safe[np.arange(len(obs)),action]
 summary[name]=dict(unique_fixes=int(good.sum()),weighted_fixes=int(weight[good].sum()),unique_changed=int((action!=baseline).sum()),weighted_changed=int(weight[action!=baseline].sum()),by_phase={label:dict(unique_states=int((phase==code).sum()),weighted_occurrences=int(weight[phase==code].sum()),unique_fixes=int((good&(phase==code)).sum()),weighted_fixes=int(weight[good&(phase==code)].sum())) for label,code in [('incoming',0),('outgoing',1)]})
for i in range(len(obs)):
 rows.append(dict(index=i,source_row=int(z['source_row'][i]),phase='incoming' if phase[i]==0 else 'outgoing',occurrences=int(weight[i]),original=int(baseline[i]),teacher=int(z['teacher'][i]),safe_actions=np.flatnonzero(safe[i]).tolist(),candidates={name:dict(action=int(pred[i].argmax()),preserves_safe_option=bool(safe[i,pred[i].argmax()]),scores=pred[i].tolist()) for name,pred in candidates.items()}))
result=dict(purpose='Eleven deduplicated original critical states; fixed candidate controls; no candidate selection or outcome claims',states=len(obs),weighted_occurrences=int(weight.sum()),baseline_safe_actions=0,summary=summary,states_detail=rows)
(OUT/'critical-candidate-actions.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))

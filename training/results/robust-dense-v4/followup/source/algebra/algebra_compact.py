"""Exact algebra only: compact the frozen curve; no matches or selection."""
from pathlib import Path
import sys,importlib.util,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
spec=importlib.util.spec_from_file_location('safe_branch','/tmp/rally-safe-branch/build.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
import numpy as np,torch
from torch import nn
OUT=Path('/tmp/rally-safe-branch')

def compact():
 source,description=builder.make();w1=source[0].weight.detach().numpy().astype(float);b1=source[0].bias.detach().numpy().astype(float)
 w2=source[2].weight.detach().numpy().astype(float);b2=source[2].bias.detach().numpy().astype(float);w3=source[4].weight.detach().numpy().astype(float);b3=source[4].bias.detach().numpy().astype(float)
 # Anchor0 and18 haveargument u±v+2>0 throughoutphysicalcomponentdomain.
 # Both sourcecolumns areopposite, hencec[(u+v+2)−(u−v+2)]=2cv.
 for row in range(31):
  assert abs(w2[row,0]+w2[row,18])<1e-6
  delta=w2[row,0]*4/3
  w2[row,37]+=delta;b2[row]-=2*delta
 # Removeincomingphaseconstant byshiftingthealways-activefirsthinge.
 base=w3[2,31];first_co=w3[2,0];b2[0]+=base/first_co
 w2=np.delete(w2,31,axis=0);b2=np.delete(b2,31);w3=np.delete(w3,31,axis=1)
 # Allremainingunitsignorethetwoincomingindicatorfeatures.
 assert np.max(abs(w2[:,[40,41]]))==0
 keep=[i for i in range(42)if i not in [0,18,40,41]]
 w1=w1[keep];b1=b1[keep];w2=w2[:,keep]
 net=nn.Sequential(nn.Linear(6,len(keep)),nn.ReLU(),nn.Linear(len(keep),len(b2)),nn.ReLU(),nn.Linear(len(b2),3)).eval()
 with torch.no_grad():
  for layer,w,b in zip([net[0],net[2],net[4]],[w1,w2,w3],[b1,b2,b3]):layer.weight.copy_(torch.from_numpy(w.astype(np.float32)));layer.bias.copy_(torch.from_numpy(b.astype(np.float32)))
 return source,net

if __name__=='__main__':
 torch.set_num_threads(1);source,net=compact();x=np.load('/tmp/rally-causal-failures/observations.npy')
 with torch.no_grad():
  expected=torch.cat([source(p)for p in torch.from_numpy(x).split(4096)]).numpy();actual=torch.cat([net(p)for p in torch.from_numpy(x).split(4096)]).numpy()
 report=dict(purpose='Exactsourceweightalgebra, nomatches/nofitting',architecture=[6,38,32,3],parameters=sum(p.numel()for p in net.parameters()),retained_reachable_probes=len(x),action_mismatches=int((expected.argmax(1)!=actual.argmax(1)).sum()),max_score_error=float(abs(expected-actual).max()),plus_twelve_recovery_hinges=dict(architecture=[6,39,44,3],parameters=2168,over_budget=120),available_recovery_hinges=9)
 (OUT/'algebra-compaction.json').write_text(json.dumps(report,indent=2)+'\n');torch.save(net.state_dict(),OUT/'algebra-identical-compact.pt');print(json.dumps(report,indent=2))

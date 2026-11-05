import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'training/scripts'));sys.path.insert(0,'/tmp/rally-safe-branch')
from handcrafted_v4 import target_tables
import importlib.util
spec=importlib.util.spec_from_file_location("safe_branch_builder","/tmp/rally-safe-branch/build.py");module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);goal=module.goal
from pace_study import tables,CONFIGS
import numpy as np
T=tables(CONFIGS['pace13-v9-scaled']);h=np.arange(4,157);c=np.arange(15,146);R=np.full((153,131),np.inf)
for i,y in enumerate(h):
 for j,p in enumerate(c):
  if abs(y-p)<=15:R[i,j]=T['risk'][13,y,y-p+15]
inc,_=target_tables();report={}
for name,g in [('pointwise_argmin',inc[13,h]),('center_contact',h),('fixed_plus6',h+6),('coherent_robust',goal(h))]:
 g=np.clip(g,15,145);possible=abs(c[None]-g[:,None])<=1.101;v=np.where(possible,R,-np.inf).max(1)
 report[name]=dict(robust_safe_heights=int((v<=0).sum()),unsafe_heights=int((v>0).sum()),invalid_catch_heights=int(np.isinf(v).sum()),mean_worst_risk=float(v.mean()) if np.isfinite(v).all() else None,max_worst_risk=float(v.max()) if np.isfinite(v).all() else None,mean_finite_worst_risk=float(v[np.isfinite(v)].mean()),max_finite_worst_risk=float(v[np.isfinite(v)].max()))
Path('/tmp/rally-safe-branch/target-comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

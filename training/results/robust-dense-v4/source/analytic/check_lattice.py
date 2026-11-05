"""Independent dense feature forecast probes, not gameplay selection."""
from pathlib import Path
import sys,json
sys.path.insert(0,'/tmp/rally-analytic-dense')
import cap_global,cap_contact
import numpy as np
OUT=Path('/tmp/rally-analytic-dense')
checks={}
for name,module,xs in [('global_all_cap',cap_global,345-13*np.arange(26)),('contact_only',cap_contact,np.array([20]))]:
 W,b,z,zb,*_=module.first();x,y,vy=np.meshgrid(xs,np.arange(4,157),np.arange(-9,10),indexing='ij');x=x.ravel();y=y.ravel();vy=vy.ravel()
 obs=np.stack([(x-180)/180,(y-80)/80,np.full(len(x),-13/6),vy/6,np.zeros(len(x)),np.zeros(len(x))],axis=1).astype(np.float32)
 h=np.maximum(obs@W.T+b,0);pred=80+80*(h@z+zb);true=y+vy*np.ceil((x-15)/13)
 error=abs(pred-true)
 checks[name]=dict(probes=len(obs),max_raw_unfolded_error_units=float(error.max()),mean_error_units=float(error.mean()),all_within_001=bool((error<.01).all()))
print(checks)
(OUT/'lattice-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
assert all(v['all_within_001'] for v in checks.values())

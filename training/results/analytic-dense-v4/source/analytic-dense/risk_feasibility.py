"""One predeclared coarse physical target prior, no observed-state fitting."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np
from handcrafted_v4 import target_tables
from pace_study import CONFIGS,tables
from env import reflect
T=tables(CONFIGS['pace13-v9-scaled']);inc,out=target_tables()
# Fixed12-unit spatial sampling plus symmetric walls and center, not fitted.
heights=np.unique(np.r_[4,np.arange(16,157,12),80,156]).astype(float)
# Physical function composed with reflection is still1-dimensional.
raw=[]
for n in range(-3,4):
 for h in heights:
  raw.extend([h+304*n,312-h+304*n])
raw=np.unique(np.r_[-240,np.array(raw),400]);raw=raw[(raw>=-240)&(raw<=400)]
values=inc[13,np.rint(reflect(raw)).astype(int)].astype(float)
slopes=np.diff(values)/np.diff(raw);coeff=np.r_[slopes[0],np.diff(slopes)];used=abs(coeff)>1e-12
h=np.arange(4,157);coarse=np.interp(h,heights,inc[13,heights.astype(int)])
def risk_at(target):
 c=np.clip(np.rint(target).astype(int),15,145);offset=h-c
 vy=np.floor(3*13*offset/65+.5).astype(int);dest=reflect(h+26*vy).astype(int)
 risk=np.maximum((T['low'][13,dest]-T['high'][13,dest])/2,np.maximum(T['low'][13,dest]-c-52,c-52-T['high'][13,dest]))
 return dict(empty_reply_regions=int((T['low'][13,dest]>T['high'][13,dest]).sum()),positive_risk=int((risk>0).sum()),mean_risk=float(risk.mean()),median_risk=float(np.median(risk)),catch_invalid=int((abs(offset)>15).sum()))
report=dict(purpose='Physical cap13 target approximation feasibility, uniform integer contactheight probes; not gameplay or fitting',height_knots=heights.tolist(),raw_knots=raw.tolist(),target_values=values.tolist(),active_curvature_hinges=int(used.sum()),width1=27,width2=int(used.sum()+3),parameters=int(7*27+(27+4)*(used.sum()+3)+3),contact_heights=len(h),risk_metrics={'exact_physical_target':risk_at(inc[13,4:]),'coarse_physical_target':risk_at(coarse),'center_contact':risk_at(h),'fixed_plus6':risk_at(h+6),'fixed_minus6':risk_at(h-6)},max_target_approx_error=float(abs(coarse-inc[13,4:]).max()),mean_target_approx_error=float(abs(coarse-inc[13,4:]).mean()))
Path('/tmp/rally-analytic-dense/risk-feasibility.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['raw_knots','target_values']},indent=2))

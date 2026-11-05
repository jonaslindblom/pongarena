from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np
from handcrafted_v4 import target_tables
incoming,outgoing=target_tables()
heights=np.arange(4,157,dtype=np.float32)
summary={};maxerr=0.;maxeval=0.; maxslope=0.
# Float32 matches scalar Add -> Relu -> MatMul -> Add construction.
for name,table in [('incoming',incoming),('outgoing',outgoing)]:
 for stage in range(5,13):
  values=table[stage,4:].astype(np.float32)
  slopes=np.diff(values); coefficients=np.r_[slopes[0],np.diff(slopes)].astype(np.float32)
  selected=coefficients!=0
  knots=np.arange(4,156,dtype=np.float32)[selected];coefficients=coefficients[selected]
  rows={}
  for perturb in [0.,-.001,.001]:
   y=heights+np.float32(perturb)
   values32=np.maximum(y[:,None]-knots[None],np.float32(0))@coefficients+values[0]
   error=np.abs(values32-values)
   rows[str(perturb)]=dict(max_error_units=float(error.max()),argmax_integer_height=int(heights[error.argmax()]))
   maxerr=max(maxerr,float(error.max()))
   if perturb==0:maxeval=max(maxeval,float(error.max()))
  maxslope=max(maxslope,float(np.abs(slopes).max()))
  summary[name+':'+str(stage)]=dict(knots=int(selected.sum()),max_abs_segment_slope=float(np.abs(slopes).max()),probe_errors=rows)
result=dict(purpose='Independent float32 target interpolation robustness; no fitting',integer_height_probes_per_table=len(heights),tables=len(summary),max_exact_input_error_units=maxeval,max_error_including_001_input_perturbation=maxerr,max_abs_segment_slope=maxslope,deadband_units=1.1,intended_target_lattice_units=.5,minimum_distance_to_action_threshold_units=.1,passed=maxerr<.1,details=summary)
assert result['passed']
Path('/tmp/rally-compact-targets/interpolation-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:v for k,v in result.items() if k!='details'})

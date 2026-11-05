"""Separate speed/ceil approximation from dense bilinear approximation offline."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np
from env import reflect
OUT=Path('/tmp/rally-analytic-audit');j=json.load(open(OUT/'analytic-failure-audit.json'));z=np.load(OUT/'analytic-states.npz');d={k:z[k] for k in z.files}
raw=d['raw'];side=raw[:,8];x=np.where(side==0,raw[:,0],360-raw[:,0]);distance=x-15
cap_exact=reflect(raw[:,1]+(distance/13+.5)*raw[:,3]);incoming=d['phase']==0
bilinear=d['analytic_forecast']-cap_exact;cap_error=cap_exact-d['landing'];bilinear[~incoming]=np.nan;cap_error[~incoming]=np.nan
summary={}
for stage in range(5,14):
 mask=incoming&(np.abs(raw[:,2])==stage)
 if mask.any():summary[str(stage)]=dict(states=int(mask.sum()),median_abs_total_forecast_error=float(np.median(np.abs(d['forecast_error'][mask]))),max_abs_total_forecast_error=float(np.max(np.abs(d['forecast_error'][mask]))),median_abs_dense_product_error=float(np.median(np.abs(bilinear[mask]))),max_abs_dense_product_error=float(np.max(np.abs(bilinear[mask]))))
errors=[r for r in j['failures'] if r['final_flight_initially_interceptable']];rows=np.array([r['first_incoming_irreversible_row'] for r in errors]);summary['catchable_failure_onsets']=dict(count=len(rows),remaining_ticks_median=float(np.median(d['ticks_to_contact'][rows])),remaining_ticks_range=[int(d['ticks_to_contact'][rows].min()),int(d['ticks_to_contact'][rows].max())],median_abs_total_error=float(np.median(np.abs(d['forecast_error'][rows]))),median_abs_exact_cap13_error=float(np.median(np.abs(cap_error[rows]))),median_abs_dense_product_error=float(np.median(np.abs(bilinear[rows]))))
summary['outgoing_center80_max_error']=float(np.max(np.abs(d['analytic_target'][~incoming]-80)))
(OUT/'forecast-summary.json').write_text(json.dumps(summary,indent=2)+'\n');np.savez_compressed(OUT/'forecast-decomposition.npz',exact_cap13_landing=cap_exact.astype(np.float32),cap13_forecast_error=cap_error.astype(np.float32),dense_bilinear_approximation_error=bilinear.astype(np.float32))
print(json.dumps(summary))

"""Offline diagnosis of one fixed analytic dense Nano; no edits or new designs."""
from pathlib import Path
import sys,json,shutil,hashlib,importlib.util
ROOT=Path.cwd();OUT=Path('/tmp/rally-analytic-audit');SOURCE=Path('/tmp/rally-analytic-dense')
sys.path.insert(0,str(ROOT/'training/scripts'));sys.path.insert(0,'/tmp/rally-causal-failures')
import numpy as np,torch
from audit_pacer_failures import collect,classify,reachability
from pace_study import decode

def main():
 torch.set_num_threads(1)
 for name in ['build.py','analytic-cap13-safeaim.pt','analytic-cap13-safeaim.onnx']:shutil.copyfile(SOURCE/name,OUT/name)
 spec=importlib.util.spec_from_file_location('analytic_fixed_source',OUT/'build.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
 actor=builder.make();actor.load_state_dict(torch.load(OUT/'analytic-cap13-safeaim.pt',weights_only=True));actor.eval()
 seeds=json.load(open('/tmp/rally-causal-failures/seeds.json'))['seeds']
 data,summaries,rallies=collect(seeds,actor);reach,catch=reachability();failures=classify(data,rallies,reach,catch)
 vx,own,other,ticks,landing=decode(data['observations']);incoming=vx<0
 target=80+80*(data['scores'][:,2]+data['observations'][:,4])
 aim=6*np.clip(10*data['observations'][:,5],-1,1)
 forecast=target-aim;forecast_error=forecast-landing
 forecast[~incoming]=np.nan;forecast_error[~incoming]=np.nan
 data.update(analytic_target=target.astype(np.float32),analytic_forecast=forecast.astype(np.float32),forecast_error=forecast_error.astype(np.float32))
 for r in failures:
  endrows=np.flatnonzero(data['rally_index']==r['index']);last=int(endrows[-1]);first=r['final_flight_start_row'];stage=int(abs(vx[last]));r.update(miss_speed_stage=stage,forecast_error_at_final_flight_start=float(forecast_error[first]),forecast_error_at_last_decision=float(forecast_error[last]),final_effective_target_error=float(np.clip(target[last],15,145)-landing[last]))
  if 'first_incoming_irreversible_row'in r:
   row=r['first_incoming_irreversible_row'];r.update(forecast_error_at_irreversible_error=float(forecast_error[row]),effective_target_error_at_irreversible_error=float(np.clip(target[row],15,145)-landing[row]),irreversible_target_outside_ball_hit_zone=bool(abs(np.clip(target[row],15,145)-landing[row])>15))
 np.savez_compressed(OUT/'analytic-states.npz',**data)
 result=dict(purpose='Offline causal geometry only; fixed analytic model remains unchanged',seeds=seeds,architecture=[6,29,9,3],decisions=len(data['observations']),summaries=summaries,rallies=rallies,failures=failures,diagnostics={},hashes={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['build.py','analytic-cap13-safeaim.pt','analytic-cap13-safeaim.onnx']})
 for opponent in ['attack','defend']:
  ff=[r for r in failures if r['opponent']==opponent]
  stage={str(s):dict(losses=sum(r['miss_speed_stage']==s for r in ff),initially_interceptable=sum(r['miss_speed_stage']==s and r['final_flight_initially_interceptable'] for r in ff)) for s in range(5,14)}
  result['diagnostics'][opponent]=dict(losses=len(ff),categories={c:sum(r['category']==c for r in ff) for c in sorted(set(r['category'] for r in ff))},by_speed_stage=stage,
   catchable_losses_target_wrong_at_error=sum(r['final_flight_initially_interceptable']and r.get('irreversible_target_outside_ball_hit_zone',False) for r in ff),
   median_abs_forecast_error_at_flight_start=float(np.median([abs(r['forecast_error_at_final_flight_start']) for r in ff])),
   median_abs_forecast_error_at_irreversible_error=float(np.median([abs(r['forecast_error_at_irreversible_error']) for r in ff if 'forecast_error_at_irreversible_error'in r])))
  print(opponent,json.dumps(result['diagnostics'][opponent]),flush=True)
 (OUT/'analytic-failure-audit.json').write_text(json.dumps(result,separators=(',',':'))+'\n');print('COMPLETE',len(data['observations']),flush=True)

if __name__=='__main__':main()

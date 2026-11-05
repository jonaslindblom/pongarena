"""One class-budget approximation: use cap13 midpoint in every projection."""
from pathlib import Path
import sys,json
sys.path.insert(0,'/tmp/rally-attacker-targets')
import parity_policy as P
import screen_endpoint as E
import numpy as np
from pace_study import make_policy,run_pair
from competition_eval import load_mlp
from policy import ROOT,OnnxPolicy
P.T=dict(P.T)
P.T['mid']=np.broadcast_to(P.T['mid'][13:14],P.T['mid'].shape).copy()
E.T=P.T
P.ENDPOINT=E.recovery_policy('predicted_endpoint','attack')
opponents={'attack':make_policy(P.RULES,'attack'),'pacer':load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt'),'craft-small':OnnxPolicy(ROOT/'training/results/compact-handcrafted-v4/handcrafted-small-v4.onnx')}
policy=lambda obs:P.parity_band(obs,endpoint=True,cap_risk=True)
record={'purpose':'Class-budget approximation: cap13 midpoint throughout parity-corrected attack and endpointforecast, fixed cap13 risk; no fitting','rng':2026110511,'seeds':P.SEEDS,'pairs':{}}
for name,op in opponents.items():
 row=run_pair(P.RULES,policy,op,P.SEEDS);record['pairs'][name]=row
 Path('/tmp/rally-attacker-targets/cap-mid-results.json').write_text(json.dumps(record,separators=(',',':'))+'\n')
 s=row['summary'];print(name,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],'paired',s['paired_match_wins'],s['paired_match_losses'],s['paired_match_draws'],flush=True)

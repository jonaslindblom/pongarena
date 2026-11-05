import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch
from pace_study import CONFIGS,make_policy,run_pair
from policy import OnnxPolicy

torch.set_num_threads(1)
seeds=(np.random.default_rng(2026110511).choice(50_000_000,24,replace=False)+4_100_000_000).tolist()
candidate=OnnxPolicy('/tmp/rally-counter-root/counter.onnx')
roster={'attack':make_policy(CONFIGS['pace13-v9-scaled'],'attack'),'pacer':OnnxPolicy('public/models/pacer-v4.onnx'),'craft-small':OnnxPolicy('public/models/handcrafted-small-v4.onnx')}
r={'purpose':'One fixed Small graph:19vy bands, conservative parity, cap13 risk/recovery prediction, no training','seeds':seeds,'pairs':{}}
for name,other in roster.items():
 row=run_pair(CONFIGS['pace13-v9-scaled'],candidate,other,seeds);r['pairs'][name]=row
 Path('/tmp/rally-counter-root/screen.json').write_text(json.dumps(r,separators=(',',':'))+'\n')
 s=row['summary'];print(name,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],s['paired_match_wins'],s['paired_match_losses'],s['paired_match_draws'],flush=True)

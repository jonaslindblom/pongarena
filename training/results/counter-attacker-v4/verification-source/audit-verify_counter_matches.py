"""Actual counter native ONNX + adaptive-v4 script / Pacer scalar full matches."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import sys
ROOT=Path.cwd();OUT=Path('/tmp/rally-attacker-audit')
sys.path.insert(0,str(ROOT/'training/scripts'))
import numpy as np,onnxruntime as ort
from pace_study import CONFIGS,make_policy
spec=importlib.util.spec_from_file_location('counter_scalar_v4',ROOT/'training/v4/reference.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('model',type=Path);args=parser.parse_args()
 modelhash=hashlib.sha256(args.model.read_bytes()).hexdigest()
 opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
 nets={name:ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider']) for name,path in {'counter':args.model,'pacer':ROOT/'public/models/pacer-v4.onnx'}.items()}
 attacker=make_policy(CONFIGS['pace13-v9-scaled'],'attack')
 schedule=[(4255000007,pair) for pair in [('counter','attack'),('attack','counter'),('counter','pacer'),('counter','counter')]]+[(4255000043,pair) for pair in [('counter','attack'),('attack','counter')]]
 replays=[];matches=[];native=0;script=0;highvy=0
 for seed,names in schedule:
  f=ref.serve(seed);loops=ref.LoopDetector();game=dict(seed=seed,players=list(names),initial=f,steps=[])
  while not ref.ended_match(f):
   if ref.ended_rally(f):scores=None;obs=None;f=ref.serve(seed,f)
   else:
    obs=[ref.observation(f,side) for side in [0,1]];scores=[]
    for side,name in enumerate(names):
     inp=np.array(obs[side],np.float32)[None]
     if name=='attack':
      action=int(attacker(inp)[0]);score=[int(action==k) for k in range(3)];script+=1
     else:
      score=nets[name].run(None,{'observation':inp})[0][0].tolist();native+=1
      assert np.isfinite(score).all()
     scores.append(score)
    highvy+=int(abs(f['vy'])>5)
    f=ref.advance(f,scores,loops)
   game['steps'].append(dict(observations=obs,scores=scores,state=f))
  replays.append(game);row=dict(seed=seed,players=list(names),frames=len(game['steps']),points=f['points']);matches.append(row);print(row,flush=True)
 assert hashlib.sha256(args.model.read_bytes()).hexdigest()==modelhash
 result=dict(complete=True,model=str(args.model),model_sha256=modelhash,verification_seeds=[4255000007,4255000043],matches=matches,frames=sum(len(g['steps']) for g in replays),native_decisions=native,adaptive_attack_decisions=script,high_vy_frames=highvy,scalar_reference='training/v4/reference.py')
 (OUT/'counter-independent-replays.json').write_text(json.dumps(replays,separators=(',',':'))+'\n')
 (OUT/'counter-independent-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()

"""Action/contact-boundary probes for a specified frozen counter export."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,onnxruntime as ort
from check_grid_counter_reference import reference,OUT

def main():
 p=argparse.ArgumentParser();p.add_argument('model',type=Path);args=p.parse_args()
 observations=[]
 for speed in range(5,14):
  for h in range(4,157):
   for own in [15,16,np.clip(h-16,15,145),np.clip(h-14,15,145),np.clip(h+14,15,145),np.clip(h+16,15,145),144,145]:
    for direction in [-1,1]:
     xpos=15+speed if direction<0 else 345-speed
     observations.append([(xpos-180)/180,(h-80)/80,direction*speed/6,0,(own-80)/80,0])
 obs=np.array(observations,np.float32);expected=reference(obs)
 opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
 s=ort.InferenceSession(str(args.model),sess_options=opts,providers=['CPUExecutionProvider'])
 actual=np.array([s.run(None,{'observation':r[None]})[0][0] for r in obs]);mis=np.flatnonzero(actual.argmax(1)!=expected.argmax(1))
 records=[dict(observation=obs[i].tolist(),native=actual[i].tolist(),expected=expected[i].tolist(),expected_advantage=float(expected[i].max()-expected[i,actual[i].argmax()])) for i in mis]
 result=dict(sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),boundary_probes=len(obs),action_mismatches=len(mis),max_score_error=float(np.max(np.abs(actual-expected))),mismatches=records)
 (OUT/'grid-counter-boundary-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='mismatches'}));print('first_mismatches',records[:3])

if __name__=='__main__':main()

"""Independent float64 formula audit of the actual compact counter ONNX."""
from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,onnxruntime as ort
from pace_study import decode,tables,CONFIGS
from env import reflect
OUT=Path('/tmp/rally-attacker-audit');TAB=tables(CONFIGS['pace13-v9-scaled'])

def reference(obs):
 vx,own,other,t,h=decode(obs);s=np.abs(vx);actor=np.where(vx<0,own,other);receiver=np.where(vx<0,other,own)
 rec=TAB['mid'][13,h];out=np.minimum(13,s+1);F=np.ceil(330/out);k=np.arange(-9,10)[None,:,None]
 scale=(65/(3*out))[:,None,None];height=h[:,None,None]
 lo=np.maximum(np.maximum(height-(k+.5)*scale,height-15),15)
 hi=np.minimum(np.minimum(height-(k-.5)*scale,height+15),145)
 parity=(actor%2)[:,None,None]
 lo=parity+2*np.clip(np.ceil((lo-parity)/2),7,73)
 empty_band=lo-hi
 hi=lo+np.where(hi-lo>=2,2,0)
 dest=reflect(height+F[:,None,None]*k)
 radius=2*(t-1);expected=np.clip(rec,receiver-radius,receiver+radius)
 nextp=np.clip(actor[:,None,None]+[-2,0,2],15,145)
 gap=np.maximum(0,np.maximum(np.maximum(lo-nextp-radius[:,None,None],nextp-hi-radius[:,None,None]),empty_band))
 immediate=np.maximum(0,np.maximum(lo-nextp,nextp-hi))
 safe_lo=TAB['low'][13,dest.astype(int)];safe_hi=TAB['high'][13,dest.astype(int)]
 risk=np.maximum(0,np.maximum(np.maximum((safe_lo-safe_hi)/2,safe_lo-hi-2*F[:,None,None]),lo-2*F[:,None,None]-safe_hi))
 quality=2000*np.abs(dest-expected[:,None,None])-1024000*(gap>1/256)-2*immediate-100*risk
 attack=quality.max(1)+[-.5,0,-.5]
 qgap=np.maximum(0,np.maximum(np.maximum(lo-actor[:,None,None]-2*t[:,None,None],actor[:,None,None]-hi-2*t[:,None,None]),empty_band))
 penalty=512*(qgap>1/256)
 top=(dest-penalty).max(1)[:,0];bottom=-(-dest-penalty).max(1)[:,0]
 target=np.where(top+bottom-2*expected>0,top,bottom)
 error=target-own
 defense=np.stack([-error,np.full(len(obs),1.1),error],1)
 return np.where((vx<0)[:,None],attack,defense)

def main():
 path=Path('/tmp/rally-counter-root/fixed-grid/counter-attacker-v4.onnx');modelhash=hashlib.sha256(path.read_bytes()).hexdigest()
 rng=np.random.default_rng(2026110522);n=8192
 x=rng.integers(15,346,n);y=rng.integers(4,157,n);vx=rng.integers(5,14,n)*rng.choice([-1,1],n);vy=rng.integers(-9,10,n);own,other=rng.integers(15,146,(2,n))
 obs=np.stack([(x-180)/180,(y-80)/80,vx/6,vy/6,(own-80)/80,(other-80)/80],1).astype(np.float32)
 expected=reference(obs)
 opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
 session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
 actual=np.array([session.run(None,{'observation':row[None]})[0][0] for row in obs]);assert np.isfinite(actual).all()
 mismatch=np.flatnonzero(actual.argmax(1)!=expected.argmax(1));severe=[];records=[]
 for i in mismatch:
  gap=float(expected[i].max()-expected[i,actual[i].argmax()])
  row=dict(observation=obs[i].tolist(),expected=expected[i].tolist(),native=actual[i].tolist(),expected_choice=int(expected[i].argmax()),native_choice=int(actual[i].argmax()),expected_choice_advantage=gap)
  records.append(row)
  if gap>1e-4:severe.append(row)
 incoming=obs[:,2]<0
 assert np.equal(actual[incoming]*2,np.rint(actual[incoming]*2)).all()
 result=dict(incoming_scores_on_half_integer_grid=True,model_sha256=modelhash,numeric_probes=n,max_score_error=float(np.max(np.abs(actual-expected))),action_mismatches=len(mismatch),non_tie_action_mismatches=len(severe),mismatches=records)
 (OUT/'grid-counter-reference-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='mismatches'}));print('first_mismatches',records[:3])
 assert not severe,severe[:3]
 assert hashlib.sha256(path.read_bytes()).hexdigest()==modelhash

if __name__=='__main__':main()

"""Static export/accounting and same-development native confirmation only."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path.cwd();OUT=Path('/tmp/rally-offense-features/nano-compression');sys.path.insert(0,str(ROOT/'training/scripts'))
import numpy as np,torch,onnxruntime as ort
from pace_study import CONFIGS,run_pair
from policy import OnnxPolicy
from env import reflect
from opposite_end_nano import make

def main():
 torch.set_num_threads(1);net=make();path=OUT/'opposite-end.onnx';digest=hashlib.sha256(path.read_bytes()).hexdigest()
 observations=[];truth=[]
 for y in range(4,157):
  for vy in range(-9,10):
   h=reflect(y+vy)
   for own in [15,80,145]:
    for other in [79,80]:
     observations.append([(20-180)/180,(y-80)/80,-13/6,vy/6,(own-80)/80,(other-80)/80]);D=4 if other>=80 else 156;error=(83/78*h-5/78*D-own)/80;truth.append([-error,1.101/80,error])
 obs=np.array(observations,np.float32);truth=np.array(truth);candidate=OnnxPolicy(path)
 with torch.no_grad():expected=torch.cat([net(x)for x in torch.from_numpy(obs).split(4096)]).numpy()
 actual=np.array([candidate.session.run(None,{'observation':row[None]})[0][0] for row in obs]);assert np.isfinite(actual).all();mismatches=int((actual.argmax(1)!=expected.argmax(1)).sum());intended=int((actual.argmax(1)!=truth.argmax(1)).sum());assert not mismatches and not intended
 result=dict(model_sha256=digest,cap_contact_probes=len(obs),native_torch_action_mismatches=mismatches,native_formula_action_mismatches=intended,max_native_torch_score_error=float(np.max(np.abs(actual-expected))),max_native_formula_error=float(np.max(np.abs(actual-truth))))
 seeds=json.load(open('/tmp/rally-causal-failures/seeds.json'))['seeds'];row=run_pair(CONFIGS['pace13-v9-scaled'],candidate,OnnxPolicy(ROOT/'public/models/pacer-v4.onnx'),seeds)
 result['native_same_development_pacer']=row
 (OUT/'native-verification.json').write_text(json.dumps(result,separators=(',',':'))+'\n');s=row['summary'];print('native',s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],'paired',s['paired_match_wins'],s['paired_match_losses'],s['paired_match_draws']);print({k:v for k,v in result.items()if k!='native_same_development_pacer'});assert hashlib.sha256(path.read_bytes()).hexdigest()==digest

if __name__=='__main__':main()

from pathlib import Path
import sys,json,hashlib,subprocess
sys.path.insert(0,'/tmp/rally-analytic-dense')
from build import make,export
import numpy as np,torch,onnxruntime as ort
OUT=Path('/tmp/rally-analytic-dense');path=OUT/'analytic-cap13-safeaim.onnx'
net=make();torch.set_num_threads(1)
obs=np.load('/tmp/rally-causal-failures/observations.npy')
idx=np.random.default_rng(2026110524).choice(len(obs),8192,replace=False);x=obs[idx]
with torch.no_grad():expected=net(torch.from_numpy(x)).numpy()
options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
actual=np.array([session.run(None,{'observation':r[None]})[0][0] for r in x])
assert np.isfinite(actual).all()
np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
mismatch=np.flatnonzero(actual.argmax(1)!=expected.argmax(1)); print('action mismatches',len(mismatch),'maximumerror',float(abs(actual-expected).max())); print([(int(idx[i]),x[i].tolist(),expected[i].tolist(),actual[i].tolist()) for i in mismatch[:10]])
code="""import {readFileSync} from 'node:fs';import {validateArenaModel} from './src/validate.ts';import {smallestDivision} from './src/competition.ts';const p=process.argv[1];const b=readFileSync(p);const s=validateArenaModel(b);console.log(JSON.stringify({bytes:b.length,params:s.params,macs:s.macs,nodes:s.nodes,widths:s.architecture.widths,division:smallestDivision(s)}));"""
validation=json.loads(subprocess.run(['node','--import','tsx','--input-type=module','-e',code,str(path)],check=True,capture_output=True,text=True).stdout)
assert validation['widths']==[6,29,9,3] and validation['division']=='nano'
export(make(),OUT/'regenerated.onnx')
assert (OUT/'regenerated.onnx').read_bytes()==path.read_bytes()
result=dict(purpose='Numerical and strict ordinary-dense contract verification, no selection',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),numeric_probes=len(x),action_mismatches=len(mismatch),max_score_error=float(abs(actual-expected).max()),arena=validation,byte_identical_regeneration=True)
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

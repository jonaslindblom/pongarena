from pathlib import Path
import sys,json,hashlib,subprocess
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch,onnxruntime as ort
from competition_eval import load_mlp
OUT=Path('/tmp/rally-analytic-dense');torch.set_num_threads(1)
obs=np.load('/tmp/rally-causal-failures/observations.npy');idx=np.random.default_rng(2026110524).choice(len(obs),8192,replace=False);x=obs[idx]
options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
code="""import {readFileSync} from 'node:fs';import {validateArenaModel} from './src/validate.ts';import {smallestDivision} from './src/competition.ts';const b=readFileSync(process.argv[1]);const s=validateArenaModel(b);console.log(JSON.stringify({bytes:b.length,params:s.params,macs:s.macs,nodes:s.nodes,widths:s.architecture.widths,division:smallestDivision(s)}));"""
result={}
for name in ['stable-cap13-safeaim','cap13-farcorner']:
 path=OUT/f'{name}.onnx';net=load_mlp(OUT/f'{name}.pt',activation='relu')
 with torch.no_grad():expected=net(torch.from_numpy(x)).numpy()
 session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']);actual=np.array([session.run(None,{'observation':r[None]})[0][0] for r in x])
 errors=np.flatnonzero(actual.argmax(1)!=expected.argmax(1));np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
 validation=json.loads(subprocess.run(['node','--import','tsx','--input-type=module','-e',code,str(path)],check=True,capture_output=True,text=True).stdout)
 result[name]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),numeric_probes=len(x),action_mismatches=len(errors),max_score_error=float(abs(actual-expected).max()),arena=validation)
 print(name,result[name],flush=True)
(OUT/'edit-verification.json').write_text(json.dumps(result,indent=2)+'\n')
assert all(r['action_mismatches']==0 for r in result.values())

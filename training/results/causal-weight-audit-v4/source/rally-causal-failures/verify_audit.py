"""Check offline reachability and sampled original ONNX parity; no modifications."""
from pathlib import Path
import sys,json
ROOT=Path.cwd();OUT=Path('/tmp/rally-causal-failures');sys.path.insert(0,str(OUT));sys.path.insert(0,str(ROOT/'training/scripts'))
import numpy as np,onnxruntime as ort
from audit_pacer_failures import reachability
from offense_eval import contact_paths
reach,catch=reachability();np.testing.assert_array_equal(reach[1:67],contact_paths()[1:67,:,:,3]>=0)
d=np.load(OUT/'pacer-failure-states.npz');obs=d['observations'];chosen=d['actions'][:,0];score=d['scores']
critical=np.load(OUT/'critical-states.npz');indices=np.unique(np.r_[np.random.default_rng(2026110523).choice(len(obs),8192,replace=False),critical['source_row']])
opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
s=ort.InferenceSession(str(ROOT/'public/models/pacer-v4.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
native=np.array([s.run(None,{'observation':row[None]})[0][0] for row in obs[indices]])
mis=np.flatnonzero(native.argmax(1)!=chosen[indices]);records=[dict(source_row=int(indices[i]),observation=obs[indices[i]].tolist(),torch_scores=score[indices[i]].tolist(),native_scores=native[i].tolist()) for i in mis]
result=dict(reachability_exact_against_contact_paths_states=66*131*131,native_onnx_observations=len(indices),native_action_mismatches=len(mis),native_max_score_error=float(np.max(np.abs(native-score[indices]))),mismatches=records,critical_states=len(critical['observations']))
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='mismatches'}));assert not len(mis),records[:3]

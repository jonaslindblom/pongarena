"""Independent static native-ONNX / scalar-v4 full matches and archive-seed audit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path('/Users/jonas/.codex-workspaces/worktrees/ios-36683f90-4afb-46b3-b505-f5d7b6d64528/weights')
OUT=Path('/tmp/rally-weight-audit')
sys.path.insert(0,str(ROOT/'training/scripts'))
import numpy as np
import onnxruntime as ort
from pace_study import tables,CONFIGS

spec=importlib.util.spec_from_file_location('frozen_v4_scalar',ROOT/'training/v4/reference.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def session(path):
    opt=ort.SessionOptions();opt.intra_op_num_threads=opt.inter_op_num_threads=1
    return ort.InferenceSession(str(path),sess_options=opt,providers=['CPUExecutionProvider'])

def seed_audit(frozen):
    final=[s for block in frozen['blocks'] for s in block]
    assert len(final)==192 and len(set(final))==192
    seen=set();files=0;leafints=0;seedints=set();overlaps=[];errors=[]
    roots=[ROOT/'training/results',ROOT/'public/training']
    for root in roots:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.suffix not in ('.json','.jsonl') or p.is_relative_to(ROOT/'training/results/handcrafted-v4'):continue
            try:
                objects=[json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.suffix=='.jsonl' else [json.loads(p.read_text())]
            except Exception as e:errors.append([str(p.relative_to(ROOT)),str(e)]);continue
            files+=1
            def visit(value,path='',seed_context=False):
                nonlocal leafints
                if isinstance(value,dict):
                    for k,v in value.items():visit(v,path+'/'+str(k),seed_context or 'seed' in str(k).lower())
                elif isinstance(value,list):
                    for i,v in enumerate(value):visit(v,path+'/'+str(i),seed_context)
                elif isinstance(value,int) and not isinstance(value,bool):
                    leafints+=1
                    if 0<=value<2**32:
                        seen.add(value)
                        if seed_context:seedints.add(value)
                        if value in final:overlaps.append([str(p.relative_to(ROOT)),path,value])
                elif isinstance(value,str) and value.isdigit() and seed_context and 0<=int(value)<2**32:
                    seedints.add(int(value));seen.add(int(value))
                    if int(value) in final:overlaps.append([str(p.relative_to(ROOT)),path,int(value)])
            for obj in objects:visit(obj)
    assert not errors,errors
    assert not overlaps,overlaps
    assert not(set(final)&set(np.random.default_rng(2026110502).choice(100_000_000,24,replace=False)+3_700_000_000))
    return dict(files_scanned=files,json_integer_leaves=leafints,unique_uint32_values=len(seen),unique_seed_context_values=len(seedints),final_seeds=192,overlaps=overlaps,parse_errors=errors,
      method='All integer leaves in archived JSON/JSONL, plus digit strings under seed-named fields; conservative superset of retained seed lists',
      scope=['training/results excluding handcrafted-v4','public/training'])


def main():
    frozen=json.loads((ROOT/'training/results/handcrafted-v4/freeze.json').read_text())
    model=ROOT/'training/results/handcrafted-v4/handcrafted-v4.onnx'
    assert digest(model)==frozen['model_sha256']
    result=dict(model_sha256=digest(model),seed_audit=seed_audit(frozen),matches=[],mismatches=[])
    (OUT/'handcrafted-independent-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    models={name:session(path) for name,path in {'handcrafted':model,'pacer':ROOT/'public/models/pacer-v4.onnx'}.items()}
    tab=tables(CONFIGS['pace13-v9-scaled']);centers=tab['centers']
    quality=-tab['risk']-.0001*np.abs(centers-np.arange(157)[:,None])[None]
    best=np.argmax(np.where(tab['legal'][None],quality,-1e12),axis=2)
    incoming=np.take_along_axis(np.broadcast_to(centers,tab['risk'].shape),best[:,:,None],axis=2)[:,:,0]
    replays=[];decisions=0;maxerr=0.;action_errors=0;high_vy_states=0
    seeds=[3991000007,3991000043]
    for seed in seeds:
        for names in [('handcrafted','pacer'),('pacer','handcrafted'),('handcrafted','handcrafted')]:
            f=ref.serve(seed);loops=ref.LoopDetector();game=dict(seed=seed,players=list(names),initial=f,steps=[])
            while not ref.ended_match(f):
                if ref.ended_rally(f):
                    scores=None;observations=None;f=ref.serve(seed,f)
                else:
                    observations=[ref.observation(f,side) for side in [0,1]];scores=[]
                    for side,name in enumerate(names):
                        obs=np.array(observations[side],np.float32)[None]
                        out=models[name].run(None,{'observation':obs})[0][0]
                        assert np.isfinite(out).all();scores.append(out.tolist());decisions+=1
                        if name=='handcrafted':
                            vx=f['vx']*(1 if side==0 else -1);x=f['x'] if side==0 else 360-f['x']
                            ticks=max(1,int(np.ceil(((x-15) if vx<0 else (345-x))/abs(vx))))
                            q=(f['y']+ticks*f['vy']-4)%304;landing=int(4+q if q<=152 else 308-q)
                            target=incoming[abs(vx),landing] if vx<0 else tab['mid'][abs(vx),landing]
                            error=(target-f['paddles'][side])/80
                            expected=np.array([-error,1.001/80,error])
                            maxerr=max(maxerr,float(np.max(np.abs(out-expected))))
                            if out.argmax()!=expected.argmax():
                                action_errors+=1;result['mismatches'].append(dict(seed=seed,players=names,frame=f,side=side,native=out.tolist(),expected=expected.tolist()))
                            high_vy_states+=int(abs(f['vy'])>5)
                    f=ref.advance(f,scores,loops)
                game['steps'].append(dict(observations=observations,scores=scores,state=f))
            replays.append(game)
            row=dict(seed=seed,players=list(names),frames=len(game['steps']),points=f['points'],last_rally=f['rally'])
            result['matches'].append(row);print(row,flush=True)
            (OUT/'handcrafted-independent-replays.json').write_text(json.dumps(replays,separators=(',',':'))+'\n')
            (OUT/'handcrafted-independent-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    result.update(complete=True,verification_seeds=seeds,frames=sum(len(g['steps']) for g in replays),native_decisions=decisions,handcrafted_ground_truth_action_mismatches=action_errors,
      max_handcrafted_score_error=maxerr,handcrafted_high_vy_decisions=high_vy_states,scalar_reference='training/v4/reference.py')
    assert digest(model)==frozen['model_sha256']
    (OUT/'handcrafted-independent-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    assert not action_errors,result['mismatches'][:3]
    print(json.dumps({k:v for k,v in result.items() if k!='matches'}),flush=True)

if __name__=='__main__':main()

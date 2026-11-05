"""Independent compact policy checks: static weights, exact targets, full v4 games."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path('/Users/jonas/.codex-workspaces/worktrees/ios-36683f90-4afb-46b3-b505-f5d7b6d64528/weights')
OUT=Path('/tmp/rally-compact-audit')
BUILT=Path('/tmp/rally-compact-root/built')
sys.path.insert(0,str(ROOT/'training/scripts'))
import numpy as np
import onnxruntime as ort
from pace_study import tables,CONFIGS
from compact_handcrafted_v4 import build
spec=importlib.util.spec_from_file_location('independent_v4_scalar',ROOT/'training/v4/reference.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def session(path):
    opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
    return ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])

def main():
    paths={size:BUILT/f'handcrafted-{size}-v4.onnx' for size in ['nano','small']}
    paths.update(pacer=ROOT/'public/models/pacer-v4.onnx',open=ROOT/'training/results/handcrafted-v4/handcrafted-v4.onnx')
    hashes={name:digest(path) for name,path in paths.items()}
    source=ROOT/'training/scripts/compact_handcrafted_v4.py';source_hash=digest(source)
    result=dict(source_sha256=source_hash,model_sha256=hashes,regeneration={},numeric={},matches=[],mismatches=[])
    with tempfile.TemporaryDirectory(dir=OUT,prefix='regenerated-') as tmp:
        for size in ['nano','small']:
            rebuilt=Path(tmp)/(size+'.onnx');build(size,rebuilt)
            assert rebuilt.read_bytes()==paths[size].read_bytes()
            result['regeneration'][size]=dict(whole_onnx_byte_identical=True)
    nets={name:session(path) for name,path in paths.items()}
    tab=tables(CONFIGS['pace13-v9-scaled']);centers=tab['centers']
    quality=-tab['risk']-.0001*np.abs(centers-np.arange(157)[:,None])[None]
    best=np.argmax(np.where(tab['legal'][None],quality,-1e12),axis=2)
    incoming=np.take_along_axis(np.broadcast_to(centers,tab['risk'].shape),best[:,:,None],axis=2)[:,:,0]
    def truth(obs,size):
        raw=obs.astype(np.float64)
        x=int(round(raw[0]*180+180));y=int(round(raw[1]*80+80));vx=int(round(raw[2]*6));vy=int(round(raw[3]*6));own=int(round(raw[4]*80+80))
        ticks=max(1,int(np.ceil(((x-15) if vx<0 else (345-x))/abs(vx))))
        q=(y+ticks*vy-4)%304;landing=int(4+q if q<=152 else 308-q)
        target=(incoming[13 if size=='nano' else abs(vx),landing] if vx<0 else (80 if size=='nano' else tab['mid'][abs(vx),landing]))
        error=target-own
        return np.array([-error,1.1,error])
    rng=np.random.default_rng(2026110510);n=8192
    x=rng.integers(15,346,n);y=rng.integers(4,157,n);vx=rng.integers(5,14,n)*rng.choice([-1,1],n);vy=rng.integers(-9,10,n);own,other=rng.integers(15,146,(2,n))
    random=np.stack([(x-180)/180,(y-80)/80,vx/6,vy/6,(own-80)/80,(other-80)/80],1).astype(np.float32)
    for size in ['nano','small']:
        boundary=[]
        for v in list(range(-13,-4))+list(range(5,14)):
            for y in range(4,157):
                target=(incoming[13 if size=='nano' else abs(v),y] if v<0 else (80 if size=='nano' else tab['mid'][v,y]))
                for offset in [-2,-1,0,1,2]:
                    p=np.clip(int(np.floor(target))+offset,15,145)
                    boundary.append([(-165 if v<0 else 165)/180,(y-80)/80,v/6,0,(p-80)/80,0])
        observations=np.concatenate([random,np.array(boundary,np.float32)])
        maxerr=0.;mismatches=[];open_action=0;open_error=0.
        for obs in observations:
            actual=nets[size].run(None,{'observation':obs[None]})[0][0];expected=truth(obs,size)
            maxerr=max(maxerr,float(np.max(np.abs(actual-expected))))
            if actual.argmax()!=expected.argmax():mismatches.append(dict(observation=obs.tolist(),actual=actual.tolist(),expected=expected.tolist()))
            if size=='small':
                old=nets['open'].run(None,{'observation':obs[None]})[0][0]
                open_action+=int(actual.argmax()!=old.argmax())
                open_error=max(open_error,float(np.max(np.abs(actual[[0,2]]-80*old[[0,2]]))))
        result['numeric'][size]=dict(probes=len(observations),fresh_random_probes=n,target_boundary_probes=len(boundary),max_exact_target_score_error_units=maxerr,exact_target_action_mismatches=len(mismatches),open_action_mismatches=open_action if size=='small' else None,open_up_down_max_score_error_units=open_error if size=='small' else None)
        result['mismatches'].extend(mismatches)
        print(size,result['numeric'][size],flush=True)
        assert not mismatches
        if size=='small':assert not open_action
    replays=[];seeds=[4251000007,4251000043];decisions=0;full_open_action=0;full_open_error=0.;full_truth_errors=0
    for size in ['nano','small']:
        for seed in seeds:
            for names in [(size,'pacer'),('pacer',size),(size,size)]:
                f=ref.serve(seed);loops=ref.LoopDetector();game=dict(seed=seed,players=list(names),initial=f,steps=[])
                while not ref.ended_match(f):
                    if ref.ended_rally(f):scores=None;observations=None;f=ref.serve(seed,f)
                    else:
                        observations=[ref.observation(f,side) for side in [0,1]];scores=[]
                        for side,name in enumerate(names):
                            obs=np.array(observations[side],np.float32)
                            actual=nets[name].run(None,{'observation':obs[None]})[0][0]
                            scores.append(actual.tolist());decisions+=1
                            if name in ['nano','small']:
                                expected=truth(obs,name)
                                full_truth_errors+=int(actual.argmax()!=expected.argmax())
                                if name=='small':
                                    old=nets['open'].run(None,{'observation':obs[None]})[0][0]
                                    full_open_action+=int(actual.argmax()!=old.argmax())
                                    full_open_error=max(full_open_error,float(np.max(np.abs(actual[[0,2]]-80*old[[0,2]]))))
                        f=ref.advance(f,scores,loops)
                    game['steps'].append(dict(observations=observations,scores=scores,state=f))
                replays.append(game);row=dict(seed=seed,players=list(names),frames=len(game['steps']),points=f['points']);result['matches'].append(row);print(row,flush=True)
    result.update(complete=True,verification_seeds=seeds,frames=sum(len(g['steps']) for g in replays),native_decisions=decisions,full_match_exact_target_action_mismatches=full_truth_errors,full_match_small_open_action_mismatches=full_open_action,full_match_small_open_up_down_score_error_units=full_open_error)
    assert source_hash==digest(source)
    for name,path in paths.items():assert hashes[name]==digest(path)
    (OUT/'compact-independent-replays.json').write_text(json.dumps(replays,separators=(',',':'))+'\n')
    (OUT/'compact-independent-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    assert not full_truth_errors and not full_open_action
    print('COMPLETE',result['frames'],decisions,flush=True)

if __name__=='__main__':main()

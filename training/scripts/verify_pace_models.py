"""Verify plain v4 exports and record native-ONNX matches for browser parity."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import onnx
import onnxruntime as ort
import torch

from competition_eval import load_mlp
from policy import ROOT

OUT=ROOT/'training/results/pace-study/training'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen=json.loads((OUT/'manifest.json').read_text())
    for name,h in frozen['files'].items():assert digest(ROOT/name)==h,name
    final=json.loads((OUT/'confirmation.json').read_text())
    assert final['complete']
    torch.set_num_threads(1)
    rng=np.random.default_rng(47019)
    sessions={}
    result=dict(purpose='Numerical/contract verification, not model selection',models={})
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    for name in ('pacer','apprentice'):
        directory=OUT/'candidate'/name
        path=directory/'best.onnx'
        graph=onnx.load(path);onnx.checker.check_model(graph)
        assert [n.op_type for n in graph.graph.node]==['Gemm','Tanh','Gemm','Tanh','Gemm']
        assert {p.key:p.value for p in graph.metadata_props}['rules']=='pong-v4'
        net=load_mlp(directory/'best.pt')
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        sessions[name]=session
        obs=rng.uniform(-2,2,(8192,6)).astype(np.float32)
        with torch.no_grad():expected=net(torch.from_numpy(obs)).numpy()
        native=np.array([session.run(None,{'observation':row[None]})[0][0] for row in obs])
        assert np.allclose(expected,native,atol=2e-5,rtol=2e-5)
        assert np.array_equal(expected.argmax(1),native.argmax(1))
        code="""import {readFileSync} from 'node:fs';import {validateModel} from './src/validate.ts';
const b=readFileSync(process.argv[1]);const s=validateModel(b);console.log(JSON.stringify({bytes:b.length,params:s.params,macs:s.macs,nodes:s.nodes,widths:s.architecture.widths}));"""
        validation=json.loads(subprocess.run(['node','--import','tsx','--input-type=module','-e',code,str(path)],cwd=ROOT,capture_output=True,text=True,check=True).stdout)
        actual=0
        maximum=float(np.max(abs(expected-native)))
        for pair,record in final['pairs'].items():
            players=pair.split(':')
            for side,player in enumerate(players):
                if player!=name:continue
                for replay in record['replays']:
                    for f in replay['frames']:
                        if f['actions'] is None:continue
                        x,y,vx,vy=f['ball'];mirror=1 if side==0 else -1
                        inp=np.array([[mirror*(x-180)/180,(y-80)/80,mirror*vx/6,vy/6,
                            (f['paddles'][side]-80)/80,(f['paddles'][1-side]-80)/80]],dtype=np.float32)
                        scores=session.run(None,{'observation':inp})[0][0]
                        assert int(scores.argmax())==f['actions'][side],(name,pair,replay['rally'],f['tick'])
                        actual+=1
        result['models'][name]=dict(sha256=digest(path),numeric_probes=8192,actual_recorded_actions=actual,
            mismatches=0,maximum_score_error=maximum,**validation)
    spec=importlib.util.spec_from_file_location('pace_v4_reference',ROOT/'training/v4/reference.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    matches=[]
    for names in [('pacer','apprentice'),('apprentice','pacer'),('pacer','pacer')]:
        for seed in (2_710_000_007,2_710_000_043):
            frame,loops,steps=ref.serve(seed),ref.LoopDetector(),[]
            while not ref.ended_match(frame):
                scores=None if ref.ended_rally(frame) else [sessions[name].run(None,
                    {'observation':np.array(ref.observation(frame,side),dtype=np.float32)[None]})[0][0].tolist()
                    for side,name in enumerate(names)]
                frame=ref.serve(seed,frame) if scores is None else ref.advance(frame,scores,loops)
                steps.append(dict(scores=scores,state=frame))
            matches.append(dict(seed=seed,players=[name+'-v4' for name in names],steps=steps))
    result.update(native_matches=len(matches),native_frames=sum(len(m['steps']) for m in matches),
                  reference_sha256=digest(ROOT/'training/v4/reference.py'))
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    data=json.dumps(matches,separators=(',',':'))+'\n'
    (OUT/'onnx-replays.json').write_text(data)
    (ROOT/'public/training/pace-onnx-replays.json').write_text(data)
    for name,h in frozen['files'].items():assert digest(ROOT/name)==h,name
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

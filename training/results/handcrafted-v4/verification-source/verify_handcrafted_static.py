"""Regenerate fixed tensors and check signed-zero observations from TS replays."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path('/Users/jonas/.codex-workspaces/worktrees/ios-36683f90-4afb-46b3-b505-f5d7b6d64528/weights')
OUT=Path('/tmp/rally-weight-audit')
sys.path.insert(0,str(ROOT/'training/scripts'))
import numpy as np
import onnx
import onnxruntime as ort
import torch
from handcrafted_v4 import trajectory,target_head
from trajectory_features import export_with_features

def main():
    torch.set_num_threads(1)
    frozen=json.loads((ROOT/'training/results/handcrafted-v4/freeze.json').read_text())
    path=ROOT/'training/results/handcrafted-v4/handcrafted-v4.onnx'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(path)==frozen['model_sha256']
    for name,expected in frozen['source_sha256'].items():assert digest(ROOT/name)==expected,name
    features,head=trajectory(),target_head()
    assert all(not p.requires_grad for p in features.parameters())
    assert all(not p.requires_grad for p in head.parameters())
    with tempfile.TemporaryDirectory(dir=OUT,prefix='regeneration-') as tmp:
        regenerated=Path(tmp)/'regenerated.onnx'
        export_with_features(features,head,regenerated,target_feedback=True)
        original=onnx.load(path);new=onnx.load(regenerated)
        assert new.graph.SerializeToString()==original.graph.SerializeToString()
        new.ClearField('metadata_props')
        for prop in original.metadata_props:new.metadata_props.add().CopyFrom(prop)
        onnx.save(new,regenerated)
        assert regenerated.read_bytes()==path.read_bytes()
    differences=json.loads((OUT/'handcrafted-signed-zero-differences.json').read_text())
    opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
    sessions={name:ort.InferenceSession(str(p),sess_options=opts,providers=['CPUExecutionProvider']) for name,p in {'handcrafted':path,'pacer':ROOT/'public/models/pacer-v4.onnx'}.items()}
    score_error=0.;action_errors=0
    for record in differences:
        obs=np.array(record['actual'],np.float32)
        obs[record['actual_negative_zero_indices']]=np.float32(-0.)
        name=record['players'][record['side']]
        scores=sessions[name].run(None,{'observation':obs[None]})[0][0]
        expected=np.array(record['native_scalar_scores'],np.float32)
        score_error=max(score_error,float(np.max(np.abs(scores-expected))))
        action_errors+=int(scores.argmax()!=expected.argmax())
    assert score_error==0 and action_errors==0
    p=OUT/'handcrafted-independent-verification.json';audit=json.loads(p.read_text())
    audit['regeneration']=dict(graph_and_tensors_byte_identical=True,whole_onnx_byte_identical_after_copying_metadata=True,all_parameter_requires_grad_false=True,frozen_sources_unchanged=True,onnx_sha256=digest(path))
    audit['typescript']['signed_zero_native_score_checks']=len(differences)
    audit['typescript']['signed_zero_native_max_score_error']=score_error
    audit['typescript']['signed_zero_native_action_mismatches']=action_errors
    p.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'regeneration':audit['regeneration'],'typescript':audit['typescript']}))

if __name__=='__main__':main()

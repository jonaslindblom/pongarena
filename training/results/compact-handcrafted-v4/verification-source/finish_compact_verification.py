"""Frozen-source, signed-zero native inference, and archived-seed checks."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path('/Users/jonas/.codex-workspaces/worktrees/ios-36683f90-4afb-46b3-b505-f5d7b6d64528/weights')
OUT=Path('/tmp/rally-compact-audit')
import numpy as np
import onnxruntime as ort

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    folder=ROOT/'training/results/compact-handcrafted-v4'
    frozen=json.loads((folder/'freeze.json').read_text())
    audit=json.loads((OUT/'compact-independent-verification.json').read_text())
    for name,expected in frozen['sources'].items():assert digest(ROOT/name)==expected,name
    for name,expected in frozen['references'].items():assert digest(ROOT/name)==expected,name
    for size,expected in frozen['models'].items():
        assert digest(folder/f'handcrafted-{size}-v4.onnx')==expected['sha256']==audit['model_sha256'][size]
    assert audit['source_sha256']==frozen['sources']['training/scripts/compact_handcrafted_v4.py']
    opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
    paths={size:folder/f'handcrafted-{size}-v4.onnx' for size in ['nano','small']};paths['pacer']=ROOT/'public/models/pacer-v4.onnx'
    nets={name:ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider']) for name,path in paths.items()}
    differences=json.loads((OUT/'compact-signed-zero-differences.json').read_text());maxerr=0.;action_errors=0
    for record in differences:
        obs=np.array(record['actual'],np.float32);obs[record['actual_negative_zero_indices']]=np.float32(-0.)
        name=record['players'][record['side']]
        scores=nets[name].run(None,{'observation':obs[None]})[0][0]
        original=np.array(record['native_scalar_scores'],np.float32)
        maxerr=max(maxerr,float(np.max(np.abs(scores-original))));action_errors+=int(scores.argmax()!=original.argmax())
    assert not maxerr and not action_errors
    audit['typescript'].update(signed_zero_native_score_checks=len(differences),signed_zero_native_max_score_error=maxerr,signed_zero_native_action_mismatches=action_errors)
    final=[v for block in frozen['blocks'] for v in block];assert len(final)==len(set(final))==192
    finalset=set(final);files=0;integers=0;seedvalues=set();values=set();overlaps=[];errors=[]
    for root in [ROOT/'training/results',ROOT/'public/training']:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix not in ('.json','.jsonl') or path.is_relative_to(folder):continue
            try:objects=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.suffix=='.jsonl' else [json.loads(path.read_text())]
            except Exception as error:errors.append([str(path.relative_to(ROOT)),str(error)]);continue
            files+=1
            def visit(value,route='',seed_context=False):
                nonlocal integers
                if isinstance(value,dict):
                    for k,v in value.items():visit(v,route+'/'+str(k),seed_context or 'seed' in str(k).lower())
                elif isinstance(value,list):
                    for i,v in enumerate(value):visit(v,route+'/'+str(i),seed_context)
                elif isinstance(value,int) and not isinstance(value,bool):
                    integers+=1
                    if 0<=value<2**32:
                        values.add(value)
                        if seed_context:seedvalues.add(value)
                        if value in finalset:overlaps.append([str(path.relative_to(ROOT)),route,value])
                elif isinstance(value,str) and value.isdigit() and seed_context and 0<=int(value)<2**32:
                    seedvalues.add(int(value));values.add(int(value))
                    if int(value) in finalset:overlaps.append([str(path.relative_to(ROOT)),route,int(value)])
            for obj in objects:visit(obj)
    assert not errors,errors
    assert not overlaps,overlaps
    dev=set((np.random.default_rng(2026110509).choice(50_000_000,24,replace=False)+4_000_000_000).tolist())
    assert not(finalset&dev) and not(finalset&set(audit['verification_seeds']))
    audit['seed_audit']=dict(files_scanned=files,json_integer_leaves=integers,unique_uint32_values=len(values),unique_seed_context_values=len(seedvalues),final_seeds=len(final),overlaps=overlaps,parse_errors=errors,development_disjoint=True,verification_disjoint=True,method='All integer leaves, plus digit strings under seed-named fields; archived training/results and public/training JSON/JSONL, excluding this experiment')
    audit['frozen_integrity']=dict(all_sources_unchanged=True,all_reference_models_unchanged=True,models_match_frozen_hashes=True)
    (OUT/'compact-independent-verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'typescript':audit['typescript'],'seed_audit':audit['seed_audit'],'frozen_integrity':audit['frozen_integrity']}))

if __name__=='__main__':main()

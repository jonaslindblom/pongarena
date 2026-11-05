"""Frozen integrity, replay formulas, signed-zero native/script, archive seeds."""
import hashlib,json,tempfile
from pathlib import Path
import sys
ROOT=Path.cwd();OUT=Path('/tmp/rally-attacker-audit');FOLDER=ROOT/'training/results/counter-attacker-v4'
sys.path.insert(0,str(ROOT/'training/scripts'))
import numpy as np,onnxruntime as ort
from counter_attacker_v4 import build
from pace_study import make_policy,CONFIGS
from check_grid_counter_reference import reference

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 frozen=json.loads((FOLDER/'freeze.json').read_text());path=FOLDER/'counter-attacker-v4.onnx'
 audit=json.loads((OUT/'counter-independent-verification.json').read_text());assert digest(path)==frozen['model_sha256']==audit['model_sha256']
 for group in ['sources','references']:
  for name,expected in frozen[group].items():assert digest(ROOT/name)==expected,name
 with tempfile.TemporaryDirectory(dir=OUT,prefix='counter-regenerated-') as tmp:
  p=Path(tmp)/'regenerated.onnx';build(p);assert p.read_bytes()==path.read_bytes()
 audit['frozen_integrity']=dict(all_sources_unchanged=True,all_references_unchanged=True,model_matches_frozen_hash=True,whole_onnx_regeneration_byte_identical=True)
 opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
 nets={name:ort.InferenceSession(str(p),sess_options=opts,providers=['CPUExecutionProvider']) for name,p in {'counter':path,'pacer':ROOT/'public/models/pacer-v4.onnx'}.items()}
 attacker=make_policy(CONFIGS['pace13-v9-scaled'],'attack')
 differences=json.loads((OUT/'counter-signed-zero-differences.json').read_text());maxerror=0.;actionerrors=0;native=0;script=0
 for record in differences:
  obs=np.array(record['actual'],np.float32);obs[record['actual_negative_zero_indices']]=np.float32(-0.)
  name=record['players'][record['side']];original=np.array(record['native_scalar_scores'],np.float32)
  if name=='attack':
   chosen=int(attacker(obs[None])[0]);actual=np.array([int(i==chosen) for i in range(3)],np.float32);script+=1
  else:actual=nets[name].run(None,{'observation':obs[None]})[0][0];native+=1
  maxerror=max(maxerror,float(np.max(np.abs(actual-original))));actionerrors+=int(actual.argmax()!=original.argmax())
 assert not maxerror and not actionerrors
 audit['typescript'].update(signed_zero_native_checks=native,signed_zero_script_checks=script,signed_zero_max_score_error=maxerror,signed_zero_action_mismatches=actionerrors)
 games=json.loads((OUT/'counter-independent-replays.json').read_text());observations=[];scores=[]
 for game in games:
  for row in game['steps']:
   if row['scores'] is None:continue
   for side,name in enumerate(game['players']):
    if name=='counter':observations.append(row['observations'][side]);scores.append(row['scores'][side])
 obs=np.array(observations,np.float32);actual=np.array(scores,np.float32);expected=reference(obs)
 mismatch=int((actual.argmax(1)!=expected.argmax(1)).sum());assert not mismatch
 incoming=obs[:,2]<0;assert np.equal(actual[incoming]*2,np.rint(actual[incoming]*2)).all()
 audit['full_match_formula_check']=dict(counter_decisions=len(obs),action_mismatches=mismatch,max_score_error=float(np.max(np.abs(actual-expected))),incoming_half_integer_scores=True)
 final=[v for block in frozen['blocks'] for v in block];assert len(final)==len(set(final))==192;finalset=set(final)
 files=0;integers=0;seedvalues=set();uints=set();overlaps=[];errors=[]
 for root in [ROOT/'training/results',ROOT/'public/training']:
  for p in sorted(root.rglob('*')):
   if not p.is_file() or p.suffix not in ('.json','.jsonl') or p.is_relative_to(FOLDER):continue
   try:objects=[json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.suffix=='.jsonl' else [json.loads(p.read_text())]
   except Exception as error:errors.append([str(p.relative_to(ROOT)),str(error)]);continue
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
      uints.add(value)
      if seed_context:seedvalues.add(value)
      if value in finalset:overlaps.append([str(p.relative_to(ROOT)),route,value])
    elif isinstance(value,str) and value.isdigit() and seed_context and 0<=int(value)<2**32:
     seedvalues.add(int(value));uints.add(int(value))
     if int(value) in finalset:overlaps.append([str(p.relative_to(ROOT)),route,int(value)])
   for obj in objects:visit(obj)
 assert not errors and not overlaps,(errors,overlaps)
 assert not(finalset&set(audit['verification_seeds']))
 dev=set((np.random.default_rng(2026110511).choice(50_000_000,24,replace=False)+4_100_000_000).tolist());assert not(finalset&dev)
 audit['seed_audit']=dict(files_scanned=files,json_integer_leaves=integers,unique_uint32_values=len(uints),unique_seed_context_values=len(seedvalues),final_seeds=192,overlaps=overlaps,parse_errors=errors,development_disjoint=True,verification_disjoint=True,method='Conservative all integer leaves plus seed-context digit strings; training/results and public/training JSON/JSONL excluding current experiment')
 audit['numeric_probes']=json.loads((OUT/'grid-counter-reference-check.json').read_text());audit['action_boundary_probes']=json.loads((OUT/'grid-counter-boundary-check.json').read_text());audit['lattice_probes']=json.loads((OUT/'grid-lattice-check.json').read_text())
 (OUT/'counter-independent-verification.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:audit[k] for k in ['frozen_integrity','typescript','full_match_formula_check','seed_audit']}))

if __name__=='__main__':main()

"""Frozen Nano class confirmation; unrelated future candidates need new seeds."""
from pathlib import Path
import sys,json,hashlib,shutil
import numpy as np,torch
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
from policy import ROOT,OnnxPolicy
from pace_study import CONFIGS,make_policy,run_pair
out=Path('/tmp/rally-robust-native-final');out.mkdir(exist_ok=False)
src=Path('/tmp/rally-safe-branch/small-robust-branch.onnx');frozen=out/'frozen.onnx';shutil.copy2(src,frozen)
hash=hashlib.sha256(frozen.read_bytes()).hexdigest();assert hash=='64b6f14a604fd6d7ff72b91a2861b7a061f6568b6edde80d4697c2e2160e31ca'
rng=np.random.default_rng(2026110530);seeds=(rng.choice(10000000,192,replace=False)+1190000000).tolist();target=set(seeds);overlaps=[];files=0;leaves=0
for root in [ROOT/'training/results',ROOT/'public/training']:
 for p in sorted(root.rglob('*')):
  if not p.is_file() or p.suffix not in ['.json','.jsonl']:continue
  values=[json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.suffix=='.jsonl' else [json.loads(p.read_text())];files+=1
  def walk(v):
   global leaves
   if isinstance(v,dict):
    for a in v.values():walk(a)
   elif isinstance(v,list):
    for a in v:walk(a)
   elif isinstance(v,int) and not isinstance(v,bool):
    leaves+=1
    if v in target:overlaps.append([str(p.relative_to(ROOT)),v])
  for v in values:walk(v)
assert not overlaps,overlaps
meta=dict(purpose='Native confirmation of frozen Small coherentcontactbranch; primarytarget Pacer plus report Nano field and adaptive script limitations',source_sha256=hash,freeze_before_seed_generation=True,seed_audit=dict(rng=2026110530,range=[1190000000,1200000000],files=files,integer_leaves=leaves,overlaps=overlaps),blocks=[seeds[i:i+64] for i in range(0,192,64)])
(out/'protocol.json').write_text(json.dumps(meta,indent=2)+'\n');print('FROZEN',hash,'fresh seeds',len(seeds),'files audited',files,flush=True)
torch.set_num_threads(1);rules=CONFIGS['pace13-v9-scaled'];candidate=OnnxPolicy(frozen)
ops={k:OnnxPolicy(ROOT/'public/models'/f'{k}.onnx') for k in ['tracker','anticipator','challenger']};ops['pacer']=OnnxPolicy(ROOT/'public/models/pacer-v4.onnx');ops['prism']=OnnxPolicy(ROOT/'public/models/prism-nano-v4.onnx');ops.update({k:make_policy(rules,k) for k in ['attack','defend']})
r=dict(protocol=meta,pairs={})
for block,seeds in enumerate(meta['blocks'],1):
 for name,op in ops.items():
  row=run_pair(rules,candidate,op,seeds);r['pairs'][f'{block}:{name}']=row
  (out/'results.json').write_text(json.dumps(r,separators=(',',':'))+'\n');s=row['summary'];print(block,name,s['wins'],s['losses'],s['draws'],'CI',s['point_margin_ci95'],'PASS',s['acceptance']['passed'],flush=True)
r['class_qualification']=all(r['pairs'][f'{b}:{k}']['summary']['acceptance']['passed'] for b in range(1,4) for k in ['tracker','anticipator','challenger'])
r['pacer_qualification']=all(r['pairs'][f'{b}:pacer']['summary']['acceptance']['passed'] for b in range(1,4))
r['great_model_qualification']=all(r['pairs'][f'{b}:{k}']['summary']['acceptance']['passed'] for b in range(1,4) for k in ['pacer','attack'])
r['complete']=True;(out/'results.json').write_text(json.dumps(r,separators=(',',':'))+'\n');print('DONE',r['class_qualification'],r['pacer_qualification'],r['great_model_qualification'],flush=True)

from pathlib import Path
import json
import numpy as np
seeds=(np.random.default_rng(2026110518).choice(30_000_000,24,replace=False)+4_050_000_000).tolist()
selected=set(seeds);overlaps=[];files=0;seed_values=set();scanned_integer_leaves=0
for directory in [Path('training/results'),Path('public/training')]:
 for path in directory.rglob('*'):
  if not path.is_file() or path.suffix not in ['.json','.jsonl']:continue
  objects=[json.loads(v) for v in path.read_text().splitlines() if v.strip()] if path.suffix=='.jsonl' else [json.loads(path.read_text())]
  files+=1
  def scan(v,trail='',context=False):
   global scanned_integer_leaves
   if isinstance(v,dict):
    for k,x in v.items():scan(x,trail+'/'+str(k),context or 'seed' in str(k).lower())
   elif isinstance(v,list):
    for i,x in enumerate(v):scan(x,trail+'/'+str(i),context)
   elif isinstance(v,int) and not isinstance(v,bool):
    scanned_integer_leaves+=1
    if context:seed_values.add(v)
    if v in selected:overlaps.append([str(path),trail,v])
   elif isinstance(v,str) and v.isdigit() and context:
    seed_values.add(int(v))
    if int(v) in selected:overlaps.append([str(path),trail,int(v)])
  for value in objects:scan(value)
result=dict(purpose='Audit actual0518development seeds against all retained archive integerleaves, including both oldfinalsets',rng=2026110518,range=[4_050_000_000,4_080_000_000],seeds=seeds,archived_files=files,archived_integer_leaves=scanned_integer_leaves,archived_seed_context_values=len(seed_values),overlaps=overlaps)
Path('/tmp/rally-attacker-targets/seed-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:v for k,v in result.items() if k!='seeds'})
assert not overlaps

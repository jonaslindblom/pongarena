"""Freeze and test one fixed Small counter on fresh seeds, without selecting weights."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'training/scripts'))
import numpy as np
from counter_attacker_v4 import build, validate
from pace_study import CONFIGS, make_policy, run_pair
from policy import OnnxPolicy

OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, separators=(',', ':'))+'\n')


def main():
    if (OUT/'freeze.json').exists():
        raise FileExistsError('Preserve the frozen final evaluation.')
    model = OUT/'counter-attacker-v4.onnx'
    build(model)
    verification = validate(model)
    save(OUT/'verification.json', verification)
    seeds = (np.random.default_rng(2026110513).choice(50_000_000, 192,
             replace=False)+4_200_000_000).tolist()
    blocks = [seeds[i:i+64] for i in range(0, 192, 64)]
    final, overlaps, count = set(seeds), [], 0

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(key, path)
                visit(child, path)
        elif isinstance(value, list):
            for child in value:
                visit(child, path)
        elif isinstance(value, int) and not isinstance(value, bool) and value in final:
            overlaps.append([str(path), value])
        elif isinstance(value, str) and value.isdigit() and int(value) in final:
            overlaps.append([str(path), value])

    for root in (ROOT/'training/results', ROOT/'public/training'):
        for path in root.rglob('*'):
            if path.suffix not in ('.json', '.jsonl') or path.is_relative_to(OUT):
                continue
            objects = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                       if path.suffix == '.jsonl' else [json.loads(path.read_text())])
            for obj in objects:
                visit(obj, path)
            count += 1
    assert not overlaps, overlaps
    sources = [ROOT/'training/scripts/counter_attacker_v4.py',
        ROOT/'training/scripts/compact_handcrafted_v4.py', ROOT/'training/scripts/pace_study.py',
        ROOT/'training/scripts/policy.py', ROOT/'training/v4/reference.py', ROOT/'src/game.ts', Path(__file__)]
    refs = [ROOT/f'public/models/{name}.onnx' for name in
            ('pacer-v4', 'handcrafted-nano-v4', 'handcrafted-small-v4')]
    protocol = dict(model_sha256=digest(model), blocks=blocks,
        rules='pong-v4; actual unchanged adaptive attack from pace_study.py',
        primary='Beat adaptive attack in each of3fresh64-seed blocks: positive whole-seed point-margin95%CI, wins exceed losses on both sides, majority paired contests',
        method='One fixed analytic Small graph; no fitting, gradients, optimizer, or mutation search',
        development='All exploratory outcomes retained separately; final candidate fixed after numerical stay-tie correction',
        runtime='Actual static batch-one native CPU ONNX export, deterministic first-argmax, exact loop detection',
        seed_audit=dict(files=count, seeds=len(seeds), overlaps=overlaps),
        sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
        references={str(p.relative_to(ROOT)): digest(p) for p in refs})
    save(OUT/'freeze.json', protocol)
    print('Frozen', verification, flush=True)
    rules = CONFIGS['pace13-v9-scaled']
    attack, defend = make_policy(rules, 'attack'), make_policy(rules, 'defend')
    candidate = OnnxPolicy(model)
    result = dict(complete=False, attacker=[], field={})
    for index, block in enumerate(blocks):
        row = run_pair(rules, candidate, attack, block)
        result['attacker'].append(row)
        save(OUT/'evaluation.json', result)
        s = row['summary']
        print('attack', index+1, s['wins'], s['losses'], s['draws'],
            s['point_margin_ci95'], s['paired_match_wins'], s['paired_match_losses'], s['paired_match_draws'], flush=True)
    for name, opponent in [('defend', defend), ('pacer-v4', OnnxPolicy(refs[0])),
                          ('handcrafted-nano-v4', OnnxPolicy(refs[1])),
                          ('handcrafted-small-v4', OnnxPolicy(refs[2]))]:
        row = run_pair(rules, candidate, opponent, blocks[0])
        result['field'][name] = row
        save(OUT/'evaluation.json', result)
        s = row['summary']
        print(name, s['wins'], s['losses'], s['draws'], s['point_margin_ci95'], flush=True)
    assert digest(model) == protocol['model_sha256']
    for name, expected in {**protocol['sources'], **protocol['references']}.items():
        assert digest(ROOT/name) == expected, name
    result['primary_passed'] = all(row['summary']['acceptance']['passed'] for row in result['attacker'])
    result['complete'] = True
    save(OUT/'evaluation.json', result)
    print('Completed; primary passed:', result['primary_passed'], flush=True)


if __name__ == '__main__':
    main()

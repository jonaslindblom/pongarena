"""Diagnostic observation-lag stress, not a model of human performance."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'training/scripts'))
from offense_eval import CONFIGS, make_policy, run_pair, torch

torch.set_num_threads(1)
seeds = json.loads(Path(__file__).with_name('screen.json').read_text())['seeds']
result = dict(seeds=seeds, interpretation='Uncompensated ball/opponent observation lag, own paddle current. No human playtest or extrapolation. Delayed policies use deadline only, no loop shortcut.', configs={})
for name, rules in CONFIGS.items():
    result['configs'][name] = {}
    for kind in ['defend', 'interceptor']:
        result['configs'][name][kind] = {}
        for delay in [0, 3, 5, 8]:
            row = run_pair(rules, make_policy(rules, 'attack'), make_policy(rules, kind, delay=delay), seeds)
            result['configs'][name][kind][str(delay)] = row
            print(name, kind, delay, {k: row['summary'][k] for k in ['wins', 'losses', 'draws', 'median_decisive_returns', 'early_finish_rate']}, flush=True)
Path(__file__).with_name('latency.json').write_text(json.dumps(result, indent=2) + '\n')

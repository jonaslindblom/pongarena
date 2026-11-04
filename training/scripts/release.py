"""Select a v2 run on validation only, preserve provenance, and evaluate exported actors."""
import argparse
import json
import shutil
from pathlib import Path
import torch
from policy import ROOT, OnnxPolicy
from evaluate import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', type=Path, default=[Path(f'training/runs/v2-seed{s}') for s in [7, 17, 29]])
    p.add_argument('--out', type=Path, default=Path('training/results/v2'))
    p.add_argument('--test-start', type=int, default=2_100_000_000)
    p.add_argument('--test-count', type=int, default=100)
    args = p.parse_args()
    torch.set_num_threads(1)
    summaries = []
    for run in args.runs:
        config = json.loads((run / 'config.json').read_text())
        assert config['rules'] == 'pong-v2' and config['max_ball_vx'] == 13
        selection = json.loads((run / 'ppo-selection.json').read_text())
        summaries.append(dict(seed=config['seed'], run=str(run), validation_score=selection['validation_score'],
                              selected_update=selection['iteration'], total_steps=config['envs'] * (config['dagger_rounds'] * 256 + config['updates'] * config['horizon'])))
    winner = max(summaries, key=lambda r: r['validation_score'])
    source = Path(winner['run'])
    args.out.mkdir(parents=True, exist_ok=True)
    files = ['config.json', 'metrics.jsonl', 'imitation-selection.json', 'ppo-selection.json', 'imitation.pt', 'ppo.pt']
    for run, record in zip(args.runs, summaries):
        target = args.out / f'seed{record["seed"]}'
        target.mkdir(exist_ok=True)
        for file in files:
            shutil.copyfile(run / file, target / file)
        config = json.loads((target / 'config.json').read_text())
        config['seed_ranges']['test'] = [args.test_start, args.test_start + args.test_count - 1]
        config['test_range_recorded_by'] = 'release.py; does not affect training or validation'
        (target / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    for file in files:
        shutil.copyfile(args.out / f'seed{winner["seed"]}' / file, args.out / file)
    # Apprentice is the selected PPO run's own warm start, for a controlled comparison.
    for stage, name in [('imitation', 'apprentice'), ('ppo', 'selfplay')]:
        shutil.copyfile(source / f'{stage}.onnx', ROOT / f'public/models/{name}-v2.onnx')
    manifest = dict(rules='pong-v2', selected_seed=winner['seed'], selection='Highest validation mean (wins + 0.5 draws) over the five reference opponents; test results excluded.',
                    runs=summaries, test_seed_start=args.test_start, test_seed_count=args.test_count)
    (args.out / 'selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Selected', winner, flush=True)
    policies = {name: OnnxPolicy(ROOT / f'public/models/{name}-v2.onnx') for name in ['selfplay', 'apprentice']}
    for name, actor in policies.items():
        result = dict(rules='pong-v2', seed_start=args.test_start, seed_count=args.test_count,
                      model=f'public/models/{name}-v2.onnx', opponents=evaluate(actor, args.test_start, args.test_count))
        (args.out / f'{name}-test.json').write_text(json.dumps(result, indent=2) + '\n')
        print(name, {k: (v['wins'], v['losses'], v['draws']) for k,v in result['opponents'].items()}, flush=True)
    matchups = evaluate(policies['selfplay'], args.test_start, args.test_count,
                        opponents={'selfplay': policies['selfplay'], 'apprentice': policies['apprentice']})
    (args.out / 'matchups.json').write_text(json.dumps(dict(rules='pong-v2', seed_start=args.test_start,
        seed_count=args.test_count, opponents=matchups), indent=2) + '\n')
    print('Learned matchups', matchups, flush=True)


if __name__ == '__main__':
    main()

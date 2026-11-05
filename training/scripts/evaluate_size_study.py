"""Freeze development-selected size-study networks and test fresh paired seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
import onnxruntime as ort
import torch

from burst_rules import CONFIGS, make_policy, run_pair
from train_size_study import ROOT, ARCHITECTURES, actions, actor, digest, export, model_stats, synthetic, tracker


def plain_model(path, hidden):
    net = actor(hidden)
    net.load_state_dict(torch.load(path, weights_only=True))
    return net.eval()


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--screen', type=Path, default=ROOT/'training/results/burst-study/models/screen')
    p.add_argument('--out', type=Path, default=ROOT/'training/results/burst-study/models/release')
    p.add_argument('--freeze', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    cfg = json.loads((args.screen/'config.json').read_text())
    rules = CONFIGS[cfg['rules']['name']]
    manifest_path = args.out/'manifest.json'
    if args.freeze:
        if args.out.exists():
            raise FileExistsError('Release already exists. Use another output directory.')
        args.out.mkdir(parents=True)
        files, models, selected = {}, {}, {}
        for name in cfg['architectures']:
            candidates = []
            for seed in cfg['init_seeds']:
                identity = f'{name}-s{seed}'
                path = args.screen/identity
                chosen = json.loads((path/'selection.json').read_text())
                models[identity] = dict(name=name, seed=seed, path=str(path.relative_to(ROOT)),
                    development_margin=chosen['development_margin'], selected_round=chosen['round'],
                    **model_stats(plain_model(path/'best.pt', ARCHITECTURES[name])))
                files[str((path/'best.pt').relative_to(ROOT))] = digest(path/'best.pt')
                candidates.append(identity)
            best = max(candidates, key=lambda identity:models[identity]['development_margin'])
            selected[name] = best
            target = args.out/name
            target.mkdir()
            source = ROOT/models[best]['path']/'best.pt'
            shutil.copyfile(source, target/'best.pt')
            net = plain_model(source, ARCHITECTURES[name])
            export(net, target/'model.onnx', 'pong-burst-v1')
            graph = onnx.load(target/'model.onnx')
            properties = {item.key:item.value for item in graph.metadata_props}
            properties.update(rule_config=rules.name,
                rule_config_sha256=hashlib.sha256(json.dumps(cfg['rules'], sort_keys=True).encode()).hexdigest(),
                prototype='true', checkpoint_sha256=digest(source))
            onnx.helper.set_model_props(graph, properties)
            onnx.save(graph, target/'model.onnx')
            files[str((target/'best.pt').relative_to(ROOT))] = digest(target/'best.pt')
            files[str((target/'model.onnx').relative_to(ROOT))] = digest(target/'model.onnx')
        for path in (args.screen/'config.json', args.screen/'training-source.py', args.screen/'rules-source.py', Path(__file__)):
            files[str(path.relative_to(ROOT))] = digest(path)
        seeds = (np.random.default_rng(20261221).choice(100_000_000, 32, replace=False)+2_900_000_000).tolist()
        json_file(manifest_path, dict(rules=cfg['rules'], models=models, selected=selected,
            test_seeds=seeds, test_rng_seed=20261221, files=files,
            selection='Highest mean development point margin per architecture; seed order breaks exact ties',
            purpose='Fresh model-size confirmation after freezing all checkpoint choices'))
        print('Frozen', manifest_path, flush=True)
        return
    frozen = json.loads(manifest_path.read_text())
    for name, sha in frozen['files'].items():
        assert digest(ROOT/name) == sha, name
    destination = args.out/'confirmation.json'
    if destination.exists():
        raise FileExistsError('Final model test is already retained; use fresh reserved seeds for future tuning.')
    seeds = frozen['test_seeds']
    nets = {identity:plain_model(ROOT/info['path']/'best.pt', ARCHITECTURES[info['name']])
            for identity, info in frozen['models'].items()}
    opponents = dict(tracker=tracker, interceptor=make_policy(rules, 'interceptor'),
        attack=make_policy(rules, 'attack'), defend=make_policy(rules, 'defend'),
        no_burst=make_policy(rules, 'no-burst'))
    result = dict(manifest_sha256=digest(manifest_path), backend='Frozen PyTorch networks; ONNX checked separately',
                  seeds=seeds, models={}, crossplay={}, ablations={}, complete=False)
    for identity, net in nets.items():
        rows = {}
        for name, opponent in [*opponents.items(), ('mirror', net)]:
            row = run_pair(rules, net, opponent, seeds, capture_replays=2 if name == 'attack' else 0)
            moves = np.array(row['records']['burst_ticks'])
            sides = np.array(row['records']['candidate_side'])
            row['summary']['candidate_mean_burst_ticks'] = float(moves[np.arange(len(sides)), sides].mean())
            rows[name] = row
            s = row['summary']
            print(identity, name, s['wins'], s['losses'], s['draws'], 'median', s['median_decisive_returns'], flush=True)
        result['models'][identity] = rows
        json_file(destination, result)
    selected = {name:nets[identity] for name, identity in frozen['selected'].items()}
    names = list(selected)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            row = run_pair(rules, selected[left], selected[right], seeds)
            result['crossplay'][left+':'+right] = row
            s = row['summary']
            print('crossplay', left, right, s['wins'], s['losses'], s['draws'], flush=True)
    # Mapping burst requests to ordinary movement tests whether the learned
    # action choice actually contributes to play; these are not new candidates.
    for name in ('tiny', 'small', 'large'):
        if name not in selected:
            continue
        net = selected[name]
        def without_burst(obs):
            chosen = actions(net, obs)
            return np.where(chosen == 3, 0, np.where(chosen == 4, 2, chosen))
        for opponent in ('interceptor', 'attack'):
            result['ablations'][name+':'+opponent] = run_pair(rules, without_burst, opponents[opponent], seeds)
    result['complete'] = True
    json_file(destination, result)
    verification = dict(models={}, observations='Recorded final-test physical states; energy changes are component-valid counterfactuals, not claims of reachability')
    rng = np.random.default_rng(20261222)
    for name, net in selected.items():
        identity = frozen['selected'][name]
        row = result['models'][identity]['attack']
        records = []
        for replay in row['replays']:
            for frame in replay['frames']:
                if frame['actions'] is None:
                    continue
                x, y, vx, vy = frame['ball']
                own, other = frame['paddles']
                energy = frame['energy']
                records.append([(x-180)/180, (y-80)/80, vx/6, vy/6,
                    (own-80)/80, (other-80)/80, energy[0]/rules.energy_max, energy[1]/rules.energy_max])
        actual = np.array(records, dtype=np.float32)
        probes = np.concatenate([actual, synthetic(rng, 4096, rules.energy_max)])
        with torch.inference_mode():
            expected = net(torch.from_numpy(probes)).numpy()
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(args.out/name/'model.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
        native = np.concatenate([session.run(None, {'observation':obs[None]})[0] for obs in probes])
        mismatches = int((expected.argmax(-1) != native.argmax(-1)).sum())
        if mismatches:
            raise AssertionError(f'{name}: {mismatches} native export action mismatches')
        counterfactuals = {}
        for column, label in ((6, 'own_energy'), (7, 'opponent_energy')):
            empty, full = actual.copy(), actual.copy()
            empty[:, column], full[:, column] = 0, 1
            a, b = actions(net, empty), actions(net, full)
            ta, tb = opponents['attack'](empty), opponents['attack'](full)
            teacher_changed = ta != tb
            counterfactuals[label] = dict(probes=len(actual), model_changes=int((a != b).sum()),
                teacher_changes=int(teacher_changed.sum()),
                both_teacher_actions_matched=int(((a == ta) & (b == tb) & teacher_changed).sum()),
                burst_requests_empty=int((a >= 3).sum()), burst_requests_full=int((b >= 3).sum()))
        verification['models'][name] = dict(**model_stats(net), bytes=(args.out/name/'model.onnx').stat().st_size,
            actual_game_probes=len(actual), total_probes=len(probes), action_mismatches=mismatches,
            maximum_score_difference=float(np.max(np.abs(expected-native))), counterfactuals=counterfactuals)
    json_file(args.out/'verification.json', verification)
    print('Completed', destination, flush=True)


if __name__ == '__main__':
    main()

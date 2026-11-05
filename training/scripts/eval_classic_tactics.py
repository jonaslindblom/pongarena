"""Freeze, then evaluate the matched Classic tactical-label experiment once.

Final outcomes must not choose checkpoints. Bootstrap paired improvements over
the same whole serve seeds; local placement probes are a separate measurement.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch

from competition_eval import load_mlp
from pace_study import run_pair, make_policy
from policy import ROOT
from train_classic_tactics import ARMS, BASELINE, RULES, digest, export_actor, json_file
from train_pace import broad_observations


def freeze(source, out, count):
    if out.exists():raise FileExistsError('Use a new evaluation directory; never replace a consumed final test.')
    if not (source/'completed.json').exists():raise ValueError('Training must be complete before freezing.')
    seeds = (np.random.default_rng(2701).choice(10_000_000,count,replace=False)+3_700_000_000).tolist()
    assert 3703494462 not in seeds
    out.mkdir(parents=True)
    models = {}
    for arm in ARMS:
        path = out/arm
        path.mkdir()
        for filename in ['best.pt','best.onnx','selection.json']:
            shutil.copyfile(source/arm/filename,path/filename)
        models[arm] = dict(checkpoint_sha256=digest(path/'best.pt'),onnx_sha256=digest(path/'best.onnx'),
                           selected_round=json.loads((path/'selection.json').read_text())['round'],role='primary development-selected checkpoint')
        if models[arm]['selected_round'] == 0:
            last = out/(arm+'-last')
            last.mkdir()
            shutil.copyfile(source/arm/'last.pt',last/'best.pt')
            export_actor(load_mlp(last/'best.pt'),last/'best.onnx',arm+'-last')
            models[arm+'-last'] = dict(checkpoint_sha256=digest(last/'best.pt'),onnx_sha256=digest(last/'best.onnx'),
                selected_round=json.loads((source/'config.json').read_text())['rounds'],
                role='secondary fixed-final-round diagnostic because development selected the unchanged initialization')
    files = [Path(__file__),ROOT/'training/scripts/train_classic_tactics.py',
             ROOT/'training/scripts/classic_tactics.py',ROOT/'training/scripts/pace_study.py',
             ROOT/'training/v4/reference.py',ROOT/'src/game.ts']
    json_file(out/'manifest.json',dict(rules='pong-v4',source=str(source.relative_to(ROOT)),
        training_config_sha256=digest(source/'config.json'),models=models,baseline_sha256=digest(BASELINE),
        probe_protocol_sha256=digest(ROOT/'training/results/classic-tactics/probes/protocol.json'),
        final_probe_seeds=json.loads((ROOT/'training/results/classic-tactics/probes/protocol.json').read_text())['seed_sets']['final'],
        final_seeds=seeds,final_seed_generator=dict(rng=2701,interval=[3700000000,3710000000],count=count),
        primary_targets=['attack','defend'],secondary_targets=['pacer'],
        primary_measure='Paired same-seed point-margin gain over frozen Pacer and equal-budget control, separately against attack and defend',
        bootstrap=dict(unit='whole seed with both sides and all five rallies',samples=10000,rng=2703,confidence=.95),
        promotion='No automatic promotion; report all opponents, branch probes, effect estimates and uncertainty',
        sources={str(path.relative_to(ROOT)):digest(path) for path in files}))
    print('Frozen',out,flush=True)


def assert_frozen(out, manifest):
    for name,info in manifest['models'].items():
        assert digest(out/name/'best.pt') == info['checkpoint_sha256']
        assert digest(out/name/'best.onnx') == info['onnx_sha256']
    assert digest(BASELINE) == manifest['baseline_sha256']
    assert digest(ROOT/'training/results/classic-tactics/probes/protocol.json') == manifest['probe_protocol_sha256']
    for name,value in manifest['sources'].items():assert digest(ROOT/name) == value,name


def paired_delta(first, second, rng):
    a = {row['seed']:row['paired_point_margin']/10 for row in first['summary']['per_seed']}
    b = {row['seed']:row['paired_point_margin']/10 for row in second['summary']['per_seed']}
    assert set(a) == set(b)
    seeds = sorted(a)
    values = np.array([a[seed]-b[seed] for seed in seeds])
    samples = rng.integers(0,len(values),(10000,len(values)))
    interval = np.quantile(values[samples].mean(1),[.025,.975])
    return dict(mean_point_margin_gain=float(values.mean()),ci95=interval.tolist(),
                whole_seed_count=len(seeds),per_seed=[dict(seed=seed,point_margin_gain=float(v)) for seed,v in zip(seeds,values)])


def verify_exports(out, nets):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    sessions = {name:ort.InferenceSession(str(out/name/'best.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
                for name in nets if name != 'pacer'}
    numeric = broad_observations(np.random.default_rng(2704),8192)
    verification = dict(models={})
    for name,session in sessions.items():
        graph = onnx.load(out/name/'best.onnx')
        onnx.checker.check_model(graph)
        assert [node.op_type for node in graph.graph.node] == ['Gemm','Tanh','Gemm','Tanh','Gemm']
        parameters = sum(numpy_helper.to_array(p).size for p in graph.graph.initializer)
        assert parameters == 17795
        with torch.inference_mode():expected = nets[name](torch.from_numpy(numeric)).numpy()
        native = np.concatenate([session.run(None,{'observation':row[None]})[0] for row in numeric])
        disagreement = int((native.argmax(-1) != expected.argmax(-1)).sum())
        assert disagreement == 0,(name,disagreement)
        verification['models'][name] = dict(parameters=parameters,numeric_probes=len(numeric),
            action_mismatches=disagreement,maximum_absolute_score_difference=float(np.abs(expected-native).max()))
    path = ROOT/'training/v4/reference.py'
    spec = importlib.util.spec_from_file_location('classic_tactics_scalar',path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    pacer = ort.InferenceSession(str(ROOT/'public/models/pacer-v4.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    matches = []
    actual_inputs = {name:[] for name in sessions}
    for name,session in sessions.items():
        for seed,side in [(3_720_000_007,0),(3_720_000_043,1)]:
            players = ['pacer-v4','pacer-v4']
            players[side] = 'classic-'+name
            pair = [pacer,pacer]
            pair[side] = session
            frame,loops,steps = reference.serve(seed),reference.LoopDetector(),[]
            while not reference.ended_match(frame):
                if reference.ended_rally(frame):
                    scores = None
                    frame = reference.serve(seed,frame)
                else:
                    obs = [np.array(reference.observation(frame,index),dtype=np.float32) for index in (0,1)]
                    actual_inputs[name].append(obs[side])
                    scores = [model.run(None,{'observation':observation[None]})[0][0].tolist() for model,observation in zip(pair,obs)]
                    frame = reference.advance(frame,scores,loops)
                steps.append(dict(scores=scores,state=frame))
            matches.append(dict(seed=seed,players=players,steps=steps))
    for name,rows in actual_inputs.items():
        values = np.array(rows,dtype=np.float32)
        with torch.inference_mode():expected = nets[name](torch.from_numpy(values)).numpy()
        native = np.concatenate([sessions[name].run(None,{'observation':row[None]})[0] for row in values])
        disagreement = int((native.argmax(-1) != expected.argmax(-1)).sum())
        assert disagreement == 0,(name,disagreement)
        verification['models'][name].update(actual_game_probes=len(values),actual_game_action_mismatches=disagreement,
            actual_game_maximum_score_difference=float(np.abs(expected-native).max()))
    (out/'onnx-replays.json').write_text(json.dumps(matches,separators=(',',':'))+'\n')
    verification.update(native_matches=len(matches),native_frames=sum(len(m['steps']) for m in matches),reference_sha256=digest(path))
    json_file(out/'verification.json',verification)
    return verification


def probe_models(data_path, checkpoints, output):
    from classic_tactics import evaluate_policy, set_states
    from pace_study import PaceArena
    if output.exists():raise FileExistsError('Do not overwrite retained probe outcomes.')
    data = json.loads(data_path.read_text())
    training = np.load(ROOT/'training/results/classic-tactics/probes/train.npz',allow_pickle=False)['obs']
    keys = {row.tobytes() for row in training}
    examples = np.load(data_path.with_suffix('.npz'),allow_pickle=False)
    inputs,labels = examples['obs'],examples['labels']
    seen_examples = np.array([row.tobytes() in keys for row in inputs])
    unique_inputs = np.unique(inputs,axis=0)
    snapshots = data['snapshots']
    env = PaceArena(len(snapshots),RULES,loop_detection=False)
    set_states(env,snapshots)
    side = np.array([s['side'] for s in snapshots])
    starts = np.where(side[:,None] == 0,env.obs(0),env.obs(1))
    seen_starts = {s['id']:row.tobytes() in keys for s,row in zip(snapshots,starts)}
    result = dict(phase=data['phase'],data_sha256=digest(data_path),seeds=data['seeds'],
        interpretation='Model chooses actions until its first contact, then the same original scripts resume. Original deadline; no loop shortcut. Full-model matches are separate.',
        overlap=dict(training_path_examples=len(training),training_unique_observations=len(keys),
            probe_path_examples=len(inputs),probe_path_examples_seen_in_training=int(seen_examples.sum()),
            probe_unique_observations=len(unique_inputs),probe_unique_observations_seen_in_training=sum(row.tobytes() in keys for row in unique_inputs),
            flight_starts=len(starts),flight_starts_seen_in_training=sum(seen_starts.values())),
        models={})
    for name,path in checkpoints.items():
        net = load_mlp(path)
        row = evaluate_policy(data,net)
        row['checkpoint_sha256'] = digest(path)
        row['start_partitions'] = {}
        for value,label in [(False,'unseen_in_training_paths'),(True,'seen_in_training_paths')]:
            subset = [r for r in row['rows'] if seen_starts[r['snapshot_id']] == value]
            opportunities = [r for r in subset if r['strict_opportunity']]
            row['start_partitions'][label] = dict(states=len(subset),
                strict_opportunities=len(opportunities),strict_opportunities_won=sum(r['value']==1 for r in opportunities),
                model_wins=sum(r['value']==1 for r in subset),baseline_wins=sum(r['baseline_value']==1 for r in subset),
                better=sum(r['value']>r['baseline_value'] for r in subset),worse=sum(r['value']<r['baseline_value'] for r in subset))
        with torch.inference_mode():predicted = net(torch.from_numpy(inputs)).argmax(-1).numpy()
        row['path_label_agreement'] = {label:dict(samples=int(mask.sum()),accuracy=float((predicted[mask]==labels[mask]).mean()) if mask.any() else None)
            for label,mask in [('all',np.ones(len(inputs),bool)),('seen',seen_examples),('unseen',~seen_examples)]}
        result['models'][name] = row
        print('probe',name,json.dumps(row['summary']),flush=True)
    json_file(output,result)


def evaluate(out):
    manifest = json.loads((out/'manifest.json').read_text())
    if (out/'confirmation.json').exists():raise FileExistsError('This final set was consumed; do not use it to choose new weights.')
    assert_frozen(out,manifest)
    torch.set_num_threads(1)
    nets = {name:load_mlp(out/name/'best.pt') for name in manifest['models']}
    nets['pacer'] = load_mlp(BASELINE)
    opponents = dict(attack=make_policy(RULES,'attack'),defend=make_policy(RULES,'defend'),pacer=nets['pacer'])
    results = dict(rules='pong-v4',seeds=manifest['final_seeds'],pairs={},paired_improvements={},complete=False)
    for name,net in nets.items():
        for other,opponent in opponents.items():
            result = run_pair(RULES,net,opponent,manifest['final_seeds'])
            results['pairs'][name+':'+other] = result
            s = result['summary']
            print(name,'vs',other,s['wins'],s['losses'],s['draws'],'median',s['median_decisive_returns'],flush=True)
    results['pairs']['tactical:control'] = run_pair(RULES,nets['tactical'],nets['control'],manifest['final_seeds'])
    rng = np.random.default_rng(2703)
    for opponent in opponents:
        for baseline in ('control','pacer'):
            key = 'tactical-minus-'+baseline+':'+opponent
            results['paired_improvements'][key] = paired_delta(results['pairs']['tactical:'+opponent],
                                                               results['pairs'][baseline+':'+opponent],rng)
        results['paired_improvements']['control-minus-pacer:'+opponent] = paired_delta(
            results['pairs']['control:'+opponent],results['pairs']['pacer:'+opponent],rng)
    assert_frozen(out,manifest)
    results['complete'] = True
    json_file(out/'confirmation.json',results)
    verification = verify_exports(out,nets)
    assert_frozen(out,manifest)
    print('Verified',verification['native_matches'],'native games;',verification['native_frames'],'frames',flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'training/results/classic-tactics/learning/run')
    parser.add_argument('--out',type=Path,default=ROOT/'training/results/classic-tactics/learning/evaluation')
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--count',type=int,default=64)
    parser.add_argument('--development-probes',action='store_true')
    parser.add_argument('--final-probes',action='store_true')
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.out = args.out.resolve()
    if args.development_probes:
        torch.set_num_threads(1)
        checkpoints = {arm+'-'+stage:args.source/arm/(stage+'.pt') for arm in ARMS for stage in ('best','last')}
        checkpoints['pacer'] = BASELINE
        probe_models(ROOT/'training/results/classic-tactics/probes/dev.json',checkpoints,args.source/'dev-probes.json')
    elif args.final_probes:
        torch.set_num_threads(1)
        manifest = json.loads((args.out/'manifest.json').read_text())
        assert_frozen(args.out,manifest)
        checkpoints = {arm:args.out/arm/'best.pt' for arm in manifest['models']}
        checkpoints['pacer'] = BASELINE
        final = json.loads((ROOT/'training/results/classic-tactics/probes/final.json').read_text())
        assert final['seeds'] == manifest['final_probe_seeds']
        probe_models(ROOT/'training/results/classic-tactics/probes/final.json',checkpoints,args.out/'tactic-probes.json')
        assert_frozen(args.out,manifest)
    elif args.freeze:freeze(args.source,args.out,args.count)
    else:evaluate(args.out)


if __name__ == '__main__':main()

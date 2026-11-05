"""One reserved evaluation of the separate burst-focused training follow-up."""
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
import onnxruntime as ort
import torch

from burst_rules import BurstArena, DEFAULT, make_policy, run_pair
from burst_tactics import make_safe_policy
from evaluate_size_study import plain_model, json_file
from train_size_study import ROOT, actions, digest, export, synthetic, tracker


def frame_state(record):
    x, y, vx, vy = record['ball']
    return dict(x=x, y=y, vx=vx, vy=vy, paddles=record['paddles'], energy=record['energy'],
                rallyTick=record['tick'], returns=record['returns'])


def scenario(net, record, energy=None, stop_at_contact=False):
    env = BurstArena(1, DEFAULT, loop_detection=True)
    f = frame_state(record)
    if energy is not None:
        f['energy'] = [*f['energy']]
        f['energy'][1] = energy
    env.load([f])
    opponent = make_policy(DEFAULT, 'attack')
    first_burst, spent, contact = None, 0, None
    for tick in range(DEFAULT.tick_limit-f['rallyTick']):
        a, b = opponent(env.obs()), actions(net, env.obs(1))
        if b[0] >= 3 and env.energy[0,1] >= DEFAULT.boost_cost:
            spent += 1
            if first_burst is None:first_burst = tick
        outcome, hit = env.step(np.stack([a, b], 1))
        if hit[0] == 1 and contact is None:
            contact = dict(tick=int(env.ticks[0]), center=float(env.paddles[0,1]), energy=int(env.energy[0,1]))
            if stop_at_contact:
                return dict(caught=True, first_burst_tick=first_burst, burst_ticks=spent, contact=contact)
        if outcome[0] >= 0:
            return dict(caught=contact is not None, outcome=int(outcome[0]), draw_reason=int(env.draw_reason[0]),
                        first_burst_tick=first_burst, burst_ticks=spent, contact=contact, end_returns=int(env.returns[0]))
    raise AssertionError('Scenario did not terminate')


def main():
    root = ROOT/'training/results/burst-study/models'
    source = root/'followup/large-s641'
    out = root/'followup-release'
    if out.exists():raise FileExistsError('The bounded follow-up is already frozen/evaluated.')
    out.mkdir()
    selection = json.loads((source/'selection.json').read_text())
    shutil.copyfile(source/'best.pt', out/'best.pt')
    net = plain_model(out/'best.pt', [128,128])
    baseline = plain_model(root/'release/large/best.pt', [128,128])
    export(net, out/'model.onnx', 'pong-burst-v1')
    graph = onnx.load(out/'model.onnx')
    props = {item.key:item.value for item in graph.metadata_props}
    props.update(rule_config=DEFAULT.name, prototype='true', checkpoint_sha256=digest(out/'best.pt'),
                 training='Pooled DAgger plus reachable burst-label oversampling; separate follow-up')
    onnx.helper.set_model_props(graph, props)
    onnx.save(graph, out/'model.onnx')
    seeds = (np.random.default_rng(20261225).choice(100_000_000,32,replace=False)+3_300_000_000).tolist()
    json_file(out/'manifest.json', dict(selection=selection, test_seeds=seeds,
        onnx_sha256=digest(out/'model.onnx'), checkpoint_sha256=digest(out/'best.pt'),
        baseline_sha256=digest(root/'release/large/best.pt'), source_sha256=digest(Path(__file__))))
    torch.set_num_threads(1)
    opponents = dict(tracker=tracker, interceptor=make_policy(DEFAULT,'interceptor'),
        attack=make_policy(DEFAULT,'attack'), defend=make_policy(DEFAULT,'defend'),
        safe=make_safe_policy(DEFAULT), baseline=baseline, mirror=net)
    results = dict(purpose='Separate bounded training follow-up; not part of equal-budget size comparison', seeds=seeds, pairs={})
    for name, opponent in opponents.items():
        row = run_pair(DEFAULT, net, opponent, seeds)
        results['pairs']['focus:'+name] = row
        s = row['summary']
        print('focus',name,s['wins'],s['losses'],s['draws'],'median',s['median_decisive_returns'],flush=True)
    for name in ('attack','safe'):
        results['pairs']['baseline:'+name] = run_pair(DEFAULT, baseline, opponents[name], seeds)
    def ordinary(obs):
        chosen = actions(net, obs)
        return np.where(chosen == 3,0,np.where(chosen == 4,2,chosen))
    for name in ('attack','interceptor'):
        results['pairs']['no_burst:'+name] = run_pair(DEFAULT, ordinary, opponents[name], seeds)
    json_file(out/'confirmation.json', results)
    tactic_path = ROOT/'training/results/burst-study/rules-tactics.json'
    tactic = json.loads(tactic_path.read_text())['actual_tactic']
    probes = dict(source_sha256=digest(tactic_path), scope='Local legal-state continuations, not an optimal-play proof', models={})
    models = {name:plain_model(root/f'release/{name}/best.pt', hidden) for name,hidden in
              [('micro',[16]),('tiny',[16,16]),('small',[32,32]),('medium',[64,64]),('large',[128,128])]}
    models['focus'] = net
    for name, candidate in models.items():
        probes['models'][name] = dict(
            depleted=scenario(candidate, tactic['final_shot'], stop_at_contact=True),
            full=scenario(candidate, tactic['final_shot'], energy=96, stop_at_contact=True),
            earlier_counter=scenario(candidate,tactic['counter_branches'][0]['start']))
    json_file(out/'tactic-probes.json', probes)
    options=ort.SessionOptions()
    options.intra_op_num_threads=options.inter_op_num_threads=1
    session=ort.InferenceSession(str(out/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    samples=synthetic(np.random.default_rng(20261226),8192,DEFAULT.energy_max)
    with torch.inference_mode():expected=net(torch.from_numpy(samples)).numpy()
    native=np.concatenate([session.run(None,{'observation':x[None]})[0] for x in samples])
    mismatches=int((native.argmax(-1)!=expected.argmax(-1)).sum())
    assert mismatches==0,mismatches
    refpath=ROOT/'training/burst/reference.py'
    spec=importlib.util.spec_from_file_location('burst_followup_scalar',refpath)
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    baseline_session=ort.InferenceSession(str(root/'release/large/model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    matches=[]
    for seed,reverse in [(3_410_000_007,False),(3_410_000_043,True)]:
        sessions=[session,baseline_session] if not reverse else [baseline_session,session]
        names=['burst-focus','burst-large'] if not reverse else ['burst-large','burst-focus']
        frame=reference.serve(seed);loops=reference.LoopDetector();steps=[]
        while not reference.ended_match(frame):
            if reference.ended_rally(frame):scores=None;frame=reference.serve(seed,frame)
            else:
                scores=[s.run(None,{'observation':np.array(reference.observation(frame,side),dtype=np.float32)[None]})[0][0].tolist() for side,s in enumerate(sessions)]
                frame=reference.advance(frame,scores,loops)
            steps.append(dict(scores=scores,state=frame))
        matches.append(dict(seed=seed,players=names,steps=steps))
    text=json.dumps(matches,separators=(',',':'))+'\n'
    (out/'onnx-replays.json').write_text(text)
    (ROOT/'public/training/burst-focus-replays.json').write_text(text)
    json_file(out/'verification.json',dict(numeric_probes=8192,action_mismatches=mismatches,
        maximum_score_difference=float(np.max(np.abs(expected-native))),native_matches=len(matches),
        native_frames=sum(len(m['steps']) for m in matches),reference_sha256=digest(refpath)))
    print('Completed',out,flush=True)


if __name__=='__main__':main()

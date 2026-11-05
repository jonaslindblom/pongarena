"""Matched Pacer continuations: original labels versus legal tactical labels.

Every actor receives the same observations, minibatches and optimizer budget.
Only labels on the oversampled counterfactual paths differ between the arms.
Exported actors are ordinary six-input/three-output Tanh networks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
import torch
from torch import nn

from competition_eval import load_mlp
from pace_study import CONFIGS, PaceArena, make_policy, run_pair
from policy import ROOT, actions, load_builtin
from reference import serve
from train_pace import broad_observations, export_v4


BASELINE = ROOT/'training/results/pace-study/training/candidate/pacer/best.pt'
RULES = CONFIGS['pace13-v9-scaled']
ARMS = ('control', 'tactical')
TRAIN_LOW, TRAIN_HIGH = 3_500_000_000, 3_510_000_000
OLD_TRAIN_SEEDS = {3502446039, 3509813992, 3509869267}
OLD_DEV_SEEDS = {3607230410, 3608270858}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_file(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


class TrainingArena(PaceArena):
    def reset(self, ids):
        super().reset(ids)
        for i in ids:
            if self.rallies[i] == 1:
                self.seeds[i] = self.rng.integers(TRAIN_LOW, TRAIN_HIGH)
                while int(self.seeds[i]) in OLD_TRAIN_SEEDS:
                    self.seeds[i] = self.rng.integers(TRAIN_LOW, TRAIN_HIGH)
                frame = serve(int(self.seeds[i]))
                self.ball[i] = [frame[k] for k in ('x', 'y', 'vx', 'vy')]


def label_conflicts(obs, labels):
    _, inverse = np.unique(obs, axis=0, return_inverse=True)
    minimum = np.full(int(inverse.max())+1, 3, dtype=int)
    maximum = np.full(len(minimum), -1, dtype=int)
    np.minimum.at(minimum, inverse, labels)
    np.maximum.at(maximum, inverse, labels)
    return dict(unique_observations=len(minimum), conflicting_observations=int((minimum != maximum).sum()),
                samples_with_conflicting_labels=int((minimum[inverse] != maximum[inverse]).sum()))


def export_actor(net, path, arm):
    export_v4(net, path)
    graph = onnx.load(path)
    props = {p.key:p.value for p in graph.metadata_props}
    props.update(training='Matched DAgger continuation with '+arm+' counterfactual labels',
                 experiment='classic-tactics', fixed_features='none')
    onnx.helper.set_model_props(graph, props)
    onnx.save(graph, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=24)
    parser.add_argument('--envs', type=int, default=64)
    parser.add_argument('--horizon', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--replay-rounds', type=int, default=4)
    parser.add_argument('--tactical-samples', type=int, default=8192)
    parser.add_argument('--synthetic', type=int, default=8192)
    parser.add_argument('--eval-every', type=int, default=4)
    parser.add_argument('--dev-seeds', type=int, default=16)
    parser.add_argument('--seed', type=int, default=701)
    parser.add_argument('--learning-rate', type=float, default=.0003)
    parser.add_argument('--anchor-weight', type=float, default=0.)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh output directory; do not overwrite retained runs.')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    teacher = make_policy(RULES, 'attack')
    defender = make_policy(RULES, 'defend')
    interceptor = make_policy(RULES, 'interceptor')
    pacer = load_mlp(BASELINE)
    tracker = load_builtin('tracker')
    collection_opponents = [teacher, defender, interceptor, pacer, tracker]
    development = (np.random.default_rng(1701).choice(10_000_000, args.dev_seeds, replace=False)+3_600_000_000).tolist()
    assert not set(development) & OLD_DEV_SEEDS
    with np.load(args.dataset, allow_pickle=False) as data:
        tactical_obs = data['obs'].astype(np.float32)
        tactical_labels = data['labels'].astype(np.int64)
        source_seeds = data['source_seeds'].astype(np.uint64)
    assert tactical_obs.ndim == 2 and tactical_obs.shape[1] == 6 and len(tactical_obs) > 0 and np.isfinite(tactical_obs).all()
    assert tactical_labels.shape == (len(tactical_obs),) and np.isin(tactical_labels, [0, 1, 2]).all()
    assert source_seeds.shape == tactical_labels.shape
    assert ((source_seeds >= TRAIN_LOW) & (source_seeds < TRAIN_HIGH)).all()
    original_labels = teacher(tactical_obs)
    args.out.mkdir(parents=True)
    models = {}
    for arm in ARMS:
        net = load_mlp(BASELINE)
        path = args.out/arm
        path.mkdir()
        models[arm] = dict(net=net, path=path, optimizer=torch.optim.Adam(net.parameters(), lr=args.learning_rate), best=-2.)
    config = {key:str(value) if isinstance(value, Path) else value for key,value in vars(args).items()}
    config.update(rules='pong-v4', rule_config=vars(RULES), architecture=[6,128,128,3], parameters=17795,
        baseline=str(BASELINE.relative_to(ROOT)), baseline_sha256=digest(BASELINE),
        development_seeds=development, training_seed_interval=[TRAIN_LOW, TRAIN_HIGH], excluded_old_training_seeds=sorted(OLD_TRAIN_SEEDS),
        tactical_dataset_sha256=digest(args.dataset), tactical_examples=len(tactical_obs),
        tactical_unique_source_seeds=len(np.unique(source_seeds)),
        tactical_label_conflicts=label_conflicts(tactical_obs,tactical_labels),
        original_label_conflicts=label_conflicts(tactical_obs,original_labels),
        labels_different_from_attack_teacher=int((tactical_labels != original_labels).sum()),
        selection='Highest mean point margin vs adaptive attack and adaptive defense on development seeds; Pacer is a secondary regression check; first checkpoint wins ties',
        method='Both arms receive same pooled trajectories, observations, permutations and epochs. Only labels on the oversampled tactical paths differ; control always uses the original adaptive ATTACK teacher, even for states sourced from defender trajectories.',
        learning_rate_schedule='Initial rate multiplied by (0.4 + 0.6 * (1 - (round-1)/rounds))', minibatch=2048,
        anchor='KL(initial Pacer probability distribution || current distribution), temperature 1, on all shared observations; same coefficient in both arms',
        scope='Finite rollout supervision may depend on branch context absent from six inputs; full held-out matches test transfer. No tactical calculations in inference.',
        sources={str(path.relative_to(ROOT)):digest(path) for path in
                 [Path(__file__),ROOT/'training/scripts/pace_study.py',ROOT/'training/scripts/classic_tactics.py']})
    json_file(args.out/'config.json',config)
    (args.out/'training-source.py').write_bytes(Path(__file__).read_bytes())
    env = TrainingArena(args.envs,RULES,seed=args.seed,loop_detection=False)
    arm_ids = rng.integers(0,len(ARMS),args.envs)
    opponent_ids = rng.integers(0,len(collection_opponents),args.envs)
    replay_x, replay_control, replay_tactical = [],[],[]
    started,total,labels_total = time.monotonic(),0,0

    def evaluate(round_index, diagnostics):
        for arm,m in models.items():
            rows = {name:run_pair(RULES,m['net'],opponent,development)['summary']
                    for name,opponent in [('attack',teacher),('defend',defender),('pacer',pacer)]}
            score = float(np.mean([rows[name]['point_margin'] for name in ('attack','defend')]))
            row = dict(round=round_index,env_steps=total,labels=labels_total,seconds=time.monotonic()-started,
                       development_margin=score,opponents=rows,**diagnostics[arm])
            with (m['path']/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            torch.save(m['net'].state_dict(),m['path']/'last.pt')
            if score > m['best']:
                m['best'] = score
                torch.save(m['net'].state_dict(),m['path']/'best.pt')
                export_actor(m['net'],m['path']/'best.onnx',arm)
                json_file(m['path']/'selection.json',row)
            print(arm,'round',round_index,'margin',round(score,3),
                  ' | '.join(f'{name} {r["wins"]}/{r["losses"]}/{r["draws"]}' for name,r in rows.items()),flush=True)

    evaluate(0,{arm:dict(cross_entropy=None,training_accuracy=None) for arm in ARMS})
    for iteration in range(1,args.rounds+1):
        xs,ys = [],[]
        beta = max(.05,.5*(1-iteration/args.rounds))
        for _ in range(args.horizon):
            obs,other = env.obs(),env.obs(1)
            labels = teacher(obs)
            learned = np.empty(args.envs,int)
            for i,arm in enumerate(ARMS):
                mask = arm_ids == i
                if mask.any():learned[mask] = actions(models[arm]['net'],obs[mask])
            chosen = np.where(rng.random(args.envs)<beta,labels,learned)
            opposing = np.empty(args.envs,int)
            for i in np.unique(opponent_ids):
                mask = opponent_ids == i
                policy = collection_opponents[i]
                opposing[mask] = actions(policy,other[mask]) if isinstance(policy,nn.Module) else policy(other[mask])
            xs.extend([obs,other]);ys.extend([labels,teacher(other)])
            terminal,_ = env.step(np.stack([chosen,opposing],1))
            ids = np.flatnonzero(terminal>=0)
            env.reset(ids)
            arm_ids[ids] = rng.integers(0,len(ARMS),len(ids))
            opponent_ids[ids] = rng.integers(0,len(collection_opponents),len(ids))
        broad = broad_observations(rng,args.synthetic)
        xs.append(broad);ys.append(teacher(broad))
        selected = rng.choice(len(tactical_obs),args.tactical_samples,replace=True)
        new_x = np.concatenate([*xs,tactical_obs[selected]])
        new_control = np.concatenate([*ys,original_labels[selected]])
        new_tactical = np.concatenate([*ys,tactical_labels[selected]])
        replay_x.append(new_x);replay_control.append(new_control);replay_tactical.append(new_tactical)
        replay_x,replay_control,replay_tactical = (items[-args.replay_rounds:] for items in [replay_x,replay_control,replay_tactical])
        x = torch.from_numpy(np.concatenate(replay_x))
        y = dict(control=torch.from_numpy(np.concatenate(replay_control)),
                 tactical=torch.from_numpy(np.concatenate(replay_tactical)))
        diagnostics = {arm:dict(loss=0.,ce=0.,kl=0.,correct=0,seen=0) for arm in ARMS}
        for m in models.values():m['optimizer'].param_groups[0]['lr'] = args.learning_rate*(.4+.6*(1-(iteration-1)/args.rounds))
        for _ in range(args.epochs):
            for ids in torch.from_numpy(rng.permutation(len(x))).split(2048):
                with torch.no_grad():
                    anchor = pacer(x[ids]).softmax(-1) if args.anchor_weight else None
                for arm,m in models.items():
                    logits = m['net'](x[ids])
                    ce = nn.functional.cross_entropy(logits,y[arm][ids])
                    kl = nn.functional.kl_div(logits.log_softmax(-1),anchor,reduction='batchmean') if anchor is not None else logits.new_zeros(())
                    loss = ce+args.anchor_weight*kl
                    m['optimizer'].zero_grad();loss.backward()
                    nn.utils.clip_grad_norm_(m['net'].parameters(),1.)
                    m['optimizer'].step()
                    d = diagnostics[arm]
                    d['loss'] += float(loss.detach())*len(ids)
                    d['ce'] += float(ce.detach())*len(ids)
                    d['kl'] += float(kl.detach())*len(ids)
                    d['correct'] += int((logits.argmax(-1) == y[arm][ids]).sum())
                    d['seen'] += len(ids)
        total += args.envs*args.horizon
        labels_total += len(new_x)
        logged = {arm:dict(cross_entropy=d['ce']/d['seen'],anchor_divergence=d['kl']/d['seen'],optimization_loss=d['loss']/d['seen'],training_accuracy=d['correct']/d['seen'],
                          optimization_examples=d['seen'],teacher_action_probability=beta) for arm,d in diagnostics.items()}
        print('round',iteration,'shared_labels',labels_total,'losses',
              {arm:round(row['cross_entropy'],4) for arm,row in logged.items()},flush=True)
        if iteration%args.eval_every == 0 or iteration == args.rounds:evaluate(iteration,logged)
    json_file(args.out/'completed.json',dict(seconds=time.monotonic()-started,env_steps=total,labels_per_arm=labels_total,
        models={arm:dict(best_development_margin=m['best'],checkpoint_sha256=digest(m['path']/'best.pt'),
                        onnx_sha256=digest(m['path']/'best.onnx')) for arm,m in models.items()}))
    print('Completed',args.out,flush=True)


if __name__ == '__main__':main()

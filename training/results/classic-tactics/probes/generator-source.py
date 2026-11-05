"""Counterfactual placement tests under frozen Classic v4 physics.

All interventions execute legal actions. Diagnostic branches use the original
1600-tick deadline, without the loop shortcut: a one-off path is not itself a
fixed stateless policy. Full learned-policy evaluation uses the normal rules.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from competition_eval import decisions, openings
from offense_eval import contact_paths
from pace_study import CONFIGS, PaceArena, decode, make_policy
from policy import ROOT

RULES = CONFIGS['pace13-v9-scaled']
OUT = ROOT/'training/results/classic-tactics/probes'
KINDS = ('attack', 'defend')
POLICIES = {kind: make_policy(RULES, kind) for kind in KINDS}
KNOWN_OLD_SEEDS = {3502446039, 3509813992, 3509869267, 3607230410, 3608270858, 3703494462}


def make_protocol(path=OUT/'protocol.json'):
    if path.exists():
        return json.loads(path.read_text())
    data = dict(rules='pong-v4', rule_config=asdict(RULES),
        pairs=[['attack', 'attack'], ['attack', 'defend']],
        sampling='Every unique flight-start physical state, both receiving sides, all five rallies.',
        branches='Unchanged scripts plus every exactly reachable legal contact center. Commit a deterministic shortest-direction-first legal path until contact, then resume the same scripts.',
        termination='Original elapsed ticks and 1600-tick deadline. No loop shortcut in any diagnostic branch.',
        labels='One strict loss-to-win branch per state, minimize action departures from the baseline pre-contact path, then center index. Original labels use the original controlled-side script on the same visited observations.',
        excluded_preexisting_seeds=sorted(KNOWN_OLD_SEEDS), seed_sets={})
    for phase, base, count, seed in [('train',3500000000,16,202611057), ('dev',3600000000,8,202611058), ('final',3700000000,16,202611059)]:
        rng = np.random.default_rng(seed)
        values=[]
        while len(values)<count:
            value=int(rng.integers(base,base+10000000))
            if value not in KNOWN_OLD_SEEDS and value not in values: values.append(value)
        data['seed_sets'][phase]=values
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n')
    return data


def state(env, i):
    return dict(ball=env.ball[i].astype(int).tolist(), paddles=env.paddles[i].astype(int).tolist(),
        tick=int(env.ticks[i]), returns=int(env.returns[i]))


def set_states(env, states):
    env.ball[:]=[s['ball'] for s in states]
    env.paddles[:]=[s['paddles'] for s in states]
    env.ticks[:]=[s['tick'] for s in states]
    env.returns[:]=[s['returns'] for s in states]
    for seen in env.seen: seen.clear()


def script_actions(env, kinds, active):
    chosen=np.ones((env.n,2),dtype=np.int64)
    for side in (0,1):
        obs=env.obs(side)
        for kind in KINDS:
            ids=active & (kinds[:,side]==kind)
            if ids.any(): chosen[ids,side]=POLICIES[kind](obs[ids])
    return chosen


def collect(seeds, pairs):
    frames,_=openings(seeds)
    frames=frames[:len(seeds)*5]
    all_frames=frames*len(pairs)
    kinds=np.array([pair for pair in pairs for _ in frames])
    env=PaceArena(len(all_frames),RULES,loop_detection=False)
    env.load(all_frames)
    done=np.full(env.n,-1)
    actions=np.full((env.n,1600),255,dtype=np.uint8)
    matches=[]
    for i,frame in enumerate(all_frames):
        local=i%len(frames)
        matches.append(dict(id=i, seed=int(seeds[local//5]), rally=local%5+1,
            kinds=kinds[i].tolist(), opening=state(env,i)))
    snapshots=[]
    seen=[set() for _ in range(env.n)]
    pending=np.ones(env.n,bool)
    for tick in range(1600):
        active=done<0
        for i in np.flatnonzero(active & pending):
            key=tuple(env.ball[i])+tuple(env.paddles[i])
            if key in seen[i]: continue
            seen[i].add(key)
            snap=state(env,i)
            snap.update(id=len(snapshots),match_id=int(i),seed=matches[i]['seed'],rally=matches[i]['rally'],
                side=int(env.ball[i,2]>0), kinds=kinds[i].tolist())
            snapshots.append(snap)
        chosen=script_actions(env,kinds,active)
        actions[active,tick]=(chosen[active,0]*3+chosen[active,1]).astype(np.uint8)
        terminal,hit=env.step(chosen,active)
        finish=active & (terminal>=0)
        done[finish]=terminal[finish]
        for i in np.flatnonzero(finish):
            matches[i].update(outcome=int(done[i]), terminal=state(env,i),
                actions=''.join(str(v) for v in actions[i,:int(env.ticks[i])]))
        pending=hit>=0
        if (done>=0).all(): break
    return matches,snapshots


def branch_options(snapshot):
    env=PaceArena(1,RULES,loop_detection=False)
    set_states(env,[snapshot])
    _,own,_,ticks,height=decode(env.obs(snapshot['side']))
    t,p,h=int(ticks[0]),int(own[0]),int(height[0])
    centers=np.arange(max(15,h-15),min(145,h+15)+1)
    legal=contact_paths()[t,p-15,centers-15,3]>=0
    return t,centers[legal].tolist()


def simulate_branches(snapshots, policy=None):
    rows=[]
    for snapshot in snapshots:
        ticks,centers=branch_options(snapshot)
        rows.extend(dict(snapshot=snapshot,target=center,flight=ticks)
            for center in ([None] if policy is not None else [None,*centers]))
    env=PaceArena(len(rows),RULES,loop_detection=False)
    set_states(env,[r['snapshot'] for r in rows])
    kinds=np.array([r['snapshot']['kinds'] for r in rows])
    sides=np.array([r['snapshot']['side'] for r in rows])
    targets=np.array([-1 if r['target'] is None else r['target'] for r in rows])
    flights=np.array([r['flight'] for r in rows])
    baseline=targets<0
    done=np.full(env.n,-1)
    center=np.full(env.n,-1)
    trace=np.full((env.n,1600),255,dtype=np.uint8)
    paths=contact_paths()
    for step in range(1600):
        active=done<0
        chosen=script_actions(env,kinds,active)
        if policy is not None:
            controlling=active & (step<flights)
            if controlling.any():
                obs=np.where(sides[:,None]==0,env.obs(0),env.obs(1))
                moves=decisions(policy,obs,controlling)
                selected=np.flatnonzero(controlling)
                chosen[selected,sides[selected]]=moves[selected]
        forcing=active & ~baseline & (step<flights)
        ids=np.flatnonzero(forcing)
        if len(ids):
            moves=paths[flights[ids]-step,env.paddles[ids,sides[ids]].astype(int)-15,targets[ids]-15,3]
            assert (moves>=0).all()
            chosen[ids,sides[ids]]=moves
        trace[active,step]=(chosen[active,0]*3+chosen[active,1]).astype(np.uint8)
        terminal,hit=env.step(chosen,active)
        contact=active & (step==flights-1)
        center[contact]=env.paddles[np.flatnonzero(contact),sides[contact]].astype(int)
        forced_contact=contact & ~baseline
        assert np.array_equal(center[forced_contact],targets[forced_contact])
        assert np.array_equal(hit[forced_contact],sides[forced_contact])
        finish=active & (terminal>=0)
        done[finish]=terminal[finish]
        if (done>=0).all(): break
    output=[]
    for i,row in enumerate(rows):
        count=int(env.ticks[i]-row['snapshot']['tick'])
        output.append(dict(snapshot_id=row['snapshot']['id'],target=row['target'],flight_ticks=row['flight'],
            contact_center=int(center[i]),outcome=int(done[i]),
            value=1 if done[i]==sides[i] else 0 if done[i]==2 else -1,
            terminal=state(env,i),actions=''.join(str(v) for v in trace[i,:count])))
    return output


def evaluate_policy(data, policy, batch=128):
    """Replace only the tested contact with learned actions; scripts then resume.

    The model receives the same six pre-move observations as in the arena.
    Whole learned-policy matches remain a separate and harder transfer test.
    """
    snapshots=data['snapshots']
    rows=[]
    for start in range(0,len(snapshots),batch):
        rows.extend(simulate_branches(snapshots[start:start+batch],policy=policy))
    candidates={s['id']:[] for s in snapshots}
    for branch in data['candidates']: candidates[branch['snapshot_id']].append(branch)
    summary=dict(states=len(rows),baseline_wins=0,baseline_losses=0,baseline_draws=0,
        model_wins=0,model_losses=0,model_draws=0,better=0,worse=0,same=0,
        strict_loss_to_win_opportunities=0,strict_opportunities_won=0,
        strict_opportunities_drawn=0,strict_opportunities_lost=0,
        strict_opportunities_selected_winning_center=0)
    for row in rows:
        branches=candidates[row['snapshot_id']]
        baseline=branches[0]
        label={-1:'losses',0:'draws',1:'wins'}
        summary['baseline_'+label[baseline['value']]]+=1
        summary['model_'+label[row['value']]]+=1
        delta=row['value']-baseline['value']
        summary['better' if delta>0 else 'worse' if delta<0 else 'same']+=1
        winners={b['target'] for b in branches[1:] if b['value']==1}
        opportunity=baseline['value']==-1 and bool(winners)
        row.update(baseline_value=baseline['value'],strict_opportunity=opportunity,
            winning_centers=sorted(winners),actions=row['actions'])
        if opportunity:
            summary['strict_loss_to_win_opportunities']+=1
            summary[{1:'strict_opportunities_won',0:'strict_opportunities_drawn',-1:'strict_opportunities_lost'}[row['value']]]+=1
            summary['strict_opportunities_selected_winning_center']+=int(row['contact_center'] in winners)
    return dict(summary=summary,rows=rows)


def path_samples(snapshot, branch):
    env=PaceArena(1,RULES,loop_detection=False)
    set_states(env,[snapshot])
    observations,labels,original=[],[],[]
    side=snapshot['side']
    for code in branch['actions'][:branch['flight_ticks']]:
        obs=env.obs(side)
        action=int(code)//3 if side==0 else int(code)%3
        observations.append(obs[0].copy());labels.append(action)
        original.append(int(POLICIES[snapshot['kinds'][side]](obs)[0]))
        env.step(np.array([[int(code)//3,int(code)%3]]))
    return observations,labels,original


def export_labels(snapshots, candidates, path):
    groups={s['id']:[] for s in snapshots}
    for branch in candidates: groups[branch['snapshot_id']].append(branch)
    obs,labels,original,seeds,ids=[],[],[],[],[]
    improved=[]
    for snapshot in snapshots:
        branches=groups[snapshot['id']]
        baseline=branches[0]
        assert baseline['target'] is None
        better=[b for b in branches[1:] if b['value']>baseline['value']]
        if better: improved.append(snapshot['id'])
        if baseline['value']!=-1: continue
        winners=[b for b in branches[1:] if b['value']==1]
        if not winners: continue
        side=snapshot['side']
        def tie(branch):
            departures=sum((int(a)//3 if side==0 else int(a)%3)!=(int(b)//3 if side==0 else int(b)%3)
                for a,b in zip(branch['actions'][:branch['flight_ticks']],baseline['actions'][:branch['flight_ticks']]))
            return departures,branch['target']
        chosen=min(winners,key=tie)
        x,y,z=path_samples(snapshot,chosen)
        obs.extend(x);labels.extend(y);original.extend(z)
        seeds.extend([snapshot['seed']]*len(y));ids.extend([snapshot['id']]*len(y))
        snapshot['selected_target']=chosen['target']
    np.savez_compressed(path,obs=np.asarray(obs,dtype=np.float32).reshape(-1,6),labels=np.asarray(labels,dtype=np.int64),
        original_labels=np.asarray(original,dtype=np.int64),source_seeds=np.asarray(seeds,dtype=np.uint32),
        branch_ids=np.asarray(ids,dtype=np.int64))
    return dict(snapshots=len(snapshots),branches=len(candidates),
        states_with_better_outcome=len(improved),loss_to_win_states=len(set(ids)),
        training_path_observations=len(labels),changed_labels=int(np.sum(np.asarray(labels)!=np.asarray(original))))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['train','dev','final'],default='train')
    parser.add_argument('--count',type=int)
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    torch.set_num_threads(1)
    protocol=make_protocol()
    seeds=protocol['seed_sets'][args.phase][:args.count]
    path=args.out/f'{args.phase}.json'
    if path.exists(): raise FileExistsError('Use a fresh output directory; retained results are immutable.')
    args.out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    matches,snapshots=collect(seeds,protocol['pairs'])
    print('Collected',len(matches),'matches and',len(snapshots),'unique flight starts',flush=True)
    candidates=[]
    for start in range(0,len(snapshots),args.batch):
        candidates.extend(simulate_branches(snapshots[start:start+args.batch]))
        print('Branched',min(len(snapshots),start+args.batch),'/',len(snapshots),'snapshots;',len(candidates),'branches;',round(time.monotonic()-started,1),'seconds',flush=True)
    summary=export_labels(snapshots,candidates,args.out/f'{args.phase}.npz')
    data=dict(phase=args.phase,seeds=seeds,rules='pong-v4',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol=protocol,summary=summary,matches=matches,snapshots=snapshots,candidates=candidates,
        seconds=time.monotonic()-started)
    path.write_text(json.dumps(data,separators=(',',':'))+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()

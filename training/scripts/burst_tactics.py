"""Stronger independent reply-envelope diagnostic for the burst prototype.

This controller enumerates every legal opposing reply angle after its candidate
return and plans energy-aware recovery. It is still finite-horizon, not solved
play. The frozen training teacher in burst_rules.py is intentionally unchanged.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from burst_rules import (DEFAULT, Config, DIRECTIONS, decode, make_policy,
    move_to, reflect, run_pair, shot, travel_table)


@lru_cache(maxsize=8)
def reply_envelopes(c=DEFAULT):
    stages = np.arange(5, c.ball_cap+.25, .5)
    height = np.arange(157)[:, None]
    offset = np.arange(-15, 16)[None, :]
    legal = (height-offset >= 15) & (height-offset <= 145)
    low, high = [], []
    for speed in stages:
        sx, sy = shot(offset, speed, c)
        landing = reflect(height+np.ceil(330/sx)*sy)
        low.append(np.min(np.where(legal, landing, 10000), axis=1))
        high.append(np.max(np.where(legal, landing, -10000), axis=1))
    return np.array(low), np.array(high)


def energy_after(distance, ticks, energy, c):
    bursts = np.ceil(np.maximum(0, distance-c.paddle_speed*ticks)/(c.boost_speed-c.paddle_speed))
    return np.clip(energy+ticks*c.recharge-bursts*(c.boost_cost+c.recharge), 0, c.energy_max).astype(int)


def make_safe_policy(c=DEFAULT):
    low, high = reply_envelopes(c)
    offsets = np.arange(-65, 66)[None, :]*2
    def policy(obs):
        vx, own, other, energies, ticks, height = decode(obs, c)
        energy = energies[:, 0]
        stage = np.clip(np.rint((np.abs(vx)-5)*2).astype(int), 0, len(low)-1)
        target = (low[stage, height]+high[stage, height])/2
        incoming = vx < 0
        centers = np.clip(own[:, None]+offsets, 15, 145)
        d = height[:, None]-centers
        distance = np.abs(centers-own[:, None])
        possible = (np.abs(d) <= 15) & (distance <= travel_table(c)[ticks, energy][:, None])
        sx, sy = shot(d, np.abs(vx[:, None]), c)
        flight = np.ceil(330/sx).astype(int)
        destination = reflect(height[:, None]+flight*sy).astype(int)
        next_stage = np.clip(np.rint((sx-5)*2).astype(int), 0, len(low)-1)
        lo, hi = low[next_stage, destination], high[next_stage, destination]
        midpoint = (lo+hi)/2
        at_contact = energy_after(distance, ticks[:, None], energy[:, None], c)
        budget = travel_table(c)[flight, at_contact]
        recovery = np.clip(midpoint, np.maximum(15, centers-budget), np.minimum(145, centers+budget))
        after_recovery = energy_after(np.abs(recovery-centers), flight, at_contact, c)
        reply_flight = np.ceil(330/np.minimum(c.ball_cap, sx+c.speed_step)).astype(int)
        catch = travel_table(c)[reply_flight, after_recovery]+15
        slack = catch-np.maximum(np.abs(lo-recovery), np.abs(hi-recovery))
        # Once all replies are covered, preserve energy and reduce movement.
        quality = np.minimum(slack, 8)+.02*after_recovery-.001*distance
        best = np.argmax(np.where(possible, quality, -1e9), axis=1)
        selected = centers[np.arange(len(obs)), best]
        fallback = np.clip(height, 15, 145)
        target = np.where(incoming, np.where(possible.any(axis=1), selected, fallback), target)
        return move_to(target, own, energy, ticks, c)
    policy.has_memory = False
    policy.description = dict(kind='reply-envelope defender', scope='one opposing reply with energy-aware recovery; not optimal')
    return policy


def exact_reachable(start, energy, ticks, c=DEFAULT):
    """Maximum remaining energy per exactly reachable paddle center.

    More energy weakly dominates less at a fixed center; retain only the maximum.
    Exhausted boost requests are dominated by the corresponding normal action.
    """
    values = np.full(131, -1, int)
    values[start-15] = energy
    centers = np.arange(15, 146)
    for _ in range(ticks):
        nxt = np.full(131, -1, int)
        for a in range(5):
            valid = (values >= 0) & ((a < 3) | (values >= c.boost_cost))
            speed = c.boost_speed if a >= 3 else c.paddle_speed
            position = np.clip(centers+DIRECTIONS[a]*speed, 15, 145).astype(int)-15
            remaining = values-c.boost_cost if a >= 3 else np.minimum(c.energy_max, values+c.recharge)
            np.maximum.at(nxt, position[valid], remaining[valid])
        values = nxt
    return values


def branch_checks(c=DEFAULT):
    # A legal capped flight reaches y156. Check every legal
    # action sequence via resource dominance, rather than a chosen controller.
    ticks, start, landing = 26, 80, 156
    rows = []
    for energy in (0, 8, 16, 32, 64, 96):
        reachable = exact_reachable(start, energy, ticks, c)
        centers = np.arange(15, 146)
        catch = (reachable >= 0) & (np.abs(centers-landing) <= 15)
        rows.append(dict(initial_energy=energy, can_return=bool(catch.any()),
            maximum_remaining_energy_at_return=int(reachable[catch].max()) if catch.any() else None,
            reachable_centers=centers[reachable >= 0].tolist()))
    # A deliberately poor early sprint and a saved burst share the same physical
    # flight but differ in whether a return exists. This isolates resource value;
    # it is not a proof that an attacker can force the earlier energy expenditure.
    return dict(incoming=dict(x=345,y=26,vx=-13,vy=5,own_paddle=80,ticks=ticks,landing=landing),
        energy_branches=rows,
        interpretation='Exact single-flight resource counterfactual. Does not prove forced multi-shot depletion.')


def best_paths(start, energy, ticks, c=DEFAULT):
    states = {start: (energy, [])}
    for _ in range(ticks):
        following = {}
        for position, (available, path) in states.items():
            for action in (1, 0, 2, 3, 4):
                if action >= 3 and available < c.boost_cost:continue
                speed = c.boost_speed if action >= 3 else c.paddle_speed
                dst = int(np.clip(position+DIRECTIONS[action]*speed, 15, 145))
                left = available-c.boost_cost if action >= 3 else min(c.energy_max, available+c.recharge)
                if dst not in following or left > following[dst][0]:
                    following[dst] = (left, path+[action])
        states = following
    return states


def actual_tactic_branches(record, c=DEFAULT):
    """Change the defender's legal return, then let both scripts continue.

    This shows counterplay within the observed pressure sequence. It does not
    establish a minimax win or loss against all future choices.
    """
    from burst_rules import BurstArena
    attack, safe = make_policy(c, 'attack'), make_safe_policy(c)
    frames = record['frames']
    last_contact = [f for i, f in enumerate(frames) if i and f['returns'] > frames[i-1]['returns']][-1]
    landing = int(reflect(last_contact['ball'][1]+26*last_contact['ball'][3]))
    victim = 1 if last_contact['ball'][2] > 0 else 0
    actual = exact_reachable(int(last_contact['paddles'][victim]), last_contact['energy'][victim], 26, c)
    full = exact_reachable(int(last_contact['paddles'][victim]), c.energy_max, 26, c)
    catchable = lambda row: bool(((row >= 0)&(np.abs(np.arange(15,146)-landing) <= 15)).any())
    summary = dict(final_shot=last_contact, destination=landing, receiver=victim,
        catchable_with_recorded_energy=catchable(actual), catchable_with_full_energy=catchable(full),
        counter_branches=[])
    for index in (727, 779, 831):
        start = next(f for f in frames if f['tick'] == index)
        candidates = best_paths(int(start['paddles'][1]), start['energy'][1], 26, c)
        for position in range(15, 20):
            if position not in candidates:continue
            energy, path = candidates[position]
            env = BurstArena(1, c, loop_detection=True)
            env.load([dict(x=start['ball'][0],y=start['ball'][1],vx=start['ball'][2],vy=start['ball'][3],
                paddles=start['paddles'],energy=start['energy'],rallyTick=index,returns=start['returns'])])
            for i, old in enumerate(frames):
                if old['tick'] > index:break
                if i and old['returns'] > frames[i-1]['returns']:
                    env.seen[0].add(tuple(old['ball'])+tuple(old['paddles'])+tuple(old['energy']))
            trajectory = []
            for tick in range(c.tick_limit-index):
                actions = [int(attack(env.obs())[0]), path[tick] if tick < 26 else int(safe(env.obs(1))[0])]
                result, _ = env.step(np.array([actions]))
                trajectory.append(dict(tick=int(env.ticks[0]),ball=env.ball[0].tolist(),paddles=env.paddles[0].tolist(),
                    energy=env.energy[0].tolist(),actions=actions,returns=int(env.returns[0])))
                if result[0] >= 0:break
            summary['counter_branches'].append(dict(start=start, target_center=position,
                energy_after_return=energy, planned_defender_actions=path, outcome=int(result[0]),
                draw_reason=int(env.draw_reason[0]), end_tick=int(env.ticks[0]), end_returns=int(env.returns[0]),
                frames=trajectory))
    return summary


def main():
    out = Path(__file__).resolve().parents[1]/'results/burst-study'
    seeds=(np.random.default_rng(20261213).choice(10_000_000,16,replace=False)+2_820_000_000).tolist()
    safe, attack = make_safe_policy(), make_policy(DEFAULT,'attack')
    result=dict(purpose='stronger-defender diagnostic, not held-out promotion proof',seeds=seeds,
        resource_branches=branch_checks(),pairs={})
    for label,a,b in [('attack:safe',attack,safe),('safe:safe',safe,safe),
                      ('safe:defend',safe,make_policy(DEFAULT,'defend'))]:
        row=run_pair(DEFAULT,a,b,seeds,capture_replays=2)
        result['pairs'][label]=row
        s=row['summary'];print(label,s['wins'],s['losses'],s['draws'],'median',s['median_decisive_returns'],flush=True)
        (out/'rules-tactics.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    result['actual_tactic']=actual_tactic_branches(result['pairs']['attack:safe']['replays'][0])
    (out/'rules-tactics.json').write_text(json.dumps(result,separators=(',',':'))+'\n')


if __name__ == '__main__':main()

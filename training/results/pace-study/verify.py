"""Independent scalar conformance and intent checks for the pace experiment."""
import hashlib
import json
from math import floor
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'training/scripts'))
from env import Arena
from pace_study import CONFIGS, PaceArena, OUT, make_policy, run_pair


def scalar(ball, paddles, tick, returns, selected, config, seen):
    x, y, vx, vy = map(int, ball)
    p = [max(15, min(145, int(q)+2*(int(a)-1))) for q, a in zip(paddles, selected)]
    x, y, tick = x+vx, y+vy, tick+1
    if y < 4:
        y, vy = 8-y, -vy
    elif y > 156:
        y, vy = 312-y, -vy
    terminal, hit, reason = -1, -1, 0
    if vx < 0 and x <= 15 or vx > 0 and x >= 345:
        side = int(vx > 0)
        d = y-p[side]
        if abs(d) > 15:
            terminal = 1-side
        else:
            speed = 5+floor(8*abs(d)/15+.5) if config.baseline else min(config.cap, abs(vx)+1)
            vertical = floor(config.vertical_max*speed*d/(15*config.cap)+.5) if config.scaled_angle else floor(config.vertical_max*d/15+.5)
            x, vx, vy = (15, speed, vertical) if side == 0 else (345, -speed, vertical)
            returns += 1
            hit = side
            key = (x,y,vx,vy,*p)
            if key in seen:
                terminal, reason = 2, 1
            seen.add(key)
    if tick >= 1600 and terminal < 0:
        terminal, reason = 2, 2
    return [x,y,vx,vy], p, tick, returns, terminal, hit, reason


def main():
    rng = np.random.default_rng(72914)
    counts = {}
    fixtures = []
    for name in ('baseline', 'pace13-v9-scaled'):
        config = CONFIGS[name]
        n = 512
        env = PaceArena(n, config, loop_detection=True)
        records = [(row.copy(), p.copy(), 0, 0, set()) for row,p in zip(env.ball, env.paddles)]
        active = np.ones(n, bool)
        transitions = 0
        for _ in range(100):
            actions = rng.integers(0, 3, (n, 2))
            terminal, hit = env.step(actions, active)
            for i in np.flatnonzero(active):
                ball, paddles, tick, returns, seen = records[i]
                b,p,t,r,end,h,why = scalar(ball,paddles,tick,returns,actions[i],config,seen)
                assert np.array_equal(b, env.ball[i])
                assert np.array_equal(p, env.paddles[i])
                assert (t,r,end,h,why) == (env.ticks[i],env.returns[i],terminal[i],hit[i],env.draw_reason[i])
                records[i] = b,p,t,r,seen
                transitions += 1
            active &= terminal < 0
        counts[name] = dict(random_scalar_transitions=transitions)
    config = CONFIGS['pace13-v9-scaled']
    # For any legal placement the speed comes from incoming speed, never offset.
    inputs = [(speed,y,p,action,side) for speed in range(5,14) for y in range(4,157)
              for p in range(15,146) for action in range(3) for side in (0,1)]
    a = np.array(inputs)
    speed,y,p,action,side = a.T
    n = len(a)
    env = PaceArena(n, config)
    env.ball[:] = np.column_stack([np.where(side == 0,16,344),y,np.where(side == 0,-speed,speed),np.zeros(n)])
    env.paddles[:] = 80
    env.paddles[np.arange(n),side] = p
    actions = np.ones((n,2), int)
    actions[np.arange(n),side] = action
    terminal, hit = env.step(actions)
    q = np.clip(p+2*(action-1),15,145)
    offset = y-q
    catches = abs(offset) <= 15
    assert np.array_equal(hit >= 0,catches)
    assert np.all(abs(env.ball[catches,2]) == np.minimum(13,speed[catches]+1))
    assert np.all(env.ball[catches,3] == np.floor(9*np.minimum(13,speed[catches]+1)*offset[catches]/195+.5))
    assert np.all(terminal[~catches] == 1-side[~catches])
    assert np.all(abs(env.ball[catches & (speed == 5),2]) == 6)
    assert np.all(abs(env.ball[catches & (speed == 13),2]) == 13)
    counts['pace13-v9-scaled']['exhaustive_contact_transitions'] = n
    # A same-position return at a new speed is not an exact loop.
    env = PaceArena(1,config,loop_detection=True)
    env.ball[0] = [16,80,-5,0]
    env.paddles[0] = [80,80]
    env.seen[0].add((15,80,5,0,80,80))
    terminal,_ = env.step(np.array([[1,1]]))
    assert terminal[0] == -1 and env.ball[0,2] == 6
    for prior_speed, prior_tick, own, expected, reason in [(13,1599,80,2,1),(5,1599,15,1,0),(5,1599,80,2,2)]:
        env = PaceArena(1,config,loop_detection=True)
        env.ball[0] = [16,80,-prior_speed,0]
        env.paddles[0] = [own,80]
        env.ticks[0] = prior_tick
        if prior_speed == 13:env.seen[0].add((15,80,13,0,80,80))
        terminal,_ = env.step(np.array([[1,1]]))
        assert (int(terminal[0]),int(env.draw_reason[0])) == (expected,reason)
    # Compare the experimental baseline against the untouched vector v3 engine.
    a,b = Arena(256,seed=17,loop_detection=True),PaceArena(256,CONFIGS['baseline'],seed=17,loop_detection=True)
    active = np.ones(256,bool)
    for _ in range(100):
        acts = rng.integers(0,3,(256,2))
        end1,hit1 = a.step(acts,active)
        end2,hit2 = b.step(acts,active)
        assert np.array_equal(a.ball,b.ball) and np.array_equal(a.paddles,b.paddles)
        assert np.array_equal(end1,end2) and np.array_equal(hit1,hit2)
        active &= end1 < 0
    for seed in (11,2026,4294967295):
        result = run_pair(config,make_policy(config,'attack'),make_policy(config,'defend'),[seed],capture_replays=5)
        fixtures.extend(result['replays'])
    (OUT/'fixtures.json').write_text(json.dumps(dict(config=vars(config),replays=fixtures),separators=(',',':'))+'\n')
    result = dict(checks=counts, baseline_vector_v3_parity=True, placement_independent_speed=True,
        first_return_is_6=True, speed_cap_is_13=True, speed_in_loop_key=True, point_loop_timeout_precedence=True,
        source_sha256=hashlib.sha256((ROOT/'training/scripts/pace_study.py').read_bytes()).hexdigest(),
        full_rally_fixtures=len(fixtures), fixture_seeds=[11,2026,4294967295])
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

"""Isolated fully observable burst-Pong prototype; released v4 is unchanged.

The geometric policies are diagnostics/teachers, never exported network layers.
"""
from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
import torch
from torch import nn

from competition_eval import openings, summarize, validate_seeds
from env import Arena, reflect


@dataclass(frozen=True)
class Config:
    name: str = 'burst96-cost8-half'
    paddle_speed: int = 2
    boost_speed: int = 4
    energy_max: int = 96
    boost_cost: int = 8
    recharge: int = 1
    speed_step: float = .5
    ball_cap: float = 13
    vertical_max: int = 9
    tick_limit: int = 1600


DEFAULT = Config()
CONFIGS = {DEFAULT.name: DEFAULT}
for capacity in (64, 96):
    for cost in (8, 12):
        c = Config(name=f'burst{capacity}-cost{cost}-half', energy_max=capacity, boost_cost=cost)
        CONFIGS[c.name] = c
DIRECTIONS = np.array([-1, 0, 1, -1, 1])


def shot(offset, incoming_speed, config=DEFAULT):
    speed = np.minimum(config.ball_cap, np.abs(incoming_speed) + config.speed_step)
    return speed, np.floor(config.vertical_max * speed * offset / (15 * config.ball_cap) + .5)


class BurstArena(Arena):
    def __init__(self, n, config=DEFAULT, seed=1, loop_detection=False):
        self.config = config
        self.energy = np.full((n, 2), config.energy_max, dtype=np.int64)
        super().__init__(n, seed=seed, loop_detection=loop_detection)

    def reset(self, ids):
        super().reset(ids)
        self.energy[ids] = self.config.energy_max

    def load(self, frames):
        super().load(frames)
        self.energy[:] = [f.get('energy', [self.config.energy_max] * 2) for f in frames]

    def obs(self, side=0):
        return np.concatenate([super().obs(side),
            self.energy[:, [side, 1-side]].astype(np.float32) / self.config.energy_max], axis=1)

    def step(self, actions, active=None):
        actions = np.asarray(actions)
        if actions.shape != (self.n, 2) or not np.isin(actions, np.arange(5)).all():
            raise ValueError('Expected one legal action 0..4 for each player.')
        if active is None:
            active = np.ones(self.n, bool)
        c = self.config
        requested = actions >= 3
        boosted = requested & (self.energy >= c.boost_cost)
        velocity = DIRECTIONS[actions] * np.where(boosted, c.boost_speed, c.paddle_speed)
        self.paddles[:] = np.where(active[:, None], np.clip(self.paddles + velocity, 15, 145), self.paddles)
        energy = np.where(boosted, self.energy-c.boost_cost,
                          np.where(requested, self.energy, np.minimum(c.energy_max, self.energy+c.recharge)))
        self.energy[:] = np.where(active[:, None], energy, self.energy)
        self.ball[:, :2] += self.ball[:, 2:] * active[:, None]
        self.ticks += active
        x, y, vx, vy = self.ball.T
        wall = active & ((y < 4) | (y > 156))
        y[wall] = np.where(y[wall] < 4, 8-y[wall], 312-y[wall])
        vy[wall] *= -1
        contact = active & (((vx < 0) & (x <= 15)) | ((vx > 0) & (x >= 345)))
        side = (vx > 0).astype(int)
        offset = y-self.paddles[np.arange(self.n), side]
        miss = contact & (np.abs(offset) > 15)
        hit = contact & ~miss
        speed, vertical = shot(offset[hit], vx[hit], c)
        x[hit] = np.where(side[hit] == 0, 15, 345)
        vx[hit] = np.where(side[hit] == 0, 1, -1)*speed
        vy[hit] = vertical
        self.returns[hit] += 1
        loop = np.zeros(self.n, bool)
        if self.loop_detection:
            for i in np.flatnonzero(hit):
                key = tuple(self.ball[i])+tuple(self.paddles[i])+tuple(self.energy[i])
                loop[i] = key in self.seen[i]
                self.seen[i].add(key)
        timeout = active & (self.ticks >= c.tick_limit) & ~miss & ~loop
        self.draw_reason[:] = np.where(loop, 1, np.where(timeout, 2, 0))
        return np.where(miss, 1-side, np.where(loop | timeout, 2, -1)), np.where(hit, side, -1)


@lru_cache(maxsize=12)
def travel_table(config=DEFAULT):
    """Exact maximum unclamped distance in t moves at each initial energy.

    Normal movement recharges. A requested but unavailable boost is dominated
    by a normal move for maximum distance and is deliberately excluded.
    """
    c = config
    energy = np.arange(c.energy_max+1)
    table = np.zeros((67, c.energy_max+1), dtype=int)
    for t in range(1, 67):
        normal = c.paddle_speed+table[t-1, np.minimum(c.energy_max, energy+c.recharge)]
        burst = np.where(energy >= c.boost_cost,
            c.boost_speed+table[t-1, np.maximum(0, energy-c.boost_cost)], -1)
        table[t] = np.maximum(normal, burst)
    return table


def decode(obs, config=DEFAULT):
    raw = np.asarray(obs, dtype=np.float64)
    x = np.rint((raw[:, 0]*180+180)*2)/2
    y = np.rint(raw[:, 1]*80+80)
    vx = np.rint(raw[:, 2]*12)/2
    vy = np.rint(raw[:, 3]*6)
    own = np.rint(raw[:, 4]*80+80).astype(int)
    other = np.rint(raw[:, 5]*80+80).astype(int)
    energy = np.clip(np.rint(raw[:, 6:8]*config.energy_max), 0, config.energy_max).astype(int)
    distance = np.where(vx < 0, x-15, 345-x)
    ticks = np.clip(np.ceil(distance/np.maximum(1, np.abs(vx))).astype(int), 1, 66)
    landing = reflect(y+ticks*vy).astype(int)
    return vx, own, other, energy, ticks, landing


def move_to(target, own, energy, ticks, config=DEFAULT, tolerance=0):
    """Save burst unless ordinary movement would miss the selected deadline."""
    c = config
    error = target-own
    action = np.where(error < -1, 0, np.where(error > 1, 2, 1))
    normal_next = np.clip(own+DIRECTIONS[action]*c.paddle_speed, 15, 145)
    next_energy = np.minimum(c.energy_max, energy+c.recharge)
    budget = travel_table(c)[np.maximum(0, ticks-1), next_energy]
    need = np.abs(target-normal_next) > budget+tolerance
    use = need & (energy >= c.boost_cost) & (action != 1)
    return np.where(use, np.where(action == 0, 3, 4), action).astype(np.int64)


def make_policy(config=DEFAULT, kind='attack', delay=0):
    if kind not in ('attack', 'defend', 'interceptor', 'tracker', 'wasteful', 'no-burst'):
        raise ValueError('Unknown policy kind')
    c = config
    history = deque(maxlen=delay+1)
    offsets = np.arange(-65, 66)[None, :]*2
    def policy(obs):
        if delay:
            history.append(obs.copy())
            old = history[0].copy()
            old[:, [4, 6]] = obs[:, [4, 6]]
            obs = old
        vx, own, other, energy, ticks, height = decode(obs, c)
        incoming = vx < 0
        target = np.full(len(obs), 80.)
        if kind == 'tracker':
            target = np.clip(np.rint(obs[:, 1]*80+80), 15, 145)
        else:
            target[incoming] = np.clip(height[incoming], 15, 145)
        if kind in ('attack', 'defend', 'no-burst'):
            centers = np.clip(own[:, None]+offsets, 15, 145)
            contact = height[:, None]-centers
            travel = np.abs(centers-own[:, None])
            budget = travel_table(c)[ticks, energy[:, 0]]
            if kind == 'no-burst':budget = c.paddle_speed*ticks
            valid = (np.abs(contact) <= 15) & (travel <= budget[:, None])
            speed, vertical = shot(contact, np.abs(vx[:, None]), c)
            flight = np.ceil(330/speed).astype(int)
            destination = reflect(height[:, None]+flight*vertical)
            # Opponent can recover toward center before this contact. We allow
            # its full movement budget and full recharge as a conservative
            # one-flight bound, not an optimal multi-exchange solution.
            opposing_budget = travel_table(c)[ticks, energy[:, 1]]
            opposing = np.clip(80, other-opposing_budget, other+opposing_budget)
            opposing_energy = np.minimum(c.energy_max, energy[:, 1]+ticks*c.recharge)
            reachable = travel_table(c)[flight[:, 0], opposing_energy]
            margin = np.abs(destination-opposing[:, None])-15-reachable[:, None]
            normal_needed = np.maximum(0, travel-c.paddle_speed*ticks[:, None])
            min_bursts = np.ceil(normal_needed/max(1, c.boost_speed-c.paddle_speed))
            remaining = np.clip(energy[:, 0, None]+ticks[:, None]*c.recharge-
                min_bursts*(c.boost_cost+c.recharge), 0, c.energy_max)
            # Demand on opponent energy matters before an outright winner.
            demand = np.maximum(0, np.abs(destination-opposing[:, None])-15-c.paddle_speed*flight)
            recovery = np.maximum(0, np.abs(centers-80)-c.paddle_speed*flight)
            if kind == 'defend':
                quality = -recovery + .04*remaining - .002*travel
            else:
                quality = margin + .2*demand - .2*recovery + .006*remaining - .001*travel
            best = np.argmax(np.where(valid, quality, -1e9), axis=1)
            planned = centers[np.arange(len(obs)), best]
            target = np.where(incoming & valid.any(axis=1), planned, target)
        action = move_to(target, own, energy[:, 0], ticks, c)
        if kind == 'wasteful':
            action = np.where(action == 0, 3, np.where(action == 2, 4, action))
        if kind == 'no-burst':
            action = np.where(action == 3, 0, np.where(action == 4, 2, action))
        return action
    policy.has_memory = bool(delay)
    policy.description = dict(kind=kind, delay_ticks=delay,
        scope='resource-aware finite-horizon heuristic, not optimal play')
    return policy


def decisions(policy, obs, active):
    selected = np.ones(len(obs), dtype=np.int64)
    if isinstance(policy, nn.Module):
        with torch.inference_mode():
            selected[active] = policy(torch.from_numpy(obs[active])).argmax(-1).numpy()
    else:
        selected[active] = policy(obs[active])
    if not np.isin(selected, np.arange(5)).all():raise ValueError('Illegal policy action')
    return selected


def run_pair(config, policy_a, policy_b, seeds, capture_replays=0):
    seeds = validate_seeds(seeds)
    frames, sides = openings(seeds)
    loops = not (getattr(policy_a, 'has_memory', False) or getattr(policy_b, 'has_memory', False))
    env = BurstArena(len(frames), config, loop_detection=loops)
    env.load(frames)
    n = env.n
    receiver = (env.ball[:, 2] > 0).astype(int)
    outcome = np.full(n, -1)
    lengths, durations, reasons = (np.zeros(n, int) for _ in range(3))
    boosted_moves = np.zeros((n, 2), int)
    replays = [dict(index=i, side=int(sides[i]), seed=seeds[(i//5)%len(seeds)], rally=i%5+1, frames=[])
               for i in range(min(capture_replays, n))]
    for tick in range(config.tick_limit):
        active = outcome < 0
        left, right = env.obs(), env.obs(1)
        ours = np.where(sides[:, None] == 0, left, right)
        theirs = np.where(sides[:, None] == 0, right, left)
        a, b = decisions(policy_a, ours, active), decisions(policy_b, theirs, active)
        chosen = np.stack([np.where(sides == 0, a, b), np.where(sides == 0, b, a)], 1)
        boosted_moves += ((chosen >= 3) & (env.energy >= config.boost_cost) & active[:, None])
        for replay in replays:
            i = replay['index']
            if active[i]:
                replay['frames'].append(dict(tick=int(env.ticks[i]), ball=env.ball[i].tolist(),
                    paddles=env.paddles[i].tolist(), energy=env.energy[i].tolist(),
                    actions=chosen[i].tolist(), returns=int(env.returns[i])))
        terminal, hit = env.step(chosen, active)
        done = active & (terminal >= 0)
        outcome[done], lengths[done], durations[done] = terminal[done], env.returns[done], env.ticks[done]
        reasons[done] = env.draw_reason[done]
        for replay in replays:
            i = replay['index']
            if done[i]:
                replay['frames'].append(dict(tick=int(env.ticks[i]), ball=env.ball[i].tolist(),
                    paddles=env.paddles[i].tolist(), energy=env.energy[i].tolist(), actions=None,
                    returns=int(env.returns[i]), outcome=int(outcome[i]), draw_reason=int(reasons[i])))
        if (outcome >= 0).all():break
    bootstrap = np.random.default_rng(20261211).integers(0, len(seeds), (2000, len(seeds)))
    summary = summarize(outcome, sides, lengths, durations, reasons, seeds, bootstrap)
    decisive = outcome < 2
    summary.update(first_receiver_win_rate=float((outcome[decisive] == receiver[decisive]).mean()) if decisive.any() else None,
        mean_burst_ticks_per_rally=float(boosted_moves.sum(1).mean()), loop_detection_enabled=loops)
    records = dict(outcome=outcome.tolist(), candidate_side=sides.tolist(), first_receiver=receiver.tolist(),
        returns=lengths.tolist(), ticks=durations.tolist(), draw_reason=reasons.tolist(), burst_ticks=boosted_moves.tolist())
    for replay in replays:
        i = replay['index']
        replay.update(outcome=int(outcome[i]), draw_reason=int(reasons[i]), returns=int(lengths[i]), ticks=int(durations[i]))
    return dict(rules=asdict(config), seeds=seeds, seed_count=len(seeds), summary=summary, records=records, replays=replays)

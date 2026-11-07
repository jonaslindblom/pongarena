"""Pong physics: a scalar reference and a vectorized training environment."""
import copy
import math
import struct
import numpy as np

MAX_BALL_VX = 13
MAX_BALL_VY = 9
RALLY_TICK_LIMIT = 1600


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def js_round(value):
    return math.floor(value + 0.5)


def serve(seed, prior=None):
    n = int(seed) & 0xffffffff

    def random():
        nonlocal n
        n = (1664525 * n + 1013904223) & 0xffffffff
        return n / 4294967296

    def sign():
        return -1 if random() < 0.5 else 1

    first_vx = 5 * sign()
    a, b = sign(), sign()
    directions = [a, b, -a, -b, sign()]
    openings = [(center + math.floor(random() * 21) - 10,
                 1 + math.floor(random() * 5)) for center in [32, 56, 80, 104, 128]]
    for i in range(4, 0, -1):
        j = math.floor(random() * (i + 1))
        openings[i], openings[j] = openings[j], openings[i]
    rally = prior['rally'] + 1 if prior else 1
    y, speed = openings[rally - 1]
    return dict(tick=prior['tick'] + 1 if prior else 0, rally=rally, rallyTick=0,
                x=180, y=y, vx=first_vx if rally % 2 else -first_vx,
                vy=speed * directions[rally - 1], paddles=[80, 80],
                points=prior['points'].copy() if prior else [0, 0],
                returns=0, event='serve', scores=[None, None])


def observation(f, side):
    mirror = 1 if side == 0 else -1
    return list(map(f32, [mirror * (f['x'] - 180) / 180, (f['y'] - 80) / 80,
                         mirror * f['vx'] / 6, f['vy'] / 6,
                         (f['paddles'][side] - 80) / 80,
                         (f['paddles'][1 - side] - 80) / 80]))


def choice(scores):
    if len(scores) != 3 or not all(math.isfinite(x) for x in scores):
        raise ValueError('Expected three finite action scores')
    return scores.index(max(scores))


class LoopDetector:
    """Exact contact-state repetition for fixed deterministic stateless policies.

    Never use this shortcut for humans, sampled actions, changing weights or
    untracked controller memory. Create/reset on restart or controller change.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.rally = None
        self.seen = set()

    def observe(self, frame):
        if frame['rally'] != self.rally:
            self.reset()
            self.rally = frame['rally']
        if frame['event'] != 'return':
            return False
        key = (frame['x'], frame['y'], frame['vx'], frame['vy'], *frame['paddles'])
        repeated = key in self.seen
        self.seen.add(key)
        return repeated


def advance(prior, scores, loops=None):
    f = copy.deepcopy(prior)
    f.pop('actor', None)
    f.pop('drawReason', None)
    f.update(tick=prior['tick'] + 1, rallyTick=prior['rallyTick'] + 1,
             x=prior['x'] + prior['vx'], y=prior['y'] + prior['vy'],
             paddles=[max(15, min(145, p + 2 * (choice(s) - 1)))
                      for p, s in zip(prior['paddles'], scores)], event=None, scores=scores)
    if f['y'] < 4 or f['y'] > 156:
        f['y'] = 8 - f['y'] if f['y'] < 4 else 312 - f['y']
        f['vy'] *= -1
    if (f['vx'] < 0 and f['x'] <= 15) or (f['vx'] > 0 and f['x'] >= 345):
        side = 0 if f['vx'] < 0 else 1
        offset = f['y'] - f['paddles'][side]
        if abs(offset) > 15:
            f['actor'] = 1 - side
            f['points'][1 - side] += 1
            f['event'] = 'point'
        else:
            speed = min(MAX_BALL_VX, abs(f['vx']) + 1)
            f.update(x=15 if side == 0 else 345,
                     vx=(1 if side == 0 else -1) * speed,
                     vy=js_round(MAX_BALL_VY * speed * offset / (15 * MAX_BALL_VX)),
                     returns=f['returns'] + 1, actor=side, event='return')
    if loops is not None and loops.observe(f):
        f['event'] = 'draw'
        f['drawReason'] = 'loop'
        f.pop('actor', None)
    if f['rallyTick'] >= RALLY_TICK_LIMIT and f['event'] not in ('point', 'draw'):
        f['event'] = 'draw'
        f['drawReason'] = 'timeout'
        f.pop('actor', None)
    return f


def ended_rally(f):
    return f['event'] in ('point', 'draw')


def ended_match(f):
    return f['rally'] == 5 and ended_rally(f)


class Arena:
    def __init__(self, n, seed=1, loop_detection=False):
        self.loop_detection = loop_detection
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.ball = np.zeros((n, 4), dtype=np.float64)
        self.paddles = np.full((n, 2), 80., dtype=np.float64)
        self.ticks = np.zeros(n, dtype=np.int64)
        self.returns = np.zeros(n, dtype=np.int64)
        self.rallies = np.zeros(n, dtype=np.int64)
        self.seeds = np.zeros(n, dtype=np.uint32)
        self.seen = [set() for _ in range(n)]
        # 0: not a draw, 1: exact loop, 2: time limit.
        self.draw_reason = np.zeros(n, dtype=np.int8)
        self.reset(np.arange(n))

    def reset(self, ids):
        for i in ids:
            self.seen[i].clear()
            self.rallies[i] = self.rallies[i] % 5 + 1
            if self.rallies[i] == 1:
                # Validation and final test use separate uint32 ranges.
                self.seeds[i] = self.rng.integers(0, 1_000_000_000)
            f = serve(int(self.seeds[i]), None if self.rallies[i] == 1 else
                      {'rally': int(self.rallies[i]) - 1, 'tick': 0, 'points': [0, 0]})
            self.ball[i] = [f[k] for k in ('x', 'y', 'vx', 'vy')]
        self.paddles[ids] = 80
        self.ticks[ids] = 0
        self.returns[ids] = 0
        self.draw_reason[ids] = 0

    def load(self, frames):
        self.ball[:] = [[f[k] for k in ('x', 'y', 'vx', 'vy')] for f in frames]
        self.paddles[:] = [f['paddles'] for f in frames]
        self.ticks[:] = [f['rallyTick'] for f in frames]
        self.returns[:] = [f['returns'] for f in frames]
        for states in self.seen:
            states.clear()
        self.draw_reason[:] = 0

    def obs(self, side=0):
        mirror = 1 if side == 0 else -1
        x, y, vx, vy = self.ball.T
        return np.stack([mirror * (x - 180) / 180, (y - 80) / 80,
                         mirror * vx / 6, vy / 6, (self.paddles[:, side] - 80) / 80,
                         (self.paddles[:, 1 - side] - 80) / 80], axis=1).astype(np.float32)

    def step(self, actions, active=None):
        if active is None:
            active = np.ones(self.n, dtype=bool)
        self.paddles[:] = np.where(active[:, None], np.clip(self.paddles + 2 * (actions - 1), 15, 145), self.paddles)
        self.ball[:, :2] += self.ball[:, 2:] * active[:, None]
        self.ticks += active
        x, y, vx, vy = self.ball.T
        wall = active & ((y < 4) | (y > 156))
        y[wall] = np.where(y[wall] < 4, 8 - y[wall], 312 - y[wall])
        vy[wall] *= -1
        contact = active & (((vx < 0) & (x <= 15)) | ((vx > 0) & (x >= 345)))
        side = (vx > 0).astype(int)
        offset = y - self.paddles[np.arange(self.n), side]
        miss = contact & (np.abs(offset) > 15)
        hit = contact & ~miss
        x[hit] = np.where(side[hit] == 0, 15, 345)
        speed = np.minimum(MAX_BALL_VX, np.abs(vx[hit]) + 1)
        vx[hit] = np.where(side[hit] == 0, 1, -1) * speed
        vy[hit] = np.floor(MAX_BALL_VY * speed * offset[hit] / (15 * MAX_BALL_VX) + .5)
        self.returns[hit] += 1
        loop = np.zeros(self.n, dtype=bool)
        if self.loop_detection:
            for i in np.flatnonzero(hit):
                key = tuple(self.ball[i]) + tuple(self.paddles[i])
                loop[i] = key in self.seen[i]
                self.seen[i].add(key)
        timeout = active & (self.ticks >= RALLY_TICK_LIMIT) & ~miss & ~loop
        self.draw_reason[:] = np.where(loop, 1, np.where(timeout, 2, 0))
        outcome = np.where(miss, 1 - side, np.where(loop | timeout, 2, -1))
        return outcome, np.where(hit, side, -1)


def reflect(y):
    z = (y - 4) % 304
    return 4 + np.where(z <= 152, z, 304 - z)

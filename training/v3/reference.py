"""Pong v3 scalar reference. Standard library only; no UI or ML dependency."""
import copy
import math
import struct

MAX_BALL_VX = 13
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
            f.update(x=15 if side == 0 else 345,
                     vx=(1 if side == 0 else -1) * (5 + js_round((MAX_BALL_VX - 5) * abs(offset) / 15)),
                     vy=max(-5, min(5, js_round(offset / 3))),
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

"""Isolated offensive-balance candidates; the released Pong v3 stays unchanged."""
from dataclasses import dataclass

import numpy as np

from env import Arena, RALLY_TICK_LIMIT


@dataclass(frozen=True)
class Rules:
    name: str
    vertical_max: int = 5
    deflection: int = 0


CONFIGS = {
    'baseline': Rules('baseline'),
    'angle8': Rules('angle8', vertical_max=8),
    'angle9': Rules('angle9', vertical_max=9),
    'motion2': Rules('motion2', deflection=2),
}


def shot(offset, movement, rules):
    """Return (positive horizontal speed, signed vertical speed) for a hit.

    movement is the sign of the hitter's actual clamped displacement this tick.
    Call only for legal contacts, abs(offset) <= 15. No input/output clipping.
    """
    offset = np.asarray(offset)
    return (5 + np.floor(8 * np.abs(offset) / 15 + .5),
            np.floor(rules.vertical_max * offset / 15 + .5) + rules.deflection * np.asarray(movement))


class OffenseArena(Arena):
    """Same action, observation, reset, termination and mask API as Arena."""
    def __init__(self, n, rules, seed=1, loop_detection=False):
        super().__init__(n, seed=seed, loop_detection=loop_detection)
        self.rules = rules

    def step(self, actions, active=None):
        if active is None:
            active = np.ones(self.n, dtype=bool)
        before = self.paddles.copy()
        self.paddles[:] = np.where(active[:, None], np.clip(self.paddles + 2 * (actions - 1), 15, 145), self.paddles)
        movement = np.sign(self.paddles - before)
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
        speed_x, speed_y = shot(offset[hit], movement[np.flatnonzero(hit), side[hit]], self.rules)
        x[hit] = np.where(side[hit] == 0, 15, 345)
        vx[hit] = np.where(side[hit] == 0, 1, -1) * speed_x
        vy[hit] = speed_y
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

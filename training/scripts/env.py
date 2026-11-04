"""Vectorized Pong v2. Episodes end per rally; all five seeded openings are used."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v2'))
from reference import serve, MAX_BALL_VX


class Arena:
    def __init__(self, n, seed=1, max_vx=MAX_BALL_VX):
        self.max_vx = max_vx
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.ball = np.zeros((n, 4), dtype=np.float64)
        self.paddles = np.full((n, 2), 80., dtype=np.float64)
        self.ticks = np.zeros(n, dtype=np.int64)
        self.returns = np.zeros(n, dtype=np.int64)
        self.rallies = np.zeros(n, dtype=np.int64)
        self.seeds = np.zeros(n, dtype=np.uint32)
        self.reset(np.arange(n))

    def reset(self, ids):
        for i in ids:
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

    def load(self, frames):
        self.ball[:] = [[f[k] for k in ('x', 'y', 'vx', 'vy')] for f in frames]
        self.paddles[:] = [f['paddles'] for f in frames]
        self.ticks[:] = [f['rallyTick'] for f in frames]
        self.returns[:] = [f['returns'] for f in frames]

    def obs(self, side=0):
        mirror = 1 if side == 0 else -1
        x, y, vx, vy = self.ball.T
        return np.stack([mirror * (x - 180) / 180, (y - 80) / 80,
                         mirror * vx / 6, vy / 6, (self.paddles[:, side] - 80) / 80,
                         (self.paddles[:, 1 - side] - 80) / 80], axis=1).astype(np.float32)

    def step(self, actions):
        self.paddles[:] = np.clip(self.paddles + 2 * (actions - 1), 15, 145)
        self.ball[:, :2] += self.ball[:, 2:]
        self.ticks += 1
        x, y, vx, vy = self.ball.T
        wall = (y < 4) | (y > 156)
        y[:] = np.where(y < 4, 8 - y, np.where(y > 156, 312 - y, y))
        vy[wall] *= -1
        contact = ((vx < 0) & (x <= 15)) | ((vx > 0) & (x >= 345))
        side = (vx > 0).astype(int)
        offset = y - self.paddles[np.arange(self.n), side]
        miss = contact & (np.abs(offset) > 15)
        hit = contact & ~miss
        x[hit] = np.where(side[hit] == 0, 15, 345)
        vx[hit] = np.where(side[hit] == 0, 1, -1) * (5 + np.floor((self.max_vx - 5) * np.abs(offset[hit]) / 15 + .5))
        vy[hit] = np.clip(np.floor(offset[hit] / 3 + .5), -5, 5)
        self.returns[hit] += 1
        outcome = np.where(miss, 1 - side, np.where(self.ticks >= 800, 2, -1))
        return outcome, np.where(hit, side, -1)


def reflect(y):
    z = (y - 4) % 304
    return 4 + np.where(z <= 152, z, 304 - z)


def scripted(obs, attack=False, max_vx=MAX_BALL_VX):
    """Exact-flight reference using only the six exposed inputs. No hidden state."""
    x = np.rint(obs[:, 0].astype(np.float64) * 180 + 180)
    y = np.rint(obs[:, 1].astype(np.float64) * 80 + 80)
    vx = np.rint(obs[:, 2].astype(np.float64) * 6)
    vy = np.rint(obs[:, 3].astype(np.float64) * 6)
    own = np.rint(obs[:, 4].astype(np.float64) * 80 + 80)
    other = np.rint(obs[:, 5].astype(np.float64) * 80 + 80)
    ticks = np.maximum(1, np.ceil((x - 15) / np.maximum(1, -vx)))
    landing = reflect(y + ticks * vy)
    target = np.clip(landing, 15, 145)
    if attack:
        # Enumerate reachable paddle centers on the two-unit movement lattice.
        moves = np.arange(-65, 66)[None, :]
        centers = np.clip(own[:, None] + 2 * moves, 15, 145)
        offset = landing[:, None] - centers
        speed_x = 5 + np.floor((max_vx - 5) * np.abs(offset) / 15 + .5)
        speed_y = np.clip(np.floor(offset / 3 + .5), -5, 5)
        flight = np.ceil(330 / speed_x)
        destination = reflect(landing[:, None] + flight * speed_y)
        margin = np.abs(destination - other[:, None]) - 15 - 2 * flight
        valid = (np.abs(moves) <= ticks[:, None]) & (np.abs(offset) <= 15)
        # Prefer shorter paddle travel on equal attacking margin.
        quality = np.where(valid, margin - .001 * np.abs(centers - own[:, None]), -1e9)
        best = np.argmax(quality, axis=1)
        target = np.where(valid.any(axis=1), centers[np.arange(len(obs)), best], target)
    target = np.where(vx < 0, target, 80)
    error = target - own
    return np.where(error < -1, 0, np.where(error > 1, 2, 1)).astype(np.int64)

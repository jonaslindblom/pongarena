"""Deterministic geometric teachers using the same six observations as models.

Attack aims beyond the opponent's reach; defend favors safe recovery. Interceptor
simply catches and recovers. These are training/evaluation tools, not networks.
"""
from functools import lru_cache
import numpy as np
from game import reflect

CENTERS = np.arange(15, 146)
# Centers and action movements are across-court Y positions in court units.
NEXT = np.clip(CENTERS[:, None] + [-2, 0, 2], 15, 145).astype(int)
MAX_FLIGHT = 66
OFFSETS = np.arange(-15, 16)

@lru_cache(maxsize=1)
def contact_paths():
    """First action reaching (final center, actual final movement) in t ticks.

    Last axis -1/0/+1 uses indices 0/1/2; index3 permits any final movement.
    -1 denotes unreachable. Every path has exactly t actions (stays allowed).
    """
    first = np.full((MAX_FLIGHT + 1, 131, 131, 4), -1, dtype=np.int8)
    p, q = CENTERS[:, None], CENTERS[None, :]
    preference = np.stack([np.where(q < p, 0, np.where(q > p, 2, 1)),
                           np.ones((131, 131), int),
                           np.where(q < p, 2, 0)], axis=-1)
    preference[q == p] = [1, 0, 2]
    for action in range(3):
        dst = NEXT[:, action] - 15
        motion = np.sign(NEXT[:, action] - CENTERS).astype(int) + 1
        first[1, np.arange(131), dst, motion] = action
        first[1, np.arange(131), dst, 3] = action
    for ticks in range(2, MAX_FLIGHT + 1):
        for rank in range(3):
            action = preference[:, :, rank]
            following = NEXT[np.arange(131)[:, None], action] - 15
            can = first[ticks - 1, following, np.arange(131)[None, :], :] >= 0
            first[ticks] = np.where((first[ticks] < 0) & can, action[:, :, None], first[ticks])
    return first


@lru_cache(maxsize=1)
def tables():
    # Legacy name: height is the ball's across-court Y position at contact.
    height = np.arange(157)[:, None]
    centers = height - OFFSETS
    legal = (centers >= 15) & (centers <= 145) & (height >= 4)
    speeds = np.minimum(13, np.arange(13 + 1) + 1)
    sy = np.floor(9 * speeds[:, None] * OFFSETS / (15 * 13) + .5)
    flights = np.ceil(330 / speeds).astype(int)
    landing = reflect(height[None] + flights[:, None, None] * sy[:, None, :]).astype(int)
    low = np.maximum(15, np.max(np.where(legal, landing - 15 - 2 * flights[:, None, None], -10000), axis=2))
    high = np.minimum(145, np.min(np.where(legal, landing + 15 + 2 * flights[:, None, None], 10000), axis=2))
    mid = (low + high) / 2
    risks = np.zeros_like(landing, float)
    for incoming in range(5, 13 + 1):
        outgoing = min(13, incoming + 1)
        destinations = landing[incoming]
        recover = np.clip(mid[outgoing, destinations], np.maximum(15, centers - 2*flights[incoming]),
                          np.minimum(145, centers + 2*flights[incoming]))
        risks[incoming] = np.maximum(low[outgoing, destinations] - recover, recover - high[outgoing, destinations])
    return dict(centers=centers, legal=legal, flight=flights, landing=landing, low=low, high=high, mid=mid, risk=risks)


def decode(obs):
    # Decode normalized inputs into court units in our paddle's view: X runs
    # along the court (own contact at 15), Y across it; velocities are units/tick.
    raw = obs.astype(np.float64)
    x = np.rint(raw[:, 0]*180 + 180).astype(int)
    y = np.rint(raw[:, 1]*80 + 80).astype(int)
    vx = np.rint(raw[:, 2]*6).astype(int)
    vy = np.rint(raw[:, 3]*6).astype(int)
    own = np.rint(raw[:, 4]*80 + 80).astype(int)
    other = np.rint(raw[:, 5]*80 + 80).astype(int)
    distance = np.where(vx < 0, x-15, 345-x)
    ticks = np.clip(np.ceil(distance/np.maximum(1, np.abs(vx))).astype(int), 1, 66)
    landing = reflect(y + ticks*vy).astype(int)
    return vx, own, other, ticks, landing



def teacher(kind='attack'):
    if kind not in ('attack', 'defend', 'interceptor'):
        raise ValueError('Choose attack, defend or interceptor')
    tab, paths = tables(), contact_paths()

    def policy(obs):
        # Legacy name: height is the predicted across-court Y landing position.
        vx, own, other, ticks, height = decode(obs)
        stage = np.minimum(13, np.abs(vx))
        target = tab['mid'][stage, height]
        result = np.argmin(np.abs(NEXT[own-15] - target[:, None]) + [1e-6, 0, 1e-6], axis=1)
        ids = np.flatnonzero(vx < 0)
        if not len(ids):
            return result.astype(np.int64)
        h, p, t, speed = height[ids], own[ids], ticks[ids], stage[ids]
        fallback = np.argmin(np.abs(NEXT[p-15] - np.clip(h, 15, 145)[:, None]) + [1e-6, 0, 1e-6], axis=1)
        if kind == 'interceptor':
            result[ids] = fallback
            return result.astype(np.int64)
        centers = tab['centers'][h]
        first = paths[t[:, None], p[:, None]-15, np.clip(centers-15, 0, 130), 3]
        valid = tab['legal'][h] & (first >= 0)
        travel = np.abs(centers-p[:, None])
        if kind == 'defend':
            quality = -tab['risk'][speed, h] - .001*travel
        else:
            expected = np.clip(tab['mid'][speed, h], other[ids]-2*np.maximum(0, t-1), other[ids]+2*np.maximum(0, t-1))
            margin = np.abs(tab['landing'][speed, h] - expected[:, None]) - 15 - 2*tab['flight'][speed, None] - 2
            quality = margin - .05*np.maximum(0, tab['risk'][speed, h]) - .001*travel
        best = np.argmax(np.where(valid, quality, -1e12), axis=1)
        result[ids] = np.where(valid.any(axis=1), first[np.arange(len(ids)), best], fallback)
        return result.astype(np.int64)

    return policy

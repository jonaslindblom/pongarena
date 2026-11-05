# Offensive-balance experiment: reproducible rules

These are isolated candidate rules, not a new arena release. The published game remains Pong v3. `baseline` reproduces v3. Each other candidate changes only the outgoing vertical velocity at a paddle contact. The candidates do not add paddle inertia, persistent spin, hidden state, extra controls, or a different observation format.

| Candidate identifier | Vertical factor `V` | Movement deflection `K` |
| --- | ---: | ---: |
| `baseline` | 5 | 0 |
| `angle8` | 8 | 0 |
| `angle9` | 9 | 0 |
| `motion2` | 5 | 2 |

## State, actions and observations

The court is 360 by 160 units. Coordinates increase rightward and downward. One tick is 1/30 second for presentation; rendering does not control simulation. Ball state is `(x, y, vx, vy)`. Paddle centers are `(p0, p1)`, left then right. Every reachable physical value is an integer. Paddles remain within `[15,145]`. Ball walls are `y=4` and `y=156`; paddle contact planes are `x=15` and `x=345`.

Each controller receives the same pre-tick state. It produces three finite action scores. The first maximum wins: index 0 means up, 1 means stay, and 2 means down. A paddle moves at most two units per tick. Both actions take effect simultaneously; neither controller sees the other's current action before choosing its own.

For side `s` (0 left, 1 right), let `m=+1` for left or `-1` for right. The six observations, in order, are:

```
[m*(x-180)/180, (y-80)/80, m*vx/6, vy/6,
 (p_s-80)/80, (p_(1-s)-80)/80]
```

Convert each observation to IEEE-754 float32. Do not clip them: for example, `angle9` can emit `vy/6 = 1.5`. Physics uses exact integer-valued state, not reconstructed observations. A plain policy's entire executable behavior remains these six inputs and three scores.

## Exact tick transition

Define `R(z) = floor(z + 0.5)` and `clamp(z,l,h) = max(l,min(h,z))`. Python's built-in `round` is not this rounding rule. Negative half ties round toward positive infinity. There are no half ties in these candidates' reachable integer-offset formulas.

1. Store the old paddle centers. For each side, update `p_s = clamp(old_p_s + 2*(action_s-1),15,145)`. Define its actual movement `u_s = sign(p_s-old_p_s)`. Thus an up action at 15 has `u=0`; an up action at 16 moves to 15 and has `u=-1`, despite moving only one unit.
2. Move the ball using its incoming velocity: `x += vx`, `y += vy`. Increment the rally tick count.
3. Reflect the ball at a wall if it has crossed it. If `y<4`, set `y=8-y` and negate `vy`. Otherwise, if `y>156`, set `y=312-y` and negate `vy`. At exactly 4 or 156, do not reflect yet. Candidate speeds cannot cross both walls within one tick.
4. Test a paddle contact: left when `vx<0 and x<=15`; right when `vx>0 and x>=345`. Use the actual post-wall-reflection `y`; do not interpolate back to the crossing point. Let the contacted side be `s`, and `d=y-p_s` using its updated paddle center.
5. If `abs(d)>15`, the opposite side scores one point and the rally ends. Equality is a hit.
6. Otherwise reset `x` to 15 for left or 345 for right and set:

```
vx = (+1 for left, -1 for right) * (5 + R(8*abs(d)/15))
vy = R(V*d/15) + K*u_s
```

Increment the return counter. The deflection changes this return once; subsequent motion is straight until a wall or paddle contact. There is no additional vertical-speed clamp. Horizontal return speeds remain 5 through 13 in all four candidates.

7. Resolve termination in this order: point, exact-loop draw, then timeout draw. A rally that has neither scored nor already drawn ends in a draw at tick 1,600. A hit exactly on the deadline therefore draws unless it also repeats an earlier contact state, in which case its reason is `loop`. A miss exactly on the deadline scores normally.

For fixed, deterministic, stateless controllers only, record the complete post-return physical tuple `(x,y,vx,vy,p0,p1)`. Repetition during the same rally proves a loop and draws immediately. Clear this set on each new rally, restart, or controller change. Tick and return counters do not affect policy inputs or physics. Disable this shortcut for humans, sampled training actions, changing networks, or untracked controller memory.

## Serves and matches

Every match contains five rallies. A point awards one to the winner; a draw awards neither side. After all five, compare points; equal totals are a tied match. Before every rally, reset both paddle centers to 80 and the rally tick and return counts to zero. Ball `x` is 180. The same match seed and rally index reproduce the opening.

The following generator defines all five openings. Each call rebuilds the sequence from the unsigned 32-bit match seed. Multiplication and addition use modulo `2**32` arithmetic:

```
n = seed modulo 2**32
random():
    n = (1664525*n + 1013904223) modulo 2**32
    return n / 4294967296
sign(): return -1 if random() < 0.5 else +1

first_vx = 5*sign()
a = sign()
b = sign()
directions = [a,b,-a,-b,sign()]
openings = []
for center in [32,56,80,104,128]:
    openings.append((center+floor(random()*21)-10,
                     1+floor(random()*5)))
for i in [4,3,2,1]:
    j = floor(random()*(i+1))
    swap(openings[i],openings[j])

For rally r in 1..5:
    y, speed = openings[r-1]
    vx = first_vx if r is odd else -first_vx
    vy = speed*directions[r-1]
```

Seed selection is separate from this transition contract. Experiments store their actual seed lists so all candidates receive identical serves. Fresh seeds on arena opening and repeatable Restart are presentation choices; these candidate scripts do not change either.

## Implementations and fixtures

`training/scripts/offense_rules.py` provides the vectorized training engine. It inherits v3 reset, serve generation, observation conversion and array layout; the transition implementation is isolated from the released engine. `training/scripts/check_offense_rules.py` contains an independent scalar transition and verifies all legal contact heights, pre-contact paddle centers and actions, plus random trajectories, boundary cases, inactive lanes and vertical reflection symmetry.

Run:

```sh
.venv/bin/python training/scripts/check_offense_rules.py
```

`verification/checks.json` records counts and source hashes. `verification/fixtures.json` is compact JSON containing candidate parameters, explicit contact cases, and complete deterministic rally traces for seeds 11, 2026 and 4,294,967,295. Each trace contains the initial frame, the two action indices at every tick, and the expected resulting frame. Replay the recorded actions directly; no controller implementation is required. Cases with pre-populated loop histories include `seen_before` tuples. Traces enable loop detection and finish at a point, a proven loop or the 1,600-tick deadline.

These checks establish implementation agreement, not game balance. They do not show that a model can learn strong attack, that no alternative safe defense exists, or that humans find a candidate enjoyable.

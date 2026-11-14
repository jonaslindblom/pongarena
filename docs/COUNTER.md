# Counter: prediction, aiming and recovery under 100 parameters

Counter is a separate handwritten model with **97 parameters** and **81 MACs
per decision**. Its layout is **6 → 6 → 4 → 3 → 3**, with ReLU hidden layers
and three linear action scores. Predict and Attack remain unchanged.

The [constructor](../handwritten/counter.py) combines Attack's bounded
forecast and corner selection with Quill's direction gate and center recovery.
Manual designs at 75 parameters or fewer sacrificed too much interception
quality. The extra layer allows these behaviors to coexist while staying below
100 stored parameters, including every zero and bias. This does not establish
that 75 parameters is an absolute limit.

All weights come from fixed formulas and manual play evaluation. There is no
training, optimizer, automated fitting, learned checkpoint or runtime helper.

## What the layers do

| Layer | Role |
| --- | --- |
| 6 → 6 | Two forecast hinges, ball and paddle carriers, an outgoing-velocity gate and an opponent-position ramp. |
| 6 → 4 | Combine and gate the forecast; carry the paddle, incoming indicator and corner-selection hinge. |
| 4 → 3 | Remove the carry offset, fold the prediction at the walls and form the paddle target. |
| 3 → 3 | Choose movement or stay using a 1.1-unit deadband. |

The formulas below use [own-paddle-relative game coordinates](RULES.md#coordinates-and-player-identity):
X runs along the court, Y across it, and raw velocity is in court units per tick.
While the ball approaches, the unfolded forecast in raw court units is:

```text
forecastY = ball_y + sign(ball_vy) · min(0.4 · distance, 18 · |ball_vy|) + 0.75 · ball_vy
```

Here `distance = ball_x − 15` is distance to our contact plane. Counter extends Attack's
12-tick lead cap to 18 ticks; the 0.75-tick term accounts approximately for
discrete contact. The network uses the centered, normalized forecast
`forecastYNorm = (forecastY − 80) / 80` and folds it at the walls, at `±0.95`,
to obtain `reflectedYNorm`. Its incoming target retains Attack's aiming law:

```text
s = clip(80 · opponent_input + 4 · ball_y_input, −1, 1)
targetYNorm = (83 / 78) · reflectedYNorm + (19 / 312) · s
```

Both `_input` values are centered, normalized across-court positions;
the raw target is `80 + 80 · targetYNorm`.
The ramp chooses the corner away from the opponent, with ball position favoring
a cross-court return when the opponent is nearly centered. The aiming constants
come from the game's contact-offset law and the 26-tick flight at along-court speed 13.

While the ball departs, `ReLU(vx_input)` suppresses the forecast.
Here `vx_input` is the own-paddle-relative along-court velocity divided by 6:
negative on approach and positive on departure. A separate
incoming indicator removes the positive carry offset only on approach, so a
suppressed forecast becomes center rather than a false wall reflection.
Coupled aiming weights shrink the remaining opponent-dependent recovery offset
as along-court speed rises. Its magnitude is at most
`(190 / 39) · (1 − speed / 13)` court units, where `speed = |ball_vx|` in raw
units per tick: below three units at speed 5,
and zero at speed 13.

## Evaluation

See the [published league](https://jonaslindblom.github.io/pongarena/#league)
for current standings across all models. The checked-in
[development standings](../public/league.md) use fixed seeds for repeatable CI checks.

Counter uses along-court velocity for recovery, but still approximates flight
time instead of dividing distance by actual incoming speed. Difficult fast
returns and multiple real wall bounces remain weaknesses. Bot results do not
establish strength against human players.

## Reproduce

Use the [shared NumPy/ONNX setup and byte-for-byte rebuild check](../handwritten/README.md)
for the standalone [Counter constructor](../handwritten/counter.py).

Expected model SHA-256:

```text
5f7a480b98c982cf403f2f8849fac1e5e02503ff12e6b0b5d1e377856534e35c
```

Replay the current development standings from the repository root:

```sh
npm ci
npm run league:check
```

See the [league rules](RULES.md#league-schedule-and-scoring) for the scoring
and the separate published evaluation.

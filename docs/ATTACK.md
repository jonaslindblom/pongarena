# Attack: corner aiming in 75 parameters

Attack is an aggressive handwritten fork of [Predict](PREDICT.md).
Both retain the **6 → 6 → 3 → 3** ReLU layout, **75 parameters** and **63 MACs
per decision**. Predict's constructor and ONNX bytes are unchanged.

Attack trades two of Predict's forecast hinges for opponent-dependent
shot placement. The [constructor](../handwritten/attack.py) writes fixed
formulas. Designs were adjusted manually and compared through play; no
optimizer, training data, learned checkpoint or automated weight search was
used. Every decision runs through the ordinary dense ONNX layers.

## How it plays

The formulas below use [own-paddle-relative game coordinates](RULES.md#coordinates-and-player-identity):
X runs along the court, Y across it, and raw velocity is in court units per tick.
The first two neurons compute `ReLU(d + v)` and `ReLU(d − v)`, where
`d = (ball_x − 15) / 180` is the scaled distance to our paddle's contact plane and
`v = ball_vy / 6`. Their difference minus `v` equals
`sign(v) · min(d, |v|)` for legal distances. This produces the unfolded
forecast, expressed here in raw court units with `distance = ball_x − 15`:

```text
forecastY = ball_y + sign(ball_vy) · min(0.4 · distance, 12 · |ball_vy|) + 0.75 · ball_vy
```

Two more neurons carry the ball's across-court position and our paddle position. The final pair
forms `s = clip(80 · opponent_input + 4 · ball_y_input, −1, 1)`. It selects
the corner away from the opponent. When the opponent is almost centered,
the ball-position term favors a cross-court return instead of a flat shot.
Both `_input` values are centered, normalized across-court positions.

The network uses the centered, normalized forecast
`forecastYNorm = (forecastY − 80) / 80`. The second layer folds it at the walls,
at `±0.95`, to obtain `reflectedYNorm`, then compares our normalized paddle
position with this offset target:

```text
targetYNorm = (83 / 78) · reflectedYNorm + (19 / 312) · s
```

In raw court units, the target is `80 + 80 · targetYNorm`.
The aiming coefficients come from the game's return law: at along-court speed
13, a return reaches the other paddle after 26 ticks, and each unit of
contact offset shifts its unfolded landing by `26 · 9 / 15 = 78 / 5` units.
Moving slightly off the predicted landing therefore sends the ball toward
the selected corner. The output keeps Predict's 1.1-unit movement deadband.

## Evaluation

See the [published league](https://jonaslindblom.github.io/pongarena/#league)
for current standings across all models. The checked-in
[development standings](../public/league.md) use fixed seeds for repeatable CI checks.

Attack's bounded forecast is an approximation: it folds only one predicted
wall bounce and lacks along-court speed awareness and center recovery while
the ball travels away. Bot results do not establish strength against human players.

## Reproduce

Use the [shared NumPy/ONNX setup and byte-for-byte rebuild check](../handwritten/README.md)
for the standalone [Attack constructor](../handwritten/attack.py).

Expected Attack SHA-256:

```text
646d1a1365e3e307bd4431fc73930c5efa19de16b7cc22bbb9200608c2b720f8
```

Replay the current development standings from the repository root:

```sh
npm ci
npm run league:check
```

See the [league rules](RULES.md#league-schedule-and-scoring) for the scoring
and the separate published evaluation.

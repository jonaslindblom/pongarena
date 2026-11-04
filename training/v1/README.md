# Pong v1 — standalone training contract

This directory is a complete, frozen description of the arena's current game. Copy it on its own. `game.ts` is the authoritative reference, identical to the arena engine at packaging. `reference.py` is an independent standard-library port. No browser, React, server, or training framework is required. Future rule changes require a new version; do not silently update v1.

## Run and verify

```sh
# Within this directory; Node 22+ and npm
npm install
npm run example -- 24
# Python 3.10+, no third-party dependencies
python3 check.py
```

The example plays two simple tracking policies through a complete match. Replace `policy()` with your model. `fixtures.json` contains four full matches (seeds 0, 24, 902, 4294967295), explicit actions at each tick, every expected state, and both float32 observations. `actions: null` means transition to the next serve without an action. Eleven additional boundary cases isolate collision order, clipping, rounding and timeout behavior. `fixtures.ts` regenerates the data from the reference; verify your port against the committed file before regenerating it. The fractional negative-rounding case is synthetic, intentionally testing a language-port trap.

## State and coordinates

The court is 360 × 160 units. X increases right, Y increases down. All state arithmetic uses JavaScript numbers (IEEE-754 float64); normal reachable states remain integers. Ball wall boundaries are Y=4 and Y=156. Paddle contact planes are X=15 and X=345. Paddle centers remain in [15,145]. A contact succeeds when `abs(ballY - paddleY) <= 15`, including ball extent. The drawn paddle is 24 units tall; rendered geometry does not define physics.

`tick` counts frame transitions including between-rally serves. `rally` runs 1…5. `rallyTick` counts action ticks within this rally. `returns` counts successful contacts. `points` is [left,right]. An event is serve, return, point, draw, or null. `actor` identifies the returning or scoring player (0 left, 1 right), and is absent otherwise. `scores` records the two raw decisions, or two nulls at a serve. Counters/events support replay; they are not extra model inputs.

## Decision and transition order

At each action tick, obtain **both** decisions from the same state before either player moves. Action scores are three finite floats, ordered up, stay, down. Select the first maximum (index 0 wins ties). Scores are not probabilities; magnitude does not change speed.

1. Move both paddle centers by `2 * (actionIndex - 1)` and clamp each to [15,145].
2. Increase tick and rallyTick by one. Add VX to ball X and VY to ball Y. Clear event and actor.
3. If Y<4, set Y=8−Y and negate VY. Otherwise if Y>156, set Y=312−Y and negate VY. Equality does not reflect. One reflection suffices for reachable speeds.
4. If VX<0 and X<=15, test left contact; if VX>0 and X>=345, test right contact. Use the **post-movement, post-wall-reflection** ball Y and moved paddle center, without interpolation to the crossing plane.
5. Let offset = ball Y − that paddle center. If |offset|>15, award one point to the other player and emit point. Keep the ball's overshot position and velocities.
6. Otherwise set ball X exactly to the contact plane. Set VX to `outgoingSign * (5 + round(10 * abs(offset) / 15))`, where outgoingSign is +1 from left, −1 from right. Set VY to `clamp(round(offset / 3), -5, 5)`. Increment returns; emit return with that paddle as actor.
7. If rallyTick>=800 and this tick did not score, emit draw and clear actor, including if it just returned. A last-tick point takes precedence.

**Rounding means `floor(x + 0.5)`, including for negative x.** Python's `round()` uses different tie behavior. There is no acceleration, paddle spin, speed ramp, randomness during a rally, action repeat, or player-dependent handicap. A returned ball's horizontal speed is 5…15 and vertical speed −5…5.

After point/draw, do not call advance again. If rally<5, call serve with the same seed and previous frame. This adds one global tick, resets paddles to80 and ball X to180, resets rallyTick/returns to0, preserves points, and increments rally. After the fifth rally, the larger point total wins; equal totals draw. Each point is worth one; a timed-out rally awards no points. At normal playback each action tick is nominally 1/30 second. Rendering, inference time, inter-rally pauses and playback speed have no effect on rules.

## Deterministic serves

Accept a uint32 seed, reducing to the low 32 bits. The app draws a fresh seed on opening; Restart reuses it. Construct the entire five-serve plan afresh from that seed every time serve is called, then choose index rally−1.

PRNG state n starts at seed. Each random call updates `n = (1664525*n + 1013904223) mod 2^32`, returning `n / 2^32`. `sign()` consumes one call and returns −1 when random<0.5, else +1. Consume calls in this exact order:

1. `firstVX = 5 * sign()`.
2. `a = sign(); b = sign(); directions = [a,b,-a,-b,sign()]`.
3. For centers 32,56,80,104,128 in order, draw `y=center+floor(random()*21)-10`, then `speedY=1+floor(random()*5)`.
4. Fisher–Yates shuffle the five `(y,speedY)` pairs: for i=4,3,2,1 draw j=floor(random()*(i+1)) and swap i,j. Do not shuffle directions.
5. Rally r uses the shuffled pair at r−1; VX=firstVX for odd r, −firstVX for even r; VY=speedY*directions[r−1].

Initial frame is tick0, rally1, rallyTick0, points[0,0], both paddles80, returns0, event serve, no actor, scores[null,null].

## Model interface

Input `observation`: float32 `[1,6]`. For side0 (left) let m=+1; for side1 let m=−1. Calculate these in float64 then cast each result to float32:

| Index | Meaning | Formula |
| --- | --- | --- |
| 0 | Ball X, own side always left | m*(x−180)/180 |
| 1 | Ball Y, down positive | (y−80)/80 |
| 2 | Horizontal velocity, negative approaching | m*vx/6 |
| 3 | Vertical velocity, down positive | vy/6 |
| 4 | Own paddle center | (paddles[side]−80)/80 |
| 5 | Opponent paddle center | (paddles[1−side]−80)/80 |

Divide by6 is only a fixed feature scale. Inputs are not clipped; a VX of15 produces magnitude2.5. Mirror X/VX only, never Y or actions. There is no memory, tick counter, return counter, or previous-action input. Rally timeout is hidden to the policy; training may track it internally.

Output `action_scores`: finite float32 `[1,3]`, [up,stay,down]. During evaluation use deterministic first-argmax, never sample a probability distribution. Training can sample actions.

## Submission file

Export one self-contained stateless ONNX inference graph with one standard-domain opset import, version18. One input and one output with the exact names/types/fixed shapes above. Embedded float32 initializers only; no external data, sparse initializers, custom functions, training graphs or runtime randomness.

Allowed operators: Gemm, MatMul, Add, Relu, Tanh. Only Gemm accepts attributes (transA, transB, alpha, beta). Nodes must be topologically ordered with one unique output each. Initializer shapes must have positive integer dimensions, at most8 dimensions, at most65,536 elements each, and complete raw or float data. Limits: 256KiB total file, 256 nodes, 65,536 multiply–accumulates per decision, 262,144 summed intermediate output elements. Reused matrices count on every execution. Hidden widths, depths and supported activations may vary. Export only the actor; training critics, optimizers and opponent pools stay outside the file.

## Training freedom and fair evaluation

Any training method is allowed. Training rewards, curricula and opponent pools are choices outside the game rules. Arena scoring remains points. Use disjoint training/validation/test seeds and both paddle sides. Report wins/losses/draws plus rally lengths; surviving by timing out is not a win. Do not present success against one exploitable opponent as universal strength.

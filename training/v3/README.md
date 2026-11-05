# Pong v3 — standalone training contract

This directory is a complete, frozen description of the arena's current game. Copy it on its own. `game.ts` is the authoritative reference, identical to the arena engine at packaging. `reference.py` is an independent standard-library port. No browser, React, server, or training framework is required. Future rule changes require a new version; do not silently update v3.

## Run and verify

```sh
# Within this directory; Node 22+ and npm
npm install
npm run example -- 24
# Disable loop detection for human/stochastic-controller semantics
npm run example -- 24 --no-loops
# Python 3.10+, no third-party dependencies
python3 check.py
```

The example plays two fixed deterministic tracking policies through a complete match, with loop detection enabled. Replace `policy()` with your model. `fixtures.json` contains six full matches (seeds 0, 24, 902, 4294967295 plus deterministic tracking matches at 24 and 902), explicit actions, every expected state, and both float32 observations. `actions: null` means transition to the next serve without an action. Each match declares `loopDetection`; the original four action sequences depend on tick and therefore disable it.

Two additional centered-ball traces apply identical stay actions with detection enabled and disabled: the former draws on a repeat at tick165; the latter reaches the 1,600-tick deadline. Eighteen boundary cases isolate collision order, clipping, rounding, deadline precedence and loop history. Five detector histories verify the state key and rally reset. Stateful boundary cases use `seenContacts` to preload the detector. `check.py` also verifies explicit restart reset.

The current fixtures contain 2,446 full-match transitions and 1,765 trace transitions. `fixtures.ts` regenerates the data from the TypeScript reference; verify your port against the committed file before regenerating it. The fractional negative-rounding case and centered traces are synthetic cases, not new serve rules.

## State and coordinates

The court is 360 × 160 units. X increases right, Y increases down. All state arithmetic uses JavaScript numbers (IEEE-754 float64); normal reachable states remain integers. Ball wall boundaries are Y=4 and Y=156. Paddle contact planes are X=15 and X=345. Paddle centers remain in [15,145]. A contact succeeds when `abs(ballY - paddleY) <= 15`, including ball extent. The drawn paddle is 24 units tall; rendered geometry does not define physics.

`tick` counts frame transitions including between-rally serves. `rally` runs 1…5. `rallyTick` counts action ticks within this rally. `returns` counts successful contacts. `points` is [left,right]. An event is serve, return, point, draw, or null. `actor` identifies the returning or scoring player (0 left, 1 right), and is absent otherwise. `scores` records the two raw decisions, or two nulls at a serve. A drawn frame has `drawReason` equal to `loop` or `timeout`; other frames omit it. Counters/events support replay; they are not extra model inputs.

## Decision and transition order

At each action tick, obtain **both** decisions from the same state before either player moves. Action scores are three finite floats, ordered up, stay, down. Select the first maximum (index 0 wins ties). Scores are not probabilities; magnitude does not change speed.

1. Move both paddle centers by `2 * (actionIndex - 1)` and clamp each to [15,145].
2. Increase tick and rallyTick by one. Add VX to ball X and VY to ball Y. Clear event, actor and drawReason.
3. If Y<4, set Y=8−Y and negate VY. Otherwise if Y>156, set Y=312−Y and negate VY. Equality does not reflect. One reflection suffices for reachable speeds.
4. If VX<0 and X<=15, test left contact; if VX>0 and X>=345, test right contact. Use the **post-movement, post-wall-reflection** ball Y and moved paddle center, without interpolation to the crossing plane.
5. Let offset = ball Y − that paddle center. If |offset|>15, award one point to the other player and emit point. Keep the ball's overshot position and velocities.
6. Otherwise set ball X exactly to the contact plane. Set VX to `outgoingSign * (5 + round(8 * abs(offset) / 15))`, where outgoingSign is +1 from left, −1 from right. Set VY to `clamp(round(offset / 3), -5, 5)`. Increment returns; emit return with that paddle as actor.
7. When a `LoopDetector` is enabled, observe this frame. It first clears its history if the rally number changed. Only a successful return contributes a state key. If that exact post-return state was already observed in the same rally, emit draw with `drawReason="loop"` and clear actor.
8. If rallyTick>=1600 and this tick is neither point nor already draw, emit draw with `drawReason="timeout"` and clear actor, including if it just returned. A last-tick point takes precedence. A repeated contact exactly on the deadline keeps the loop reason.

**Rounding means `floor(x + 0.5)`, including for negative x.** Python's `round()` uses different tie behavior. There is no acceleration, paddle spin, speed ramp, randomness during a rally, action repeat, or player-dependent handicap. A returned ball's horizontal speed is 5…13 and vertical speed −5…5.

After point/draw, do not call advance again. If rally<5, call serve with the same seed and previous frame. This adds one global tick, resets paddles to80 and ball X to180, resets rallyTick/returns to0, preserves points, and increments rally. After the fifth rally, the larger point total wins; equal totals draw. Each point is worth one; a drawn rally awards no points, regardless of reason. At normal playback each action tick is nominally 1/30 second. Rendering, inference time, inter-rally pauses and playback speed have no effect on rules.

## Loop detection and controller mode

Use one detector per match, supplied to every action transition:

```ts
const loops = new LoopDetector()
let frame = serve(seed)
// Both policies are fixed deterministic functions of their six inputs.
frame = advance(frame, [leftScores, rightScores], loops)
```

The scalar Python API is equivalent: `loops = LoopDetector()` and `advance(frame, scores, loops)`. Omit the third argument for human matches and stochastic training. Both modes retain the same 1,600-tick maximum and physics.

The exact key is `(x, y, vx, vy, paddles[0], paddles[1])` **after a successful contact**. Use the full float64 values, not rounded observations or an approximate distance. `LoopDetector.observe(frame)` returns true if this key appeared at an earlier successful contact in the current rally; otherwise it stores the key and returns false. It ignores non-return frames. Stored scores, return counts, time and match score are excluded because they do not influence the fixed policy or the physics. Checking at contacts gives a simple deterministic termination rule; a repeated non-contact frame alone does not trigger an earlier draw.

The proof depends on both controllers remaining **stateless, deterministic and unchanged throughout the rally**, with decisions depending only on the exposed observations. A repeated complete physical state then reproduces the same two actions and all future transitions. The current ONNX submission contract guarantees stateless deterministic inference, and first-argmax makes action selection deterministic. The arena enables detection only when both paddles use such models.

Do **not** enable this reduced-key detector for a human, sampled policy actions, changing model weights, tick-dependent actions, or hidden/recurrent controller memory. Seeing the same positions does not prove those controllers will make the same next decision. A controller with memory would require that memory in the proof and a different specified key; it is not part of this contract. In particular, PPO's sampled training rollouts must use the full 1,600-tick deadline without loop exits. Validation and final evaluation use fixed argmax policies and can enable them. Loop exits shorten episodes, so using them in a different deterministic training scheme also changes reward accumulation/discounting; document that choice.

A detector clears automatically when an observed frame's rally number changes. On **Restart**, new match, changed controller, changed weights or changed controller mode, call `reset()` or create a new detector even if the rally number remains1. When replaying already-recorded frames, do not observe them again; retain the detector at the end of the generated history. If branching from an earlier frame, rebuild its contact history from that rally's recorded prefix.

The draw marker is replay metadata, not an extra model input. A loop draw awards neither player a point. It proves that these particular controllers cannot finish this rally; it does not claim equal overall model strength.

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

Divide by6 is only a fixed feature scale. Inputs are not clipped; a VX of13 produces magnitude13/6 (about2.17). Mirror X/VX only, never Y or actions. There is no memory, tick counter, return counter, or previous-action input. Rally timeout is hidden to the policy; training may track it internally.

Output `action_scores`: finite float32 `[1,3]`, [up,stay,down]. During evaluation use deterministic first-argmax, never sample a probability distribution. Training can sample actions, with loop detection disabled.

## Submission file

Export one self-contained stateless ONNX inference graph with one standard-domain opset import, version18. One input and one output with the exact names/types/fixed shapes above. Embedded float32 initializers only; no external data, sparse initializers, custom functions, training graphs or runtime randomness.

Allowed operators: Gemm, MatMul, Add, Relu, Tanh. Only Gemm accepts attributes (transA, transB, alpha, beta). Nodes must be topologically ordered with one unique output each. Initializer shapes must have positive integer dimensions, at most8 dimensions, at most65,536 elements each, and complete raw or float data. Limits: 256KiB total file, 256 nodes, 65,536 multiply–accumulates per decision, 262,144 summed intermediate output elements. Reused matrices count on every execution. Hidden widths, depths and supported activations may vary. Export only the actor; training critics, optimizers and opponent pools stay outside the file.

## Training freedom and fair evaluation

Any training method is allowed. Training rewards, curricula and opponent pools are choices outside the game rules. Arena scoring remains points. Use disjoint training/validation/test seeds and both paddle sides. Report wins/losses/draws, loop-vs-timeout counts and rally lengths; surviving by drawing is not a win. For competition, aggregate both side assignments on the same five serves and repeat over many seeds; one five-rally match is a viewing unit, not a reliable strength estimate. Do not present success against one exploitable opponent as universal strength.

## Change from v2

The maximum rally length changes from800 to1,600 ticks (53⅓ simulated seconds at30ticks/s), and fixed deterministic model matches end early when a post-contact state repeats exactly. Draws carry a reason. Points take precedence on the last tick; match scoring remains five rallies with one point per miss and none for a draw.

**Physics is unchanged:** court, paddle dimensions and speed, contact geometry and order, wall reflections, shot formula, return velocities, seeded serves, tick rate, rounding and all six input/three output meanings remain identical to v2. No speed ramp, hidden difficulty phase or handicap is introduced. V1/v2 model files remain interface-compatible, but their performance must be evaluated under v3 termination rules. Prior v1/v2 packages remain frozen.

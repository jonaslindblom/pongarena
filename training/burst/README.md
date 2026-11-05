# Pong Burst v1 — experimental standalone contract

Rules ID: **`pong-burst-v1`**. Candidate: **`burst96-cost8-half`**. This is a reproducible preview for human playtesting and model-size experiments, not a claim of solved game balance. Classic Pong v4 remains a separate, frozen competition. Scores from the two games must not share a ranking.

Copy this directory on its own. `game.ts` and the independent standard-library `reference.py` specify the complete game. No React, browser, ML framework or parent repository is required.

```sh
npm install
npm run example -- 24
npm run check
python3 check.py
```

The example uses fixed tracking policies. Replace its `policy()` with model inference. Disable exact loop termination with `--no-loops` for humans, changing weights, sampled actions or controllers with hidden memory. The fixture file contains 102 isolated transitions and three complete five-rally matches, 11,905 frames including next-serve transitions, using seeds 11, 2026 and 4,294,967,295. Both reference implementations reproduce every recorded action and state. `actions: null` advances to the next serve without a player decision.

## Court and state

The court is 360 by 160 units. X increases right and Y down. Ball centers reflect at Y=4 and Y=156; paddle contact planes are X=15 and X=345. Paddle centers remain in [15,145]. A hit succeeds when `abs(ballY - paddleY) <= 15`, including ball extent. The drawn paddle height does not define collisions.

State contains ball `(x,y,vx,vy)`, both paddle centers and both integer energy values. Energy is between 0 and 96 inclusive and begins at 96 for each player on every serve. All future physical behavior is determined by those values and the two actions. Ball X and horizontal velocity use exact half-units; Y, vertical velocity, paddles and energy are integers in normally reachable states. Arithmetic is float64; observation tensors are float32.

Replay fields `tick`, `rally`, `rallyTick`, `returns`, `points`, `event`, `actor`, `drawReason` and `scores` have the same roles as v4. Counters do not secretly govern movement, burst availability or speed progression.

## Input and output

Input tensor: `observation`, float32 `[1,8]`. Let `m=1` for the left observer and `m=-1` for the right observer. The observer always sees its own paddle as the left paddle.

| Index | Value |
| --- | --- |
| 0 | `m * (ballX - 180) / 180` |
| 1 | `(ballY - 80) / 80` |
| 2 | `m * vx / 6` |
| 3 | `vy / 6` |
| 4 | `(ownPaddleY - 80) / 80` |
| 5 | `(opponentPaddleY - 80) / 80` |
| 6 | `ownEnergy / 96` |
| 7 | `opponentEnergy / 96` |

Do not clip velocity observations to [-1,1]. Output tensor: `action_scores`, finite float32 `[1,5]`, ordered **up, stay, down, burst up, burst down**. The first maximum wins ties. Scores are not probabilities and their magnitudes do not change movement.

Both players decide from the same pre-tick state. Models remain stateless, but energy gives a visible consequence of earlier movement choices. Any architecture may be used within the arena's declared class/compute limits. Prototype models must declare compatible input/output shapes and should carry `rules=pong-burst-v1` metadata. Earlier six-input/three-output v4 weights cannot silently enter this competition.

## One tick, in order

1. Resolve both action indices. Normal up/down moves 2 units; stay moves zero. A burst action moves 4 units in its chosen direction **only if energy is at least 8 at the start of the tick**. Subtract 8 energy for a successful burst, even if the paddle clamps at the court boundary. Clamp paddle centers to [15,145].
2. Every normal action, including normal movement and stay, recharges 1 energy up to 96. A burst request with fewer than 8 energy moves at normal speed 2, spends no energy and **does not recharge**. These rules apply simultaneously to both players.
3. Increment tick and rallyTick. Add the incoming VX and VY to the ball position. Clear event, actor and drawReason.
4. If ball Y<4, set Y=8−Y and negate VY. Otherwise, if Y>156, set Y=312−Y and negate VY. Equality does not reflect; one reflection is enough at legal speeds.
5. If VX<0 and X<=15, test the left paddle. If VX>0 and X>=345, test the right. Use the already-updated paddle center and already-reflected ball Y. If the absolute offset exceeds 15, award one point to the other side and end the rally.
6. On a hit, clamp X to that paddle's contact plane. For offset `d=ballY-paddleY`, set `s=min(13,abs(incomingVX)+0.5)`, outgoing VX to `+s` from the left or `-s` from the right, and outgoing VY to `floor(9*s*d/195 + 0.5)`. Increment returns. This rounding definition applies to negative values too.
7. For two fixed deterministic stateless policies only, draw if the exact post-contact tuple `(x,y,vx,vy,leftPaddle,rightPaddle,leftEnergy,rightEnergy)` has previously occurred in this rally. Energy cannot be omitted. Reset the detector on every rally, restart, policy change or rules change.
8. Draw after 1,600 action ticks if no point or loop draw already ended the rally. Precedence is point, then loop, then timeout. Draws award no points.

There are five seeded rallies per match. Higher point total wins; equal totals draw. Serve generation is unchanged from v4 and is completely defined by `serve()` in both references: uint32 LCG with multiplier 1,664,525, increment 1,013,904,223, five shuffled height bands, alternating receiving sides and paired rising/falling serve directions. `Restart` must reuse the same uint32 seed.

The serve has horizontal speed 5. Its first return is 5.5; the cap is reached after 16 returns. Full court traversal at boost speed takes 33 ticks if energy were unlimited, but a full meter pays for only 12 consecutive boosted ticks. The difference between that short burst and sustained speed is intentional.

## Training and evaluation

The standalone rules do not prescribe how weights are obtained. Hand editing, supervised learning, reinforcement learning and evolutionary search enter the same rules/architecture class. Sampled training actions must disable loop exits. Fixed deterministic validation can enable them.

The repository's optional vector environment and diagnostic teacher are in `training/scripts/burst_rules.py`. These are training aids, not part of the exported model. The scalar reference here is the porting authority; energy update order, half-step VX and the exact loop key must agree before comparing results.

Use paired serve seeds with both side assignments when ranking weights. Keep training/development seeds separate from final evaluation. A long rally or a win against a weak tracker is not evidence that a model learned resource tactics. Current results and limitations live in the repository's burst-study report.

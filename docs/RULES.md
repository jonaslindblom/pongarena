# Game and model rules

PongArena accepts plain dense neural networks with handwritten, trained, or prompted weights. Eligibility depends on the validated graph and resource limits. The rules identifier is `pong-v3`, the model contract is `dense-mlp`, and league scoring uses `paired-round-robin`.

## Game

Each match contains five seeded rallies. Both players act from the same state at 30 decisions per second. A paddle moves two units per decision and its center stays between 15 and 145. Missing the ball awards the opponent one point; a drawn rally awards neither player a point.

The simulation court is 360 units along the court × 160 units across it. The ball starts with an along-court speed of 5 units per tick, gains one per return up to 13, and reflects at the side walls. Its contact offset sets its outgoing across-court velocity, rounded to an integer with magnitude at most 9. Exact update and serve-generation rules are in [game.ts](../src/game.ts).

Every serve begins at the center, (180, 80), with both paddles centered at 80. Its angle is 10–15° from the along-court axis: across-court speed is 0.89–1.33 units per tick. Five shuffled angle bands vary the openings; receivers alternate, and each receiver faces both across-court directions in the first four rallies. No opening hits a wall before the paddle, and a paddle moving toward the landing position after a 600 ms reaction delay can catch every serve. Return physics and speed progression are unchanged.

Interactive new matches, including **Play again**, draw a fresh seed. **Restart** and **R** replay the same seed. League evaluations keep paired, replayable seed sets. `pong-v3` changes the opening distribution from the original `pong` rules, so its benchmark identity and standings are new. Model observations, actions, eligibility and shipped weights are unchanged; retraining is not required.

A rally draws after 1,600 ticks. For deterministic automated players, an exactly repeated post-return physical state also ends the rally: the same decisions would repeat indefinitely. Human matches use the full time limit.

## Coordinates and player identity

Game coordinates stay fixed regardless of how the court is displayed:

| Quantity | Meaning |
| --- | --- |
| `X`, `VX` | Position and velocity **along the court**, between the paddle ends. X runs from 0 to 360. |
| `Y`, `VY` | Position and velocity **across the court**, left to right on screen. Y runs from 0 to 160; ball walls are at 4 and 156. |
| Simulation side `0` | Paddle contact plane at X = 15; mint paddle. |
| Simulation side `1` | Paddle contact plane at X = 345; orange paddle. |

Raw positions use court units and raw velocities use court units per tick. These are different from the centered, scaled model inputs below. A raw across-court position becomes `(Y − 80) / 80`; convert it back with `Y = 80 + 80 · normalizedY`.

Each model observes the game relative to its own paddle. Only X and VX are mirrored for side 1: its own end is always negative in normalized along-court position, and **negative observed VX means approaching its paddle**. Y and VY are never mirrored; positive VY always moves right on screen. For formulas using unnormalized, own-paddle-relative X, use world X for side 0 and `360 − worldX` for side 1; either model's own contact plane is then at X = 15.

Displayed **P1 is the top paddle** and **P2 is the bottom paddle**. These labels are not simulation indices or fixed colors. Side 0 is normally at the bottom; when the sole human occupies side 1, side 1 is shown at the bottom instead. In court units, `screenX = simulationY` in both orientations. With side 0 at the bottom, `screenY = 360 − simulationX`; with side 1 at the bottom, `screenY = simulationX`. Screen Y increases downward. Rendering scales these coordinates to the canvas.

## Observations and actions

The input is float32 `observation` with shape `[1, 6]`:

| Index | Value | Meaning |
| --- | --- | --- |
| 0 | `m · (ballX − 180) / 180` | Ball along the court, relative to own paddle |
| 1 | `(ballY − 80) / 80` | Ball across the court |
| 2 | `m · ballVX / 6` | Ball velocity toward (negative) or away from (positive) own paddle |
| 3 | `ballVY / 6` | Ball velocity left (negative) or right (positive) |
| 4 | `(ownPaddleY − 80) / 80` | Own paddle across the court |
| 5 | `(opponentPaddleY − 80) / 80` | Opponent paddle across the court |

Here `m = 1` for simulation side 0 and `m = −1` for side 1. Ball side walls are at normalized Y = ±0.95; legal paddle centers are within ±0.8125. Velocity inputs can exceed magnitude one.

The output is float32 `action_scores` with shape `[1, 3]`. The first maximum selects action index 0, 1 or 2. The legacy code names are `Up`, `Stay`, `Down`, requesting a Y change of −2, 0 or +2 before clamping to the paddle limits: **left, stay, right** on screen for either paddle. Scores must be finite. The tensor order and action names remain unchanged for model compatibility.

## ONNX contract

- One file, at most 256 KiB, using opset 18.
- One sequential chain of one to five `Gemm` layers: zero to four hidden layers and a linear three-score output.
- The same ordinary `Relu` or `Tanh` after every hidden layer; no output activation.
- Each layer owns a unique embedded float32 matrix and bias, all finite and used exactly once. `alpha = beta = 1`, `transA = 0`; `transB` may select matrix storage orientation.
- No extra inputs, branches, skip connections, shared tensors, computed weights, extra features, or input/output processing. Standalone `MatMul` and `Add` nodes are not accepted.

Hidden neurons can encode useful features through their weights, as [Follow](FOLLOW.md) and [Quill](QUILL.md) demonstrate. The submitted graph still consists entirely of these dense layers and activations.

| Class | Maximum stored parameters | Maximum MACs per decision |
| --- | ---: | ---: |
| Small | 256 | 1,024 |
| Large | 65,536 | 65,536 |

Both limits apply. Count all stored weights and biases, including zeros. MACs are the sum of input width × output width for each dense layer. Small models may compete in Large. Each class league runs its own complete tournament among eligible entrants.

`npm run models:check` validates every submitted file and its roster entry, rejects duplicate model hashes, and checks WASM inference on 217 observations, including legal extremes. Ports and runtime outputs must have the exact shapes and float32 types above; scores must be finite. Every decision during league play is checked too. Invalid submissions fail the PR gate.

## League schedule and scoring

Each unordered model pair plays every seed twice, swapping sides. Both games replay the same five serves. Add each model's rally points across both games to decide the paired contest: more points wins, equal points draw. A contest awards one league point for a win, half for a draw and zero for a loss. Rank by total league points, with equal totals sharing rank.

Returns, rally duration, size and speed do not break ties. Pairwise results remain in the JSON because matchups can be non-transitive. An inference failure stops the league as failed; incomplete runs receive no ranks.

The checked-in development field is **Follow, Predict, Attack, Counter, Quill, Small, Large and OneShot**. It uses 16 fixed public seeds for 448 paired contests, or **896 games**. The generator uses single-threaded ONNX Runtime Web CPU/WASM and the same game and scoring code as the browser. [Serve seeds](../src/league.ts)

```sh
npm run league        # write public/league.json and public/league.md
npm run league:check  # replay and require exact matches
```

Model PRs include both regenerated development files; [normal CI](../.github/workflows/check.yml) checks them. The separate [submission gate](../.github/workflows/model-evaluation.yml) freezes the full proposed roster before drawing 512 fresh shared serve sets. It saves the plan and results, reuses the plan on retries, and draws new serves for changed model bytes. Missing or expired recorded plans fail instead of rerolling. Pages publishes the verified results matching the merged models and evaluator. See [contributing](../CONTRIBUTING.md).

For eight models, the published evaluation contains 28,672 games, with 1,024 games and 5,120 rallies per matchup. Each seed supplies paired games against every opponent and counts as one independent sample. Replaying a deterministic match does not add observations. Publication requires a recorded 512-set plan for the current roster and evaluator.

Standings and league reports show approximate 95% confidence intervals for each model's league-point rate (points divided by available points). The percentile bootstrap uses 5,000 reproducible resamples of whole shared serve sets, preserving both sides and every matchup together. Intervals describe uncertainty across serves, not variability between training runs or confidence in a rank. Fewer than two sets, or no observed variation in a model's score across sets, produces no interval instead of a misleading zero-width estimate. Small local leagues give coarse estimates; published results use all 512 sets.

## Reproducibility

Each model's identity is the SHA-256 of its exact ONNX bytes. Duplicate model identities cannot enter twice. The benchmark hash includes the rules, contract, scoring protocol, class and budgets, sorted model hashes and resource counts, seed set and schedule. Display names do not affect it.

Compare scores only with the same benchmark inputs. Public seeds support reproducible development; after using evaluation results to tune a model, use a new evaluation set for independent claims. Submitted display names do not change the saved seed-plan identity. Browser exports are local records and do not change published rankings.

# Game and model rules

PongArena accepts plain dense neural networks. Parameters may be handwritten or trained; eligibility depends on the validated graph and resource limits. The rules identifier is `pong-v2`, the model contract is `dense-mlp`, and league scoring uses `paired-round-robin`.

## Game

Each match contains five seeded rallies. Both players act from the same state at 30 decisions per second. A paddle moves two units per decision and its center stays between 15 and 145. Missing the ball awards the opponent one point; a drawn rally awards neither player a point.

The simulation court is 360 × 160, with paddle contact planes at x = 15 and x = 345. The ball starts with horizontal speed 5, gains one per return up to 13, and reflects at y = 4 and y = 156. Its contact offset sets its outgoing vertical velocity, rounded to an integer with magnitude at most 9. Exact update and serve-generation rules are in [game.ts](../src/game.ts).

Every serve begins at the center, (180, 80), with both paddles centered at 80. Its angle is 15–20° from the receiving axis in simulation coordinates: lateral speed is 1.34–1.81 units per tick. Five shuffled angle bands vary the openings; receivers alternate, and each receiver faces both lateral directions in the first four rallies. No opening hits a wall before the paddle, and a paddle moving toward the landing position after a 300 ms reaction delay can catch every serve. Return physics and speed progression are unchanged.

Interactive new matches, including **Play again**, draw a fresh seed. **Restart** and **R** replay the same seed. League evaluations keep paired, replayable seed sets. `pong-v2` changes the opening distribution from the original `pong` rules, so its benchmark identity and standings are new. Model observations, actions, eligibility and shipped weights are unchanged; retraining is not required.

A rally draws after 1,600 ticks. For deterministic automated players, an exactly repeated post-return physical state also ends the rally: the same decisions would repeat indefinitely. Human matches use the full time limit.

## Observations and actions

The input is float32 `observation` with shape `[1, 6]`:

| Index | Value |
| --- | --- |
| 0 | `(ballX − 180) / 180` |
| 1 | `(ballY − 80) / 80` |
| 2 | `ballVX / 6` |
| 3 | `ballVY / 6` |
| 4 | `(ownPaddleY − 80) / 80` |
| 5 | `(opponentPaddleY − 80) / 80` |

For the right player, indices 0 and 2 are negated so its own paddle is always on the left in observation space. Velocity inputs can exceed magnitude one.

The output is float32 `action_scores` with shape `[1, 3]`. The first maximum selects up, stay or down in simulation coordinates. The app rotates the court: these actions display as left, stay and right. Scores must be finite.

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

The checked-in development field is **Follow, Quill, Small and Large**, using 16 fixed public seeds in [league.ts](../src/league.ts): 96 paired contests, or **192 games**. The generator uses single-threaded ONNX Runtime Web CPU/WASM and the same game and scoring code as the browser.

```sh
npm run league        # write public/league.json and public/league.md
npm run league:check  # replay and require exact matches
```

Model PRs include both regenerated development files; [normal CI](../.github/workflows/check.yml) checks them. The separate [submission gate](../.github/workflows/model-evaluation.yml) freezes the full proposed roster before drawing 128 fresh shared serve sets. It saves the plan and results, reuses the plan on retries, and draws new serves for changed model bytes. Missing or expired recorded plans fail instead of rerolling. Pages publishes the verified results matching the merged models and evaluator. See [contributing](../CONTRIBUTING.md).

## Reproducibility

Each model's identity is the SHA-256 of its exact ONNX bytes. Duplicate model identities cannot enter twice. The benchmark hash includes the rules, contract, scoring protocol, class and budgets, sorted model hashes and resource counts, seed set and schedule. Display names do not affect it.

Compare scores only with the same benchmark inputs. Public seeds support reproducible development; after using evaluation results to tune a model, use a new evaluation set for independent claims. Submitted display names do not change the saved seed-plan identity. Browser exports are local records and do not change published rankings.

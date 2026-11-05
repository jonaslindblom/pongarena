# Local weight competition

The arena accepts weights without prescribing how they were created. Hand-edited weights, imitation, self-play and evolutionary search compete under the same rules. The game specification, observation/action contract and allowed inference graph determine eligibility; training provenance does not award points.

The current ranking feature is a **local round-robin benchmark**. It is useful for comparing uploaded weights against a fixed field and inspecting size/strength tradeoffs. It is not a verified public leaderboard. Browser results and downloaded JSON files are under the local user's control.

## Provisional model classes

Both limits must be satisfied. Smaller models may compete in any larger class. A class run includes only the selected eligible entrants and runs its own complete round-robin; filtering a mixed-size table does not create a class championship.

| Class | Embedded float32 scalars | Multiply–accumulates per decision |
| --- | ---: | ---: |
| Nano | 512 | 2,048 |
| Small | 2,048 | 8,192 |
| Standard | 20,480 | 65,536 |
| Open | 65,536 | 65,536 |

These are initial engineering budgets, not measured capability boundaries. The model-size study should inform future class revisions. A changed class budget changes the benchmark identity.

The existing file, operator and intermediate-tensor limits also apply. A small parameter count does not excuse excessive computation. Count every embedded float32 initializer scalar, including biases and any fixed handcrafted transforms; do not rely on the author's claimed number of trainable parameters. Count a matrix multiplication every time the graph executes it, including reused matrices. Parameters and MACs are derived from the validated graph, not filename or metadata.

For the experimental eight-input/five-action interface, these plain networks illustrate the budgets:

| Architecture | Parameters | Dense MACs | Smallest class |
| --- | ---: | ---: | --- |
| `8 → 16 → 5` | 229 | 208 | Nano |
| `8 → 16 → 16 → 5` | 501 | 464 | Nano |
| `8 → 32 → 32 → 5` | 1,509 | 1,440 | Small |
| `8 → 64 → 64 → 5` | 5,061 | 4,928 | Standard |
| `8 → 128 → 128 → 5` | 18,309 | 18,048 | Standard |

Parameter count measures capacity; it does not promise a tactical or entertaining opponent. Inference latency is reported separately when measured because it depends on browser, hardware, runtime and graph structure. It does not determine the class.

## Identity and compatibility

Each entrant's identity is the SHA-256 of its complete ONNX bytes. A renamed file remains the same entrant. Byte-identical files cannot enter twice, even under different display names. Different serializations of the same function can still have different hashes; a public system would additionally need submission and duplicate-policy controls.

Before entering a run, every model must pass the appropriate game input/output validator. Legacy six-input/three-action weights and experimental eight-input/five-action weights do not enter the same game by silently ignoring or inventing outputs. Any explicit compatibility adapter is part of a separate declared contract and its compute/parameter budget. A controller's human label or training method cannot replace graph validation.

The benchmark fingerprint includes:

- Ranking protocol version (`paired-round-robin-v1`).
- An immutable rules identifier and the observation/action contract identifier.
- Class identifier and its exact parameter/MAC budgets.
- The selected roster's sorted file hashes and validated resource counts.
- The sorted distinct unsigned 32-bit seed set.
- Five rallies per game, both side assignments, and the scoring/tie policy.

Display names and UI ordering are excluded. The caller must change the rules identifier whenever simulation, termination or policy evaluation semantics change. An identifier containing the frozen engine's content hash is preferable for a published competition.

## Schedule and scoring

Each unordered model pair faces every seed twice: first A on the left, then B on the left. Both games replay the same five seeded rallies. All pairs use the same seed set. The arena can run four, eight or sixteen seed sets per pairing; the runner supports up to 64 distinct seeds.

The two games form one **paired contest**, with ten rallies in total. Add each model's rally points across its two side assignments. More rally points wins the contest; equal points draw it. League points are one for a contest win, half for a draw and zero for a loss. Sort the complete table by total league points, with equal scores sharing rank. Do not silently break a tie using returns, rally length, parameter count, inference time or a name. Rally points for/against and drawn rallies are descriptive diagnostics.

This makes side assignment symmetric and avoids ranking a player by a favorable five-rally serve order. It does not eliminate sampling noise or non-transitive matchups. An A-beats-B-beats-C-beats-A field may tie; the pairwise records are retained in the JSON export. Scores across different rosters or seed sets are different benchmarks and are not interchangeable ratings.

The game rules determine when a rally ends, including timeouts and exact-loop detection. A drawn rally awards no rally point. A technical inference failure stops the local league as failed; it does not silently award the opponent a win. A future public competition may define disqualification separately.

## Cancellation, storage and reproduction

The runner evaluates one game at a time and yields between games. The injected headless game runner must also yield periodically during the game and check its `AbortSignal` before accepting inference results. The UI can cancel without skipping simulation states. Only completed side pairs enter provisional totals. Cancelled, failed or still-running leagues have no assigned ranks.

The browser retains the last complete league in local storage when available. Restoring checks its benchmark fingerprint, full pair/seed schedule and valid game scores, then reconstructs standings. This catches accidental incomplete or incompatible records; it does not authenticate scores against a malicious local editor. If storage is unavailable, the completed result can still be exported. Exported JSON contains the full specification and both game results for every completed contest.

**Repeat benchmark** uses the saved weights, seed set, class and rules contract. All those model files must still be loaded. **New serves** draws a fresh comparison seed set. New model development should use fresh reserved evaluation seeds rather than repeatedly optimizing against a displayed leaderboard batch. Seeds are not explicit model inputs, but a network can recognize opening states, so a fixed public seed list is not a sufficient anti-overfitting measure.

## What a public leaderboard needs later

Public standings require an independent evaluator to validate the uploaded bytes, enforce class limits, execute the frozen game/runtime protocol and publish results tied to model/rules hashes. A browser may preview a match, but its claimed score is not authoritative. Keep private or rotating evaluation seeds, separate development and final rounds, rerun a common opponent field, and publish enough match records for auditing after each round. Freeze each season's rules and class budgets; changes start a new season.

Ranking quality also depends on the reference field. Include defensive, attacking and resource-aware opponents and multiple learned policies. Keep tactical probes and human playability measurements separate from league points: a long rally, high teacher agreement or a correctly timed burst is useful evidence but is not itself a match win.

## Verify

```sh
node --import tsx --test tests/competition.test.ts
```

The tests enforce side-pair symmetry, size/compute boundaries, stable benchmark identity, shared ranks for non-transitive ties, duplicate-file rejection, cancellation/failure handling and complete-record restoration.

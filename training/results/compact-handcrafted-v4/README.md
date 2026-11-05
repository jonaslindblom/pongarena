# Compact handcrafted Classic v4 weights

> **Archived and ineligible:** this study used physics/planning graph operations that do not satisfy the current plain dense-network contract (`dense-mlp-v2`). Its models are removed from the active catalog and cannot be imported for active play. Scores below are historical results under the former general-graph contract, not current dense-network standings.

Both new no-training models beat all ten active presets in the recorded head-to-head comparisons. Against Pacer over three fresh 64-seed blocks, **Nano wins 1,397, loses 480, draws 43**; **Small wins 1,428, loses 422, draws 70**. Both pass the declared criteria in every block.

Small preserves the full original handcrafted policy's target tables with **94.6% fewer parameters**. Nano uses a simplified incoming placement and recovery policy. These are fixed physics networks built analytically, with no gradient updates, optimizer, distillation, or mutation search. Seven manually specified compact target designs, including failed controls, are retained in `development.json`.

## Actual model budgets

“Tiny” is represented by the arena's official Nano class. Both files use the current six-input/three-output Classic v4 contract; the archived Burst Tiny is a different model and was excluded.

| Model | Parameters | MACs / decision | Nodes | ONNX bytes | Smallest class |
| --- | ---: | ---: | ---: | ---: | --- |
| Nano | 418 | 262 | 44 | 3,672 | Nano: ≤512 parameters / 2,048 MACs |
| Small | 1,835 | 1,015 | 127 | 12,017 | Small: ≤2,048 parameters / 8,192 MACs |
| Original handcrafted | 33,793 | 30,742 | 155 | 150,982 | Open |

The actual arena validator derives these counts from every embedded scalar and executed matrix multiplication. Shared tensors count once for storage and every time for compute. Fewer MACs do not establish browser latency; latency was not measured.

## What changed

The earlier trajectory network enumerated many velocity/flight cases. These graphs instead use supported scalar MatMul products: calculate distance, interpolate reciprocal horizontal speed exactly at integer speeds 5–13, calculate flight time, multiply by vertical velocity, and reflect the landing height at the walls. ReLU ramps implement the flight-time ceiling. Binary thresholds k+1/64 and k+1/32 give exact integer plateaus; all 5,958 distance/speed reconstruction checks passed with zero tick error.

Target tables remain derived directly from `pace_study.py` geometry. Sparse piecewise-linear curves omit zero slope changes and broadcast their knot biases, removing dense matrices of zeros and ones. Stages 12 and 13 share identical tables. The Small model retains all stage-aware incoming contact targets and outgoing reply-coverage midpoints from the original Open model.

Nano keeps only the cap-speed incoming contact table and recovers toward court center 80 while the ball travels away. That saves parameters but weakens recovery against difficult opponents. Both use a 1.1-unit deadband. Integer/half-integer targets produce the same intended actions as the original 1.001-unit deadband, with more room for floating-point error.

## Frozen fresh-seed evaluation

`freeze.json` records both model hashes, sources, reference hashes, and 192 disjoint final seeds before matches. Its archive scan checks all retained integer/numeric-string leaves for overlap. The final models are evaluated through their actual static batch-one ONNX files in native CPU ONNX Runtime, under v4 rules with deterministic first argmax and exact loop detection. Each seed contributes five rallies on each side. Confidence intervals resample whole seeds. These are matchup tests, not a complete league or public leaderboard.

| Model / Pacer block | Rally W / L / D | Paired contest W / L / D | Point-margin 95% CI |
| --- | ---: | ---: | ---: |
| Nano 1 | 452 / 175 / 13 | 55 / 1 / 8 | +0.364 to +0.506 |
| Nano 2 | 466 / 159 / 15 | 59 / 3 / 2 | +0.408 to +0.552 |
| Nano 3 | 479 / 146 / 15 | 63 / 1 / 0 | +0.467 to +0.575 |
| Small 1 | 488 / 133 / 19 | 59 / 1 / 4 | +0.486 to +0.617 |
| Small 2 | 466 / 148 / 26 | 60 / 2 / 2 | +0.427 to +0.566 |
| Small 3 | 474 / 141 / 25 | 62 / 1 / 1 | +0.455 to +0.586 |

Every block has a majority of paired contests won, a positive margin confidence interval, and more rally wins than losses on each physical side. Remaining pairings use the first block, 640 rallies each:

| Opponent | Nano W / L / D | Small W / L / D |
| --- | ---: | ---: |
| apprentice-v4 | 583 / 53 / 4 | 594 / 24 / 22 |
| contender-v3 | 518 / 115 / 7 | 557 / 45 / 38 |
| strategist-v3 | 495 / 123 / 22 | 603 / 17 / 20 |
| selfplay-v3 | 560 / 69 / 11 | 574 / 25 / 41 |
| apprentice-v3 | 582 / 48 / 10 | 632 / 8 / 0 |
| selfplay-v2 | 510 / 128 / 2 | 634 / 5 / 1 |
| challenger | 632 / 6 / 2 | 631 / 9 / 0 |
| tracker | 631 / 9 / 0 | 634 / 6 / 0 |
| anticipator | 627 / 13 / 0 | 629 / 11 / 0 |
| original-open | 215 / 373 / 52 | 134 / 134 / 372 |
| attack | 72 / 568 / 0 | 78 / 562 / 0 |
| defend | 42 / 537 / 61 | 131 / 355 / 154 |

Nano versus Small is **215 / 373 / 52**, exactly the same result as Nano versus the original Open model on these seeds. Small versus Open is **134 / 134 / 372**, with all paired contests tied. This supports preserved behavior rather than claiming an improvement over the larger original. Both compact models remain weak against adaptive attack; Nano's center recovery also regresses sharply against adaptive defense. Neither is a universal upgrade or a proof of optimal play.

## Verification and files

`verification.json` records validator budgets and 8,192 numeric probes per export with zero action mismatches against the integer-table policy. `independent-verification.json` adds 21,962 independent probes per model, including action/deadband boundary cases; all pass. Small matches the actual Open model's actions on every checked probe and encountered match decision.

Twelve complete native matches produced 22,196 frames and 44,296 decisions. Both the frozen v4 and live TypeScript engines reproduce all transitions, with zero state/action mismatches. Signed-zero representation differences were checked separately and produce identical native scores/actions. Both model files regenerate byte-identically. Zero-input import probes pass. Browser WASM parity also passed all 12 matches, 22,196 frames and 29,844 compact-model decisions: zero score, action or state differences from the native replay. See `browser-verification.json`. Complete native match evidence is compressed in `independent-replays.json.gz`; exact audit/development source snapshots are in `verification-source/`.

The candidate sources, reference weights, original Open model, and prior investigation remain unchanged. No presets or arena rules were edited. Use **Import model** to load either `handcrafted-nano-v4.onnx` or `handcrafted-small-v4.onnx`.

Reproduce with the existing project Python and npm dependencies, using a fresh output directory:

```sh
.venv/bin/python training/scripts/compact_handcrafted_v4.py --out /tmp/compact-v4 --evaluate
```

Omit `--evaluate` to construct and validate the files without replaying the consumed final benchmark. Future weight selection requires fresh reserved seeds. The script refuses to overwrite result directories.

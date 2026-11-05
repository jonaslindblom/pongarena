# Handcrafted v4 weights, without training

> **Archived and ineligible:** this study used physics/planning graph operations that do not satisfy the current plain dense-network contract (`dense-mlp-v2`). Its models are removed from the active catalog and cannot be imported for active play. Scores below are historical results under the former general-graph contract, not current dense-network standings.

`handcrafted-v4.onnx` beats every current arena preset in the recorded fresh-seed comparisons. Against Pacer, it wins **1,412**, loses **455**, and draws **53** rallies across three separate 64-seed blocks. It passes the declared head-to-head criteria in every block. It is not universally stronger: it performs worse than Pacer against the adaptive attacking script.

This is a new **hand-constructed physics network**, rather than an improved learned Pacer checkpoint. No gradient updates, optimizer, distillation, or evolutionary weight search were used. Simulated matches were used for development and evaluation. Eight manually specified designs or diagnostic policies were examined; their successes and failures are retained in `development.json`.

## What the existing weights reveal

Pacer's original held-out v4 results already show a tactical gap: 163/439/38 against adaptive attack and 77/441/122 against adaptive defense, despite strong wins against simpler models. See `../pace-study/training/README.md`. Its development selection averages margins against both strong and weak opponents; this permits gains against weak opponents to offset tactical regressions.

Pacer also makes inconsistent decisions on vertically reflected integer game states. However, averaging its reflected scores did not establish a general improvement: on 240 development rallies it scored 121/113/6 against Pacer, while worsening against attack and Contender. Four direct stay-bias edits, ±0.5 and ±1.0, all lost to Pacer. These are failed proposals, not upgrades.

The older Strategist contains a concrete transfer limitation: its fixed trajectory features represent vertical velocities only from −5 to 5, while v4 permits −9 to 9. Extending that geometry while retaining its learned head improved it against original Strategist (156/73/11), but still lost badly to Pacer (34/190/16). Correct physics alone does not repair an older learned target policy.

## How the proposed weights work

The six observations enter a fixed ReLU trajectory network adapted from `trajectory_features.py`. It computes the next contact height and flight time, including integer velocity selection and wall reflection under v4.

The target head contains tables calculated directly from `pace_study.py`'s v4 geometry. For an incoming ball, it chooses the legal contact center with the lowest one-return recovery risk; ties favor a small contact offset. For an outgoing ball, it moves toward the midpoint of the universal reply-coverage interval. Velocity selectors choose the relevant speed stage. ReLU plateaus implement the integer-height tables without fitting data and tolerate small floating-point errors. Fixed paddle-position feedback converts the target into up/stay/down scores, with a 1.001-unit deadband.

The selected incoming center may be unreachable from the current paddle position. The policy does not optimize against the opponent's position or solve multi-return play. These limitations help explain its weakness against the attacking script.

The ONNX has **33,793 parameters**, **30,742 MACs per decision**, 155 nodes, and 150,982 bytes. It passes the current six-input/three-output arena validator and fits **Open**. Pacer has 17,795 parameters and 17,536 MACs and fits Standard; this is not an equal-budget Standard-class improvement.

## Frozen final results

The candidate and 192 unique test seeds were frozen in `freeze.json` before final gameplay. All matches use the actual batch-one ONNX files in native CPU ONNX Runtime, v4 rules, exact loop detection, common serves, and both side assignments. Each seed contributes ten rallies; confidence intervals resample whole seeds. These are matchup tests, not a complete round-robin league.

| Pacer block | Candidate W / L / D | Paired contest W / L / D | Point-margin 95% CI |
| --- | ---: | ---: | ---: |
| 1 | 457 / 160 / 23 | 56 / 3 / 5 | +0.388 to +0.542 |
| 2 | 472 / 151 / 17 | 61 / 2 / 1 | +0.434 to +0.564 |
| 3 | 483 / 144 / 13 | 59 / 0 / 5 | +0.455 to +0.597 |

Every block has a majority of paired contests won, a positive margin interval, and more rally wins than losses on each physical side. The remaining presets and scripts use the first final block, 640 rallies per pairing:

| Opponent | Candidate W / L / D |
| --- | ---: |
| Apprentice v4 | 575 / 41 / 24 |
| Contender v3 | 562 / 34 / 44 |
| Strategist v3 | 611 / 15 / 14 |
| Selfplay v3 | 551 / 41 / 48 |
| Apprentice v3 | 630 / 10 / 0 |
| Selfplay v2 | 629 / 6 / 5 |
| Challenger | 633 / 7 / 0 |
| Tracker | 634 / 6 / 0 |
| Anticipator | 622 / 18 / 0 |
| Adaptive attack | 97 / 543 / 0 |
| Adaptive defense | 123 / 379 / 138 |

It won all 64 paired contests against each of the nine additional presets. Pacer on those same final seeds scored 151/451/38 against attack and 64/442/134 against defense: the candidate improves against defense and regresses against attack. Archived Burst weights have a different contract and were excluded. Older presets' results here concern transfer to v4, not their original rules.

## Verification and use

`verification.json` records 8,192 component-valid numeric probes with zero action mismatches against the integer-table policy, all 2,754 target-table entries, 1,341 exact wall-fold probes, and arena validation. Maximum native score error was 1.17e−5. Synthetic component-valid probes are not all asserted to be reachable.

The independent full-match and archive-seed checks are recorded separately in `independent-verification.json`. The current v4 fixture checker and all 53 application tests also passed, with no skipped tests. Frozen models, rules, presets, and historical reports were left unchanged.

Use **Import model** in the arena to select `handcrafted-v4.onnx`. It remains a separate candidate and is not installed as a default preset. Reproduce into a fresh directory with the project Python environment:

```sh
.venv/bin/python training/scripts/handcrafted_v4.py --out /tmp/handcrafted-v4-reproduction
```

The script requires the existing Python training dependencies and npm dependencies for the arena validator, but performs no training. It refuses to overwrite a result directory. The recorded final seeds are now consumed and must not be used for further weight selection.

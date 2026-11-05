# Point-mechanism audit under unchanged Pong v3 rules

This diagnostic asks whether Contender scores through immediate unreachable shots or mainly concedes fewer points. No rules, weights, arena UI or reference opponents were changed. It is not a new model-selection or final-test result.

[Design findings and the defensive guarantee](FINDINGS.md) connect these measurements to the exhaustive geometry audit and possible future rule experiments.

All rows use the same 100 dispersed diagnostic seeds from [1,400,000,000,1,500,000,000), sampled with NumPy RNG 2201, all five serves and both physical sides: 1,000 rallies per matchup. Models run with PyTorch first argmax; the current vector engine enables exact model loops and the 1,600-tick limit.

## Scoring and conceding

| Candidate | Opponent | W / L / D | Candidate points: unreachable return / reachable return / serve | Opponent points: unreachable return / reachable return / serve | Median returns in scored rallies |
| --- | --- | --- | --- | --- | ---: |
| contender | selfplay-v2 | 401 / 17 / 582 | 113 / 244 / 44 | 1 / 0 / 16 | 16.0 |
| contender | attacker | 214 / 22 / 764 | 214 / 0 / 0 | 6 / 0 / 16 | 19.0 |
| contender | interceptor | 36 / 16 / 948 | 36 / 0 / 0 | 0 / 0 / 16 | 1.0 |
| prior-ppo | selfplay-v2 | 396 / 108 / 496 | 243 / 109 / 44 | 1 / 72 / 35 | 7.0 |
| prior-ppo | attacker | 196 / 72 / 732 | 196 / 0 / 0 | 6 / 31 / 35 | 7.5 |
| prior-ppo | interceptor | 260 / 54 / 686 | 260 / 0 / 0 | 0 / 19 / 35 | 7.0 |
| selfplay-v2 | selfplay-v2 | 306 / 306 / 388 | 208 / 55 / 43 | 208 / 55 / 43 | 6.0 |
| selfplay-v2 | attacker | 111 / 434 / 455 | 111 / 0 / 0 | 388 / 3 / 43 | 4.0 |
| selfplay-v2 | interceptor | 830 / 74 / 96 | 830 / 0 / 0 | 0 / 31 / 43 | 8.0 |

## What the comparison says

Against v2, Contender scores 401 points versus its prior PPO initializer’s 396, while concessions fall from 108 to 17. Of the 96-point improvement in net rally margin, five are additional scores and 91 are avoided concessions. This arithmetic decomposition is not a causal ablation, but it supports the observation that the gain is predominantly defensive.

Against the attacker, scores rise 196→214 while concessions fall 72→22. Fifty of the 68 net additional points come from fewer concessions. All 214 Contender scoring returns against the attacker were mechanically unreachable at release, so real shot pressure does exist against that controller. That does not establish a forceable win against an optimally positioned defender.

Against the interceptor, Contender’s scoring falls 260→36 relative to its initializer, with 948/1,000 rallies drawing. Original v2 scores 830 against that same defender. This is a substantial opponent-dependent tradeoff, not universal superiority.

## Meaning and limits of “unreachable”

At every return, freeze the receiver’s position after that tick’s simultaneous paddle movements. Compute exact next contact time and reflected landing height. The receiver can move 2 units on each subsequent tick, bounded to centers 15..145, and catches when within 15 of landing. A positive distance beyond that reachable contact interval means no possible movement after release could catch the shot. At the eventual contact, predicted height and flight duration must match the engine. A successful catch of an “unreachable” shot is an assertion failure.

A locally unreachable shot can result from the receiver’s avoidable positioning before release. It is not proof that an attack defeats perfect strategic defense. Conversely, a reachable miss is not automatically an unforced error: the policy may have attempted a particular return, and this local test does not attribute intent or analyze alternative earlier choices. Initial-serve misses are reported separately. The audit does not award quality points or change scoring.

Shot statistics include emitted returns, scorable returns that can arrive before the deadline, and actually completed return flights separately. A shot emitted near timeout might be unreachable but never award a point before the rally ends. Reported point classifications above count actual points only.

## Stress test against universal coverage

The separately constructed [`attack_geometry.safe_defender`](../../scripts/attack_geometry.py) uses the same six observations, intercepts incoming shots, and recovers into a band that covers every legal opponent return. On the same 100 diagnostic seeds:

| Opponent of safe defender | Wins / losses / draws |
| --- | --- |
| Contender | 0 / 129 / 871 |
| Prior PPO | 0 / 610 / 390 |
| Selfplay v2 | 0 / 772 / 228 |
| Scripted attacker | 0 / 0 / 1,000 |

None produces a locally unreachable return against the safe defender. All 129 Contender concessions remain physically reachable at release: 113 after returns and 16 on serves. Contender's almost error-free interception against its selected opponents therefore does not generalize to every shot distribution. Conversely, its locally forced points against the original attacker demonstrate exploitation of that controller's earlier recovery choices, not an attack that defeats correct coverage. These finite matches corroborate the separate exhaustive geometry argument in [geometry.json](geometry.json); they are not themselves a proof against every possible policy. The defender is a diagnostic script, not a submitted fully learned model.

## Verification and reproduction

The reachability checker passed 25,000 randomly sampled physical launch states against direct ideal-chase simulation in the unchanged engine: exact landing, contact tick and catch/miss agree. It includes 5,232 unreachable and 19,768 reachable flights. Every recorded match contact also checks predicted landing and flight duration, and any locally unreachable catch fails loudly.

```sh
.venv/bin/python training/scripts/analyze_points.py
.venv/bin/python training/scripts/analyze_points.py --safe-defender
```

`points.json` retains exact seed lists, model/source hashes, per-seed outcomes, every point’s release-reachability classification, shot-offset/speed distributions, draw reasons, and paired evaluation fields. No independent-rally confidence claim is made; the existing evaluator’s margin interval resamples whole seeds.

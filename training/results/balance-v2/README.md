# Choosing Pong v2's speed cap

The only rule varied was the horizontal return speed: `5 + round((cap - 5) * abs(offset) / 15)`. Paddles remain2 units/tick, vertical return velocity remains `round(offset/3)` clamped to ±5, and every candidate gets the identical seed, serve height, direction and speed. The original six-float observation and three-score output contract is unchanged.

## Selection criterion

Prefer longer **scored** rallies between strong players, a smaller first-receiver advantage, and continued scoring against a defensive player. A timeout is not a successful long rally. Use scripted players first to avoid conflating rule balance with retraining. Frozen v1 network results are a transfer diagnostic only. Do not choose the cap on the final retrained-policy test seeds.

## Results

Initial scan: seeds3,000,000,000–3,000,000,059, five rallies each, both sides, 600 rallies/pair/cap. Tested caps15,14,13,12,11,10. Cap13 was the strongest compromise. Caps10 and11 produced 100% draws in attacker mirror matches, while cap12 produced92% draws. Cap14 still gave the first receiver70.0% of decisive wins and30.0% of all rallies ended before the second return.

Confirmation: independent seeds3,100,000,000–3,100,000,199, five rallies each, both sides, 2,000 rallies/pair/cap.

| Metric | v1: cap15 | Chosen v2: cap13 | cap12 |
| --- | ---: | ---: | ---: |
| Attacker vs itself: median returns in scored rallies | 1 | 5 | 4 |
| Attacker vs itself: first receiver wins, decisive only | 93.4% | 50.9% | 27.4% |
| Attacker vs itself: draws | 33.4% | 0% | 93.8% |
| Attacker vs itself: ends before second return | 57.6% | 7.8% | 1.0% |
| Attacker vs itself: scores after ≥6 returns, all rallies | 0% | 49.8% | 0% |
| Attacker vs interceptor: wins | 98.7% | 62.4% | 2.9% |
| Attacker vs interceptor: median decisive returns | 2 | 7 | 3 |
| Frozen v1 Selfplay mirror: median decisive returns | 4 | 9 | 9 |
| Frozen v1 Selfplay mirror: draws | 16.3% | 43.1% | 79.8% |

Cap13 reduces the steep edge bonus without making the defensive script effectively unbeatable. The frozen network's draw rate demonstrates why retraining is necessary; it is not used to claim final v2 model quality. The two-stage comparison supports this choice among tested candidates; it does not establish a global optimum over all Pong rules.

## Human considerations

The fastest return now takes26 ticks (~0.87s) instead of22 (~0.73s). A near-edge offset14 produces speed12 (28 ticks), instead of14 (24 ticks). Humans get more time; immediate returns can still punish bad positioning. A five-tick-delayed interception script was included as a sensitivity check: attacker matchups last a median3 decisive returns at cap13 versus2 at cap15, and early finishes fall from35.6% to21.0%. This is **not** a human playtest or a prediction of human win rates. Baseline Tracker remains an easier opponent. No human-only speed or physics modifier is used.

## Reproduce

From the repo root, using the training virtual environment:

```sh
.venv/bin/python training/scripts/balance.py
.venv/bin/python training/scripts/balance.py --count 200 --start 3100000000 \
  --caps 15 13 12 --pairs AA AD NN AT AH \
  --out training/results/balance-v2/holdout.json
```

Pair codes: A = attacker, D = interceptor, N = frozen v1 Selfplay, T = Tracker, H = five-tick-delayed interceptor. The attacker uses the candidate's exact speed formula when choosing shots. The delayed policy sees its own paddle in the delayed observation too; it is a deliberately limited controller. Both side assignments are included, so mirror matches contain mirrored duplicates. Percentages are descriptive; no independence-based confidence interval is claimed. Raw records are in `scan.json` and `holdout.json`.

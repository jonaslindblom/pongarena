# Experimental learned-weight comparison

This is a limited fine-tuning study, not an estimate of the best policy attainable under each rule. The released arena, its v3 contract, and its public models are unchanged. Each candidate starts from the same fully learned Contender checkpoint and keeps its plain `6 → 64 → 64 → 3` Tanh architecture. No trajectory calculation, scripted action, or fixed feature layer is present in the exported model.

## Matched search budget

For each of `baseline`, `angle8`, `angle9`, and `motion2`:

- 20 generations, each screening the current network and eight antithetic all-parameter mutations. Noise scales are `0.003`, `0.006`, `0.012`, and `0.024`.
- Eight fresh training seeds per generation. The two best mutations and incumbent are compared on 16 fixed training confirmation seeds; only a strict training improvement is accepted.
- Three equally weighted opponents: the rule-aware attacker, the rule-aware defender, and the original frozen Contender. The scripted opponents are training/evaluation tools, never components of the submitted model.
- Fitness is the mean across opponents of `point_margin + 0.3 × ordinary_five_rally_game_win_rate`, breaking ties by mean rally scoring rate. Draws remain in all denominators.
- Checkpoint selection uses 32 separate development seeds at generations 0, 5, 10, 15, and 20. Training and development seed lists, initial checkpoint hash, source hashes, raw generation metrics and selected generation are retained.
- Initial weights, mutation RNG, seed sets, opponent roles and generation counts match across rules. **Active simulation decisions and wall time differ** because rallies end at different times; this is the same search budget, not identical numbers of gradient updates or game ticks.

Training seeds are sampled below one billion with RNG `20261105`. Development seeds use RNG `20261106` in `[1.6 billion, 1.7 billion)`. No final confirmation seeds are used by this script. Five standard serves are run with the candidate on both sides for each seed. All opponents are deterministic and stateless, so the candidate engine uses the ordinary 1,600-tick limit and exact post-contact loop termination.

The outer search loop uses PyTorch parameters and batched matrix multiplication, not PPO. It tests local adaptations of an existing learned policy. A failure to score after this small search does not prove that a more thoroughly trained network cannot learn to attack.

## Files and reproduction

The retained comparison is `search/`. Each rule directory contains the unchanged starting weights, selected and last checkpoints, standalone experimental ONNX files, `config.json`, `selection.json`, development curves, training search records and export verification. `search/learning-curves.png` and `.svg` plot the development objective and mean scoring rate across the three opponents. The mean scoring curve includes games against frozen Contender; inspect the per-opponent metrics when assessing attacking ability against the stronger scripts.

```sh
MPLCONFIGDIR=/private/tmp/rally-offense-matplotlib .venv/bin/python training/scripts/train_offense.py \
  --out training/results/offense-study/training/new-search --check
```

Use a fresh output directory. The default command repeats all four rules with the recorded search budget. It writes experimental rule metadata (`offense-study-baseline`, `offense-study-angle8`, etc.) and does not label these weights as an official v3 or v4 submission.

Population inference is checked against ordinary PyTorch inference; duplicated population members are compared with the separate pair evaluator. Each final ONNX is checked for exactly three `Gemm` layers and two `Tanh` layers, exact equality of all checkpoint parameters, correct experimental metadata, and action parity with PyTorch on 8,192 numeric probes. These probes check export/runtime agreement, not playing strength.

## Completed development results

All four runs completed 20 generations: 84,721,236 active candidate decisions in total, approximately 238 seconds summed over the four runs. Selection used only the fixed development set; the numbers below are **development measurements**, not independent final results. Each opponent column contains 320 rallies, five serves on both sides for each of 32 seeds.

| Rules | Selected generation | Adaptive attacker W/L/D | Adaptive defender W/L/D | Starting Contender W/L/D |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 5 | 0 / 3 / 317 | 0 / 0 / 320 | 3 / 2 / 315 |
| Angle8 | 15 | 0 / 138 / 182 | 0 / 62 / 258 | 182 / 130 / 8 |
| Angle9 | 10 | 0 / 233 / 87 | 0 / 197 / 123 | 146 / 155 / 19 |
| Motion2 | 5 | 0 / 262 / 58 | 0 / 124 / 196 | 178 / 137 / 5 |

Every selected model still scored **zero points against both adaptive scripts**. Baseline fine-tuning mainly removed defensive mistakes; angle8 and motion2 improved against the starting neural opponent. These runs do not establish that the revised rules support learned offensive mastery. In particular, motion2 improved against the attacker but conceded more against the defender than its starting network (124 versus 105), a tradeoff accepted by the aggregate training objective.

Each of the four exports passed all 8,192 action comparisons, with zero mismatches and maximum absolute score error at most `4.77e-6`. Learning curves were generated and visually inspected. Independent confirmation is recorded separately by the study's frozen evaluation protocol.

`search/opponent-source.py` preserves the exact adaptive-opponent source used for training, SHA-256 `1167be32c2b9dcb9c1016a83e51613b07ec8eb2371ebf1e6d7613b907b8d8bc6`. After training, the evaluator gained terminal-frame replay recording for the viewer; policy behavior and physics did not change. The per-run source hashes refer to the archived training source.

## Superseded pilot

`pilot/` measured one baseline generation before launching the comparison. The first multi-rule run was stopped after a review found that the motion2 policy's hypothetical reply envelope included two impossible final paddle/movement combinations. Actual contact actions were already filtered by exact movement reachability, and baseline/angle rules were unaffected, but all four runs were restarted with one corrected policy implementation. The stopped outputs are retained in `superseded-policy-envelope/` with an explicit invalidation record. Neither pilot nor superseded outputs are eligible for final checkpoint selection.

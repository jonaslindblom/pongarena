# Why the released v3 policy regressed

These are diagnostics of frozen weights under the unchanged Pong v3 rules, **not final testing of a new policy**. They use 100 dispersed development seeds sampled without replacement from [1,200,000,000,1,300,000,000), sampler seed 1847, five serves and both side assignments. Each row below contains 1,000 rallies. Raw seeds, outcomes, reward components and policy hashes are in `diagnosis.json`.

## What is verified

The v3 run changed both the training episode horizon and the validation seeds; it did not simply continue the best v2 weights. All three v3 runs started again from random initialization and imitation. The selected v3 policy has a different opponent-specific tradeoff, and performance is not monotonic with a longer deadline or more training.

| Frozen policy | Attacker W / L / D | Interceptor W / L / D | Original v2 W / L / D |
| --- | --- | --- | --- |
| Released v2 | 110 / 406 / 484 | 830 / 65 / 105 | 309 / 309 / 382 |
| v3 seed 7 selected | 105 / 436 / 459 | 581 / 121 / 298 | 349 / 372 / 279 |
| v3 seed 17 selected | 87 / 487 / 426 | 788 / 52 / 160 | 296 / 400 / 304 |
| v3 seed 29 released | 192 / 334 / 474 | 454 / 31 / 515 | 191 / 412 / 397 |

These development results reproduce the qualitative regression observed on the prior published test set: seed 29 improves attack resistance but loses scoring ability against defense and loses to the original v2 policy. Other training seeds have different tradeoffs; none consistently beats both requested targets.

### The training objective and requested objective differ

- **Original v2 never appears in training or validation opponents.** The league contains three easy baselines, two scripts and snapshots of the current learner. It cannot directly optimize beating v2 if that opponent is absent.
- **Checkpoint selection averages five opponents equally.** Three of five are the easy baseline models; only one is the attacker, and v2 is absent. The requested success criterion—beating both v2 and attacker—is a different objective. This observation does not by itself prove a different saved checkpoint would solve the problem.
- **Long-delayed points are heavily discounted.** With gamma .99, a reward 400 ticks away has only 1.795% of its immediate value; at 800 ticks it has 0.0322%. Return bonuses arrive much sooner. GAE lambda .95 also limits direct long-horizon credit, leaving the critic to propagate distant outcomes.
- **The return bonus can reward a losing policy.** For released v3 vs v2 with greedy actions and the full 1,600-tick training horizon, the model scores 191W/ 412L/ 397D, but its mean discounted training return is positive: +.003826. The discounted net point component is −.016582, outweighed by +.020409 from return bonuses. Against the interceptor, 98.0% of the net discounted return comes from return bonuses: .018708 out of .019096. The point component is only .000388 despite 454 wins and 31 losses.

The reward calculations are recomputed on actual trajectories with the recorded training formula: terminal ±1 plus .02 for each own return, discounted by .99 from the first action. They establish an objective mismatch; a controlled training ablation is still needed to assign causal responsibility for the regression.

### Sampling changes the task the learner experiences

The browser uses deterministic first-argmax. PPO samples the same logits. A small per-action chance of moving differently compounds across a long rally and can change contact placement or escape a deterministic loop.

| Released v3 opponent | Greedy W / L / D | Sampled W / L / D |
| --- | --- | --- |
| Attacker | 192 / 334 / 474 | 65 / 825 / 110 |
| Interceptor | 454 / 31 / 515 | 536 / 177 / 287 |
| Original v2 | 191 / 412 / 397 | 264 / 618 / 118 |

Both columns use the full 1,600-tick horizon with no loop shortcut. The sampled column uses one reproducible action RNG stream; it is a diagnostic, not a confidence interval over sampling repetitions. Deterministic loop exits produce exactly the same point outcomes as continuing every greedy rally to 1,600 for every policy/opponent pairing tested. Thus loop detection itself does not explain the score regression.

Sampling is necessary exploration for the original PPO recipe, and this gap is not itself an implementation bug. It means that training reward measured under a sampled policy is not a direct measure of deployed greedy strength. A useful check is to track deterministic target performance throughout training, and consider lower exploration near precision contacts, a deterministic value-based method, or a stronger imitation/planning warm start.

### A league-maintenance defect adds nonstationarity

After the history pool exceeds five snapshots, `train.py` removes the oldest snapshot and rerandomizes **every** lane's `opponent_ids`, including unfinished rallies. This first occurs at PPO update 125 and recurs every 25 updates. An opponent can therefore change in the middle of a rally. A fixed opponent assignment until the rally terminates would provide a more coherent learning target. The released seed 29 checkpoint is from update 100, before this first replacement, so this defect **cannot explain that checkpoint's regression**, although it affects later training and the seed 7 selected checkpoint at 225.

## What to change in training

Keep physics, deadline, observation and deployment rules fixed. Start from an existing competent policy, include both original v2 and the attacker explicitly, and select on the weaker of the two target results rather than a mean dominated by easy opponents. Test a terminal-focused reward and longer credit horizon. Keep opponents fixed within each rally and preserve deterministic deployment evaluations.

Separate these confirmed properties from untested hypotheses: architecture capacity may matter, the opponent-unconditioned critic may have a noisy value target, and different rewards may improve offense. This diagnostic does not prove any one of those changes will produce a broadly stronger model.

## Reproduce

```sh
.venv/bin/python training/scripts/diagnose_training.py
```

The script evaluates four frozen actors × three opponents × three action/termination modes = 36,000 development trajectories. It asserts that every greedy loop-shortcut outcome agrees with its full-horizon continuation. Paired sides are dependent observations, especially for a policy playing itself. These descriptive counts are not independent-rally confidence estimates. The same six observations and original v3 physics are used throughout; no rule or model file changes.

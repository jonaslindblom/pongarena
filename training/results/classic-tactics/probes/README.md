# Classic placement counterfactuals

The frozen `pong-v4` rules are unchanged. The test asks whether a reachable
alternative contact improves the result against the same fixed script. It does
not solve optimal play or rank a scripted action intervention as a model.

`protocol.json` reserves disjoint training, development and final seed sets.
Each seed supplies all five rallies for attacker/attacker and attacker/defender
matches. Every distinct physical flight-start state in each match is tested,
for whichever side is receiving. Repeated identical states within the same
match are counted once. Observations can still recur across seeds and pairs.

For each state, the unchanged scripts form one baseline branch. Every exactly
reachable legal contact center forms another branch. The controlled paddle
follows a deterministic legal path until that contact, then resumes its original
script. The opposing script acts normally throughout. Both paddles decide from
the same pre-move frame. No paddle is teleported.

Every branch retains the original elapsed rally ticks and the 1,600-tick
deadline. **Loop shortcuts are disabled for all diagnostic branches**, including
the baseline, because an intervention changes the continuation policy. Full
matches between fixed learned policies are evaluated separately with the normal
loop rule. Outcome ordering is win, draw, loss; longer survival alone does not
count as improvement.

| Split | Unique flight starts | Enumerated branches, including baseline | States with an improved outcome | Strict loss → win states | Path observations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training, 16 seeds | 1,972 | 22,567 | 578 | 414 | 16,898 |
| Development, 8 seeds | 965 | 10,939 | 294 | 207 | 8,367 |

These are correlated diagnostic counts, not independent statistical trials.
The final set is reserved until training checkpoint selection is frozen.

Each NPZ retains only one strict loss-to-win path per qualifying state. Ties
prefer fewer action departures from the baseline pre-contact path, then lower
contact center. `obs` contains float32 six-input observations; `labels` contains
the selected legal actions; `original_labels` contains the original script's
action on those same observations. `source_seeds` and `branch_ids` retain the
grouping. Labels use a continuation against a known fixed opponent; they are
not universal optimal-action labels. Identical observations and potentially
conflicting labels must be reported by the learning experiment.

The JSON files retain every baseline match and branch. Joint-action strings
encode each simultaneous left/right action pair as `left * 3 + right`, with
ordinary actions numbered up=0, stay=1, down=2. `snapshots` points back to its
baseline match and elapsed tick, so its complete legal prefix is retained.
`candidates` includes terminal states, outcomes, actual contact centers and all
continuation actions. `generator-source.py` exactly matches the recorded source
hash for the training/development data.

`representative.json` is an explicitly selected training-set illustration,
not a favorable held-out result. At seed 3,507,017,852, rally 2, after six prior
returns, changing the left paddle's contact center from 24 to 25 changes a loss
at tick 436 into a win at tick 358. The shared prefix and both continuations are
reconstructed through the independent scalar engine. The auditor additionally
checked that the winning full trajectory has no conflicting actions for an
identical controlled-side float32 observation and no premature exact loop.

Reproduce the retained seed comparisons in a fresh output directory:

```sh
.venv/bin/python training/scripts/classic_tactics.py --phase train --out /tmp/classic-tactic-reproduction
.venv/bin/python training/scripts/classic_tactics.py --phase dev --out /tmp/classic-tactic-reproduction
.venv/bin/python training/results/classic-tactics/probes/export_representative.py
```

`evaluate_policy(data, model)` in `classic_tactics.py` lets a learned policy
control only the tested approach/contact, then resumes the original scripts.
It reports actual placement and continuation outcomes, rather than just action
agreement. Whether the same model improves in complete games is a separate
evaluation. Reproducing a retained final block does not make its seeds fresh for
future model selection.

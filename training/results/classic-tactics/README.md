# Can learned weights exploit better shot placement?

This experiment keeps the released **Pong v4** physics, six observations, three actions, model budgets and match scoring fixed. It separates two questions: whether a strong scripted controller leaves a better legal choice available, and whether training a plain network on such choices improves held-out play.

## Experimental design

1. Generate real matches between the existing adaptive scripts. Inspect incoming flights and enumerate exactly reachable paddle contact centers. Execute each candidate through legal tick-by-tick actions, then resume the same controllers. Retain unsuccessful alternatives as well as improvements.
2. Compare the original continuation with each alternative at the original rally deadline. A scheduled one-off intervention is history-dependent, so these local probes disable the exact-loop shortcut for **both** alternatives and baseline. They do not reset the elapsed rally clock. These probes are counterfactual diagnostics, not leaderboard submissions.
3. Continue two copies of the same frozen Pacer network with the same observations and optimization budget. Treatment uses improving-placement action labels; control uses the original script's labels on those same positions. Inference remains a plain learned `6 → 128 → 128 → 3` Tanh network.
4. Select checkpoints on development data, freeze hashes, then evaluate new tactical examples and complete games on separate seeds. Complete matches use the arena's normal exact-loop rule for deterministic stateless policies, all five serves and both side assignments.
5. Compare treatment with both its equal-budget control and the unchanged Pacer baseline. Report full-match outcomes against the adaptive attacker, adaptive defender and Pacer, with paired differences resampled by whole serve seed.

The independent checker uses the frozen dependency-free `training/v4/reference.py`. It reconstructs each recorded prefix, executes the alternative action path, checks the resumed trajectory and original deadline, and checks whether identical observations require conflicting actions in the displayed example.

Training/discovery, development and final evaluation occupy separate seed sets. Exact seeds are checked against prior experiments; disjoint-looking numeric ranges alone are insufficient. Published final seeds become consumed and cannot select future improvements.

## Interpretation

A legal improving branch establishes a weakness in that particular controller's continuation. It does not prove optimal Pong is winning, that the whole tactic can be forced against every opponent, or that a small network can represent it. Better local imitation does not establish stronger complete-match play. A negative training result is evidence about this method and budget, not a proof that learning cannot succeed.

The original teacher supplies only a heuristic placement estimate. The intervention tests whether outcome-based branch labels add useful information beyond more training with that teacher. It does not compare all possible reinforcement-learning methods.

No rule changes or model promotions are implied by this experiment. Results, exact commands and verification will be added after the frozen evaluations complete.

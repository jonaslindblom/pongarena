# Can learned weights exploit better shot placement?

This experiment keeps the released **Pong v4** physics, six observations, three actions, model budgets and match scoring fixed. It separates two questions: whether a strong scripted controller leaves a better legal choice available, and whether training a plain network on such choices improves held-out play.

**Result:** legal placement improvements exist, but the tested learned continuation is not an overall upgrade. It improves against the defender and regresses against the attacker. Arena presets and physics remain unchanged. Open `/training/classic-tactics.html` for two synchronized counterfactual replays, both attempts' learning curves and final comparisons.

## Findings

| Probe phase | Serve seeds | Unique flight-start states | Branches including unchanged baseline | Loss-to-win opportunities |
| --- | ---: | ---: | ---: | ---: |
| Training discovery | 16 | 1,972 | 22,567 | 414 |
| Development | 8 | 965 | 10,939 | 207 |
| Final, after weights froze | 16 | 1,950 | 22,350 | 410 |

These are correlated conditional opportunities against the specified script continuations, not independent match trials. Every candidate contact uses an exactly reachable action path. In the final block, 269 of 410 strict opportunities have at least three winning contact centers; 172 have at least three consecutive integer centers. Outplay is not confined to single-unit precision in this sampled controller field.

The displayed broader example starts after four returns. A near-center contact at paddle Y=141 sends a flat ball across at Y=142, keeping the opponent near the bottom. After the same two scripts resume, a later shot targets the top while the opponent cannot recover. The original contact at Y=132 loses. Eight of the fifteen reachable contact centers win in this continuation, including 139–144. A second, narrower example changes Y=24 to Y=25 after six returns. Both examples are explicitly selected training-set illustrations, not randomly selected final tests.

## Controlled learning results

The first 24-round continuation used many tactical rows and both arms regressed; development selection retained the initial Pacer weights. One bounded follow-up used a smaller learning rate, fewer tactical rows and a KL anchor to Pacer's original output distribution. It ran 48 rounds, selected treatment round 44, and again selected unchanged Pacer for the control. The fully trained control at round 48 is retained as a secondary diagnostic. No final outcome selected these checkpoints or triggered another training attempt.

Both attempts used the same `6 → 128 → 128 → 3` architecture with 17,795 learned parameters, matching observations, minibatches and optimization budget between arms. Control labels come from the original adaptive **attacking teacher**, including on positions originally collected from the defending script. Treatment changes only the oversampled successful-path labels. [Learning methods and reproduction](learning/README.md) documents the exact settings and failed first attempt.

| Fresh full matches, 640 rallies per pairing | Unchanged Pacer W / L / D | Selected treatment W / L / D |
| --- | --- | --- |
| Adaptive attacker | 156 / 453 / 31 | 114 / 508 / 18 |
| Adaptive defender | 78 / 450 / 112 | 94 / 432 / 114 |
| Pacer | 310 / 310 / 20 | 301 / 320 / 19 |

Paired point-margin change versus Pacer is **−0.1516** against the attacker (95% whole-seed bootstrap interval −0.2141 to −0.0891), **+0.0531** against the defender (+0.0063 to +0.1000), and −0.0297 against Pacer (interval crosses zero). These intervals are descriptive and not adjusted for multiple comparisons. The primary selected control has the same weights as Pacer, so it produces identical outcomes. The final trained control, all per-seed results and paired differences are retained rather than silently discarded.

On the final placement probes, the selected treatment finds winning placements in **54/410** strict opportunities, compared with **55/410** for Pacer. Its action-label agreement nevertheless rises from 80.30% to 81.76%. Thus higher imitation accuracy does not establish better placement or stronger complete matches.

Disjoint seeds do not imply disjoint states: 12,162 of the 16,616 final path observations (73.2%) already occur in tactical training paths when numeric +0 and −0 are treated as equal. Among the 103 strict opportunities whose initial observation is absent from those paths, treatment wins 25 and Pacer 26. This subset is unseen relative to the tactical paths, not necessarily the original Pacer training history or the continuation's pooled observations. `verification.json` records this canonical numeric audit; the frozen evaluator's original byte-based partition counts signed zeros separately and reports 104 rather than 103.

## Experimental design

1. Generate real matches between the existing adaptive scripts. Inspect incoming flights and enumerate exactly reachable paddle contact centers. Execute each candidate through legal tick-by-tick actions, then resume the same controllers. Retain unsuccessful alternatives as well as improvements.
2. Compare the original continuation with each alternative at the original rally deadline. A scheduled one-off intervention is history-dependent, so these local probes disable the exact-loop shortcut for **both** alternatives and baseline. They do not reset the elapsed rally clock. These probes are counterfactual diagnostics, not leaderboard submissions.
3. Continue two copies of the same frozen Pacer network with the same observations and optimization budget. Treatment uses improving-placement action labels; control uses the original attacking teacher's labels on those same positions. Inference remains a plain learned `6 → 128 → 128 → 3` Tanh network.
4. Select checkpoints on development data, freeze hashes, then evaluate new tactical examples and complete games on separate seeds. Complete matches use the arena's normal exact-loop rule for deterministic stateless policies, all five serves and both side assignments.
5. Compare treatment with both its equal-budget control and the unchanged Pacer baseline. Report full-match outcomes against the adaptive attacker, adaptive defender and Pacer, with paired differences resampled by whole serve seed.

The independent checker uses the frozen dependency-free `training/v4/reference.py`. It reconstructs each recorded prefix, executes the alternative action path, checks the resumed trajectory and original deadline, and checks whether identical observations require conflicting actions in the displayed example.

Training/discovery, development and final evaluation occupy separate seed sets. Exact seeds are checked against prior experiments; disjoint-looking numeric ranges alone are insufficient. Published final seeds become consumed and cannot select future improvements.

## Interpretation

A legal improving branch establishes a weakness in that particular controller's continuation. It does not prove optimal Pong is winning, that the whole tactic can be forced against every opponent, or that a small network can represent it. Better local imitation does not establish stronger complete-match play. A negative training result is evidence about this method and budget, not a proof that learning cannot succeed.

The original teacher supplies only a heuristic placement estimate. The intervention tests whether outcome-based branch labels add useful information beyond more training with that teacher. It does not compare all possible reinforcement-learning methods.

One limitation of these labels is the continuation assumption: the branch search optimizes one contact, then hands control back to the scripts. A network playing an entire match must also make every later recovery and placement decision. This is a plausible reason for weak transfer, not an isolated causal diagnosis. Neither this imitation recipe nor its restrained follow-up establishes the ceiling of outcome-based learning or self-play.

## Verification and reproduction

- [Probe protocol, data format and commands](probes/README.md) includes complete prefixes, all alternative action paths and both displayed examples.
- [Training and evaluation commands](learning/README.md) includes both retained attempts, the pre-final freeze, exports and numerical checks.
- `check_classic_tactics.py` independently checks branch actions through the frozen scalar reference, reachable contact sets, path labels, observation consistency and seed separation. `verification.json` records exact counts.
- Browser ONNX inference matches all six native full-game fixtures: **12,476 frames / 24,904 decisions**, zero action or state disagreements, maximum score difference 9.54e-6. See `browser-verification.json`.
- `report_classic_tactics.py` builds the public report, PNG/SVG curves and raw measurements without rerunning training or selecting weights.

```sh
.venv/bin/python training/scripts/check_classic_tactics.py
.venv/bin/python training/scripts/report_classic_tactics.py
```

The scalar and TypeScript game sources remain unchanged. No experimental weights are promoted to the active arena. Native and browser parity verify implementation agreement; full-match performance is measured separately with the frozen PyTorch checkpoints.

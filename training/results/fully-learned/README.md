# Contender: fully learned weights, fixed Pong v3 rules

**Contender beats original v2 and the unchanged scripted attacker under the stricter ordinary-game criterion in all three reserved test blocks.** It is a plain `6→64→64→3` Tanh MLP: 4,803 learned parameters, five ONNX operators, 4,672 MACs per decision and 19,643 bytes. No trajectory calculation, target-position feedback, hand-written action rule or extra actor input is embedded in the submitted graph. Normal arena observation normalization and first-argmax action selection are unchanged.

Open `/training/fully-learned.html` for the curves, comparisons and final block results. The arena now defaults to **Contender vs Selfplay v2**. The library exposes its actual architecture and downloadable ONNX file. The earlier hybrid Strategist remains labelled as a hybrid; it does not satisfy this experiment's fully learned requirement.

## Final results

One export was frozen in commit `955f5d3` before testing. Three disjoint blocks each contain 200 fresh seeds, five serves and both sides. Final evaluation uses the actual candidate and v2 ONNX files. Each opponent therefore faces 6,000 rallies, or 1,200 ordinary five-rally games. Paired contests aggregate the two side assignments; they are not additional independent observations.

| Opponent | Rally W / L / D | Ordinary game W / L / D | Game win rate | Paired contest W / L / D |
| --- | --- | --- | ---: | --- |
| Original Selfplay v2 | 2,360 / 70 / 3,570 | 1,086 / 5 / 109 | 90.5% | 594 / 0 / 6 |
| Scripted attacker | 1,339 / 97 / 4,564 | 822 / 23 / 355 | 68.5% | 540 / 5 / 55 |
| Interceptor guardrail | 199 / 66 / 5,735 | 181 / 59 / 960 | 15.1% | 158 / 51 / 391 |

Draws count in every win-rate denominator. Attacker game wins are 69.75%, 70.5% and 65.25% in the three blocks; v2 game wins are 91.25%, 91% and 89.25%. Every primary matchup/block also has majority paired-contest wins, a positive lower endpoint for the 95% whole-seed bootstrap point-margin interval, and more rally wins than losses on each physical side. The final protocol was committed in `5a571fd` before candidate selection; `protocol.json` and `holdouts/` retain the exact definitions, hashes, outcomes and uncertainty estimates.

This is targeted improvement, not universal dominance. Contender draws most rallies against the interceptor and wins fewer games against that opponent than its starting policy did on the shared development set. The interceptor is a reported guardrail, not a primary acceptance target. Even against the attacker, most individual rallies draw; ordinary matches usually win by taking some points without conceding many.

## How the model learned

The starting weights came from the earlier fully learned PPO branch, not from the hybrid Strategist. Its lineage is original learned v2 → terminal-reward PPO fine-tuning → undiscounted PPO fine-tuning. Original v2 used a supervised warm start followed by PPO. The retained baseline audit confirms that every parameter changed through subsequent learning, and the inference graph has no engineered feature branch.

We tried three more PPO variants and two direct weight-search variants. Two PPO continuations regressed and were stopped, with their failed curves and interruption records retained. Sharper action sampling and a stronger KL penalty against the learned initial policy improved the third PPO run. On the shared 512 fresh development seeds, its ordinary attacker-game win rate rose from the starting policy's 51.37% to 57.62%.

Direct weight search was stronger. It perturbs learned weights, tests each candidate's actual deterministic actions on matched training serves, and retains improvements. It adds no computation to the network. Full-network search selected generation 25: 67.97% ordinary attacker-game wins and 89.84% against v2 on the shared 512-seed comparison. Output-layer-only search reached 58.69% against the attacker there. The full-network candidate was selected before opening the final tests. This model uses gradient training followed by derivative-free optimization; it is not a claim that PPO alone produced the final gain.

All 4,803 affine weights and biases belong to the learned model. Every exported tensor and node matches a plain PyTorch checkpoint exactly. The actor sees only the original six observations and emits three action scores. A training-only critic can see opponent identity and remaining time; neither reaches the exported policy. Training rewards, loss functions and weight mutations are training choices, not game-rule changes.

## Reproduce training and selection

Run from the repository root with the pinned environment in `training/requirements.txt`. Use fresh output directories. The released full-weight search can be reproduced from the retained learned PPO baseline:

```sh
.venv/bin/python training/scripts/train_search.py \
  --out training/runs/reproduce-contender --seed 91 --generations 30 \
  --scope all --sigma .003 --check
```

The selected checkpoint is generation 25, not necessarily the final generation. Training uses serves below 1 billion and separate 64-seed development validation. A subsequent shared 512-seed development comparison selects between experiment finalists. Three reserved final blocks in [3.0 billion, 3.3 billion) are evaluated once after freezing the chosen export.

[SEARCH.md](SEARCH.md) records both search commands, exact fitness and seed selection, population-evaluator checks, and retained artifacts. [PPO.md](PPO.md) records all three PPO commands, critic warm-up, sampling, anchoring and interruptions. The earlier [focused PPO methods](../training-improvement/PPO.md) reproduce the starting policy's lineage. Runs retain configurations, selected checkpoint/export pairs, full learning measurements and console logs. Exact resume and bitwise training portability across platforms are not promised.

## Verify and evaluate

```sh
.venv/bin/python training/scripts/check_plain_protocol.py
.venv/bin/python training/scripts/check_plain_model.py \
  --checkpoint training/results/fully-learned/candidate/best.pt \
  --onnx public/models/contender-v3.onnx \
  --output /tmp/contender-model-check.json
.venv/bin/python training/scripts/verify_competitor.py \
  --results training/results/fully-learned --name contender-v3 --seed-start 1370000000
node --import tsx training/scripts/check_replays.ts training/results/fully-learned/verification/onnx-replays.json
.venv/bin/python training/scripts/plain_report.py
npm test
npm run build
```

The final evaluator is `evaluate_frozen.py --results training/results/fully-learned --block 0` (then blocks 1 and 2). It verifies frozen model/rules/opponent/seed hashes and refuses to overwrite recorded results. For an independent rerun, use `competition_eval.py` with an explicit seed-array JSON and a new output path. That general evaluator reports the earlier paired criterion; `evaluate_frozen.apply_protocol` additionally enforces this experiment's stricter ordinary-game majority. Future tuning must use fresh development data and new final seeds; the published tests are now historical evidence.

The selected model agrees with native ONNX on all 8,192 reachable PyTorch observations. Six complete matches reproduce in the live TypeScript engine and browser WASM: 38,914 actions, 19,481 frames, 22 loop draws and one timeout, with identical decisions and physical states. The browser's maximum score difference is 8.58e-6. Verification records are under `verification/` and `candidate/model-check.json`. Rules, scripted opponents, upload limits and standalone v1/v2/v3 packages are unchanged.

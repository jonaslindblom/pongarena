# Training improvement with Pong v3 rules fixed

**Strategist passes the predeclared success criteria against both original Selfplay v2 and the unchanged scripted attacker in each of three reserved test blocks.** Only one candidate was tested on these final seeds. Its checkpoint and ONNX hash were frozen in commit `683c40f` before evaluation. Physics, observations, output meanings, deadlines, loop detection and upload limits are unchanged.

Open `/training/improvement.html` in the local app for learning curves, development comparisons, the diagnosis and final results. The model is selectable as **Strategist** and defaults to playing Selfplay v2. Download `public/models/strategist-v3.onnx` or use the library's download button.

## Final results

Each opponent faces 600 fresh dispersed seeds, all five serves and both physical sides: 6,000 rallies, 1,200 ordinary five-rally games, or 600 paired ten-rally contests. These are different aggregations of the same trajectories.

| Opponent | Rally W / L / D | Five-rally game W / L / D | Paired contest W / L / D |
| --- | --- | --- | --- |
| Original v2 | 2,917 / 1 / 3,082 | 1,153 / 0 / 47 | 600 / 0 / 0 |
| Scripted attacker | 790 / 1 / 5,209 | 625 / 1 / 574 | 474 / 1 / 125 |
| Interceptor guardrail | 2,608 / 1 / 3,391 | 1,121 / 0 / 79 | 582 / 0 / 18 |

This is a consistent advantage, not a win in every match. Against the attacker, 86.8% of rallies draw, all from exact loops. Ordinary browser-length games win 52.1%, lose 0.08% and otherwise draw; paired contests win 79%. The acceptance criterion requires more than half of paired contests won, a positive lower endpoint for the whole-seed bootstrap point-margin interval, and more wins than losses separately on each side, against both primary targets in every block. All pass. See `protocol.json` and `holdouts/` for block results, per-seed outcomes and intervals. The interceptor is a reported guardrail, not a primary selection target.

## Why v3 regressed

The original v3 recipe never trained or validated against v2. Checkpoint selection averaged several opponents, including three easy baselines. It therefore optimized a different contest from the one requested here. Its `.99` discount reduces a point 400 ticks away to 1.8% of immediate reward, while the `.02` return bonus can make a losing policy's discounted training return positive. Sampling during training also produced very different outcomes from first-argmax deployment. The [diagnostic study](../training-diagnosis/README.md) quantifies these effects.

Exact loop exits preserve the outcomes of the tested fixed greedy policies; they did not cause this regression. A separate historical-opponent replacement bug was found in the old training loop, but it begins after the released checkpoint and cannot explain that checkpoint's performance. These are supported mechanisms and objective mismatches, not an isolated causal attribution of every loss.

## What changed in training

Focused PPO included v2 and attacker as fixed per-rally opponents and selected the weaker target score. Removing return bonuses and then fine-tuning with undiscounted point outcomes produced a small, fully learned `6→64→64→3` actor that beat both targets on the shared development set. It was less reliable than the selected candidate and was not final tested. [PPO.md](PPO.md) records all five variants and commands.

The stronger route used DAgger supervision from a teacher with better recovery positioning. Raw-input action imitation and target regression failed. The selected hybrid embeds fixed ReLU trajectory calculations, a learned `8→128→128→64→1` Tanh target head, and fixed paddle feedback in one stateless ONNX graph. It still receives only the original six inputs and produces three scores; no external script participates at inference. Of 48,063 total parameters, 25,985 are learned. The export is 208,140 bytes and uses 45,152 MACs per decision. Its geometry is a disclosed hand-designed prior, not something discovered entirely through training.

[DISTILLATION.md](DISTILLATION.md) contains exact commands, failed variants, the interrupted run's provenance and reproduction caveats. The selected seed 83 checkpoint is round 12; its planned 24-round schedule was interrupted after 16 to investigate another loss function. Preserve `--rounds 24 --classification-weight .01` when reproducing it. No exact-resume checkpoint or cross-platform bitwise-training guarantee is provided.

## Verification and reproduction

The environment remains pinned in `training/requirements.txt`. From the repository root:

```sh
.venv/bin/python training/scripts/check_competition_eval.py
.venv/bin/python training/scripts/verify_trajectory_model.py
.venv/bin/python training/scripts/verify_competitor.py
node --import tsx training/scripts/check_replays.ts training/results/training-improvement/verification/onnx-replays.json
.venv/bin/python training/scripts/improvement_report.py
npm test
npm run build
```

`evaluate_frozen.py --block 0` (then 1 and 2) is the exact final-test entry point. It verifies frozen hashes and refuses to overwrite recorded results. For independent repeat checks, pass an explicit seed list and a new output file to `competition_eval.py`; use fresh development seeds for future tuning. Published final seeds are now historical evidence, not a fresh test set for another candidate.

Verification includes 393,226 geometry probes, exact selected-checkpoint/export correspondence, 8,192 native ONNX/PyTorch action comparisons, and six complete matches in the live TypeScript engine and browser WASM: 26,838 decisions, 13,443 frames, 16 loop draws, three timeouts, no action or physical-state discrepancies. The recorded local mean inference time for both players was about 0.052 ms per tick, excluding rendering; this is not a cross-device guarantee. See `verification/`.

# Fully learned starting policy audit

The starting checkpoint is `training/results/training-improvement/runs/focus-outcome-stage2-s73/best.pt`: a plain **6→64→64→3 Tanh MLP**, selected at PPO update 275 during the earlier development investigation. It contains 4,803 trainable parameters, five ONNX operators, 4,672 MACs and 19,643 bytes. Its three learned affine layers connect directly from the six public observations to the three action scores. There are no trajectory features, feedback branches, policy buffers, fixed scalar targets or additional inference inputs.

`check_plain_model.py` re-exports the checkpoint through the plain MLP exporter and requires exact graph/tensor correspondence with the saved ONNX file. The actual arena validator accepts it. On 8,192 reachable development observations, native ONNX and PyTorch agree on every action; maximum score error is 4.77e−6. This validates graph structure and numerical correspondence, while the run records establish training provenance.

The checkpoint lineage is original v2 actor → terminal-only focused PPO stage 1 → undiscounted focused PPO stage 2. Every one of the 4,803 parameter values differs from both original v2 and the stage-1 checkpoint; no layer was frozen in the focused PPO optimizer. Original v2 itself was trained with a DAgger warm start and PPO. Supervision in that historical training is distinct from fixed inference calculations. The rejected hybrid Strategist was never part of this checkpoint lineage.

## New development benchmark

These 64 dispersed seeds use NumPy sampler 502 in `[1,300,000,000, 1,400,000,000)`. They are development data shared with the new training experiments, not final held-out evidence. Each opponent faces 640 rallies, 128 five-rally games and 64 paired ten-rally contests.

| Opponent | Rally W / L / D | Ordinary game W / L / D | Ordinary game wins | Paired contest wins |
| --- | --- | --- | ---: | ---: |
| Original v2 | 249 / 74 / 317 | 99 / 17 / 12 | 77.34% | 87.50% |
| Scripted attacker | 133 / 46 / 461 | 71 / 19 / 38 | 55.47% | 65.63% |
| Interceptor | 155 / 49 / 436 | 75 / 13 / 40 | 58.59% | 73.44% |

Drawn games remain in the win-rate denominator. The seed-cluster 95% point-margin intervals are `[.2203, .3234]`, `[.0969, .1766]`, and `[.1250, .2078]`, respectively. Both physical sides have more rally wins than losses. Full outcomes are in `development.json`; exact seeds are in `development-seeds.json`.

Requiring a majority of ordinary five-rally games in every reserved block is stricter in practice than the earlier paired-only criterion and directly reflects the arena's visible game length. It is not mathematically interchangeable with paired performance: the two side scores can offset, so report and check both. The starting attacker's 55.47% rate has limited margin on this small development set; it must not be presented as final proof of consistency.

## Training audit

The old focused trainer's actor consumes only the original six inputs. Its training-only critic additionally receives opponent identity and remaining time; neither is exported or read by the deployed actor. That is training context, not hidden inference information. Opponents stay fixed until a rally ends. Stochastic training correctly disables exact-loop exits, while fixed greedy validation enables them. Points and the 1,600-tick deadline are unchanged.

No GAE indexing or terminal-mask error was found in the focused PPO implementation. A newly initialized critic can still provide poor early advantages when fine-tuning a competent actor; critic warm-up and conservative actor updates are reasonable safeguards. Sampling versus greedy deployment remains a real optimization mismatch, so deterministic evaluation and direct weight search are useful complementary approaches. Selection should retain ordinary game win rates and not rely only on rally score with half-credit for draws.

Run `.venv/bin/python training/scripts/check_plain_model.py` from the repository root to reproduce the structure and native-inference check. The verifier also accepts explicit `--checkpoint`, `--onnx`, and `--output` paths for a later selected plain MLP.

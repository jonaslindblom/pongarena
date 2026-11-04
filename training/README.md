# Train a Pong v1 paddle

The game is unchanged. Start with the self-contained [v1 rules and reference package](v1/README.md). It includes a headless example, a standard-library Python port, four full deterministic match traces, and eleven boundary cases. The downloadable copy is `public/training/pong-v1.zip` (also in the arena's Field guide).

This directory adds an actual PyTorch experiment: a DAgger imitation warm start followed by PPO with historical self-play. The scripts run on CPU; no GPU, service, account or browser is required for training. This is a contemporary, well-established RL pipeline, not a claim of state-of-the-art benchmark performance.

## Setup and parity checks

Run commands from the repository root. The recorded run used Python 3.12 on Apple Silicon, one Torch CPU thread. Direct dependencies are pinned; `requirements-lock.txt` records the full environment used. A fresh compatible Python 3.12 environment is recommended.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r training/requirements.txt
python3 training/v1/check.py
.venv/bin/python training/scripts/check_env.py
```

The scalar reference must match all fixtures exactly, including float32 features. The batched simulator then verifies the same transition rules on 878 action-bearing fixture/boundary cases and 32,000 randomized transitions. Serves between rallies have no action. Do these checks before training a new port.

## Reproduce the recorded run

Use a fresh output directory; the script refuses to append to an existing metrics file.

```sh
.venv/bin/python training/scripts/train.py \
  --out training/runs/reproduction \
  --seed 7 --dagger-rounds 10 --updates 150 --eval-every 15
```

Defaults are 256 parallel environments, 128 rollout steps, 64-wide Tanh layers. The recorded run took about 73 seconds for training and periodic validation on this machine; installation and final ONNX evaluations are additional. Speed and exact floating-point results vary by hardware and library versions. A quick smoke run can use `--envs 16 --horizon 16 --dagger-rounds 1 --updates 2 --eval-every 1`.

Outputs include `imitation.pt`, `imitation.onnx`, `ppo.pt`, `ppo.onnx`, `last.pt`, `training-state.pt`, `metrics.jsonl`, `config.json`, and selection records. The two named models are selected on validation results, not simply the final iteration. `training-state.pt` retains actor, critic and optimizer for inspection; there is no exact-resume facility (RNG, environment and opponent-pool state are not checkpointed). Only load trusted PyTorch checkpoint files. ONNX actors are standalone browser submissions.

## What training does

- **DAgger** collects 65,536 decisions per round. A predictive attacker labels the learner's visited states. Teacher control falls from 75% to zero over the first four rounds. Six supervised epochs fit up/stay/down logits; a four-round replay window retains previous examples. Adam uses learning rate .001. The best of ten rounds becomes Apprentice.
- **PPO** starts from that selected actor. A separate `6→64→64→1` critic is used only during training. The opponent pool contains the three original ONNX models, a centering interceptor, a placement-seeking attacker, and historical copies of the learner. Opponents are assigned per rally; the bounded pool rotates between updates after five snapshots have accumulated.
- **Rewards** are +1 for scoring, −1 for conceding, and +.02 for a successful return. These are training choices. Arena scoring stays one point per miss and zero for a draw. No game speed, collisions, serve distribution or observation features change.
- **Optimization** uses clipped PPO (.2), clipped value loss, GAE (gamma .99, lambda .95), four epochs, minibatches of 2,048, normalized advantages, gradient norm .5, Adam eps 1e−5, a learning rate declining from .00025 toward .00005, and entropy coefficient .01. An imitation anchor starts at .05 and decays to zero. A mean approximate-KL threshold .03 stops further epochs. PPO ends episodes at each scored/drawn rally, with no bootstrap through terminal states; all five openings in each seeded match are used.
- **Selection** averages `wins + 0.5 * draws` per rally over the five fixed validation opponents. This diagnostic is separate from match points. The training policy samples categorical actions; all reported evaluations and browser matches use first-argmax.

The predictive scripts see only the six policy inputs. They recover integer geometry, predict wall bounces and discrete collision ticks, and the attacker enumerates reachable contact positions to maximize a simple travel-margin heuristic. They are deliberately strong references, not proofs of optimal play. Imitation is imperfect, and the learned policies remain much weaker than the attacker.

Training match seeds lie in [0,999,999,999]. Validation uses [1,000,000,000,1,000,000,023], both sides and five rallies (240 rallies/opponent/checkpoint). Final test seeds are [2,000,000,000,2,000,000,099] (1,000 rallies and 200 matches/opponent). These intervals do not overlap. Only one training seed was run; there are no multi-seed uncertainty estimates.

## Evaluate your exported model

```sh
.venv/bin/python training/scripts/evaluate.py \
  training/runs/reproduction/ppo.onnx \
  --seed-start 2000000000 --seeds 100 \
  --output training/runs/reproduction/test.json
```

This executes the fixed `[1,6]` exported ONNX actor, paired on both sides against every reference. The output records rally wins/losses/draws, match wins/losses/draws, decisive-rally median returns, mean duration, and early finishes. Choose fresh test seeds for further model development rather than repeatedly tuning to the published test set.

Recorded test results:

| Opponent | Apprentice rally W / L / D | Selfplay rally W / L / D | Selfplay match W / L / D |
| --- | --- | --- | --- |
| Tracker | 787 / 213 / 0 | 930 / 38 / 32 | 199 / 0 / 1 |
| Anticipator | 779 / 221 / 0 | 931 / 47 / 22 | 199 / 0 / 1 |
| Challenger | 780 / 215 / 5 | 943 / 51 / 6 | 199 / 1 / 0 |
| Interceptor script | 848 / 141 / 11 | 864 / 23 / 113 | 200 / 0 / 0 |
| Attacker script | 83 / 908 / 9 | 86 / 886 / 28 | 0 / 200 / 0 |

The best PPO checkpoint occurred at update45 / 2,129,920 cumulative decisions, despite running to 5,570,560 decisions. More training did not reliably improve this run. Both actors have 4,803 parameters, 4,672 MACs per decision, two hidden Tanh layers, and a raw linear three-score output. They are approximately20KiB each. Graphs are explicitly serialized from PyTorch Linear/Tanh layers into Gemm/Tanh ONNX opset18, IR10, so export does not introduce unsupported operators. Training machinery is excluded.

## Inspect curves and reproduce the report

Open the arena's Model library and follow **Learning curves & evaluation** on either trained card, or open `/training/report.html` on the local dev server. The report contains raw (unsmoothed) validation curves, imitation cross-entropy and PPO action entropy. Additional loss, approximate-KL and clipping diagnostics are in the JSONL log. Final-minibatch losses are diagnostics, not full-dataset averages. Original logs, selected PyTorch weights, source/config metadata, final tests and replay checks are committed in `training/results/seed7/`.

```sh
.venv/bin/python training/scripts/report.py
python3 training/scripts/package.py
.venv/bin/python training/scripts/verify_export.py
node --import tsx training/scripts/check_replays.ts
```

The export verifier compares 8,192 observations per policy with PyTorch (100% identical actions; maximum score error under 0.000007), then generates eight full native-ONNX match replays. The TypeScript checker reproduces all 7,210 frames against the actual unchanged browser engine. `npm test` also checks that the frozen `game.ts` remains byte-identical to the live rules.

To bring a new run into the arena, click **Import model** and select its `.onnx`. The shipped Apprentice and Selfplay files are included as downloadable presets for easy comparison. Model cards infer layer widths, activations and operator graphs from the actual ONNX bytes, including imports.

## Method references

- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [DAgger: A Reduction of Imitation Learning and Structured Prediction](https://arxiv.org/abs/1011.0686)
- [CleanRL's PPO implementation details](https://docs.cleanrl.dev/rl-algorithms/ppo/)

The implementation is specific to the small v1 arena and written here. No remote training service or external logging service is used.

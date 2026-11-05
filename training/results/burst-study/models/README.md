# Small learned paddles on the burst prototype

This is a bounded model-capacity and learnability experiment under `burst96-cost8-half`, exposed as the provisional `pong-burst-v1` contract. It does not establish the smallest possible network or an optimal tactical policy. The live rules decision belongs to the parent rule study.

Every export is a plain Tanh MLP with eight raw observations and five action scores. Geometric prediction is used by training teachers only; no fixed trajectory or feedback layers are embedded in the exported networks.

## Controlled size comparison

| Label | Architecture | Learned parameters | MACs / decision | Selected seed |
| --- | --- | ---: | ---: | ---: |
| Micro | 8 → 16 → 5 | 229 | 208 | 611 |
| Tiny | 8 → 16 → 16 → 5 | 501 | 464 | 617 |
| Small | 8 → 32 → 32 → 5 | 1,509 | 1,440 | 617 |
| Medium | 8 → 64 → 64 → 5 | 5,061 | 4,928 | 617 |
| Large | 8 → 128 → 128 → 5 | 18,309 | 18,048 | 611 |

All five architectures use two initialization seeds, 611 and 617. Training collects a common pooled DAgger dataset from their mixed teacher/student trajectories, against an attacker, defender, interceptor and Tracker. Both player perspectives are labeled by the attacking teacher. Every network receives exactly the same examples, minibatch order and optimization-pass budget. Larger networks necessarily consume more arithmetic; this is equal-data training, not an equal-FLOP contest.

The screen ran 24 rounds × 128 environments × 128 ticks = 393,216 active environment steps. Both sides contribute 786,432 game observations, plus 196,608 broad synthetic inputs, for **983,040 labels per model**. These are not all unique states. Synthetic observations cover input ranges; some are unreachable. Replay retains four rounds; four supervised epochs per round, minibatch 2,048, Adam learning rate 0.001 decreasing toward 0.0004, gradient clipping at 1. Teacher-action probability declines to 5%. Exact runtime/configuration/source hashes are retained in `screen/`.

Every six rounds, eight development seeds evaluate Tracker, resource-aware interceptor and attacking teacher. Mean point margin chooses each checkpoint, then the better development seed per architecture selects its release. Seed order resolves an exact tie. **Every choice preceded the final test.** All ten selected-per-run networks remain in the raw results, including weaker initializations.

## Fresh final tests

Each pair uses 32 seeds × five serves × both side assignments = 320 rallies. These tests run frozen PyTorch checkpoints in the verified vector environment. ONNX parity is checked separately. Wins/losses/draws are from the selected network's perspective.

| Network | Tracker W / L / D | Attacker W / L / D | Self-mirror median decisive returns | Self-mirror draws |
| --- | --- | --- | ---: | ---: |
| Micro | 115 / 205 / 0 | 0 / 320 / 0 | 1 | 0% |
| Tiny | 223 / 97 / 0 | 0 / 320 / 0 | 3 | 0% |
| Small | 228 / 92 / 0 | 0 / 320 / 0 | 1 | 18.1% |
| Medium | 270 / 50 / 0 | 0 / 320 / 0 | 6.5 | 52.5% |
| Large | 298 / 19 / 3 | 0 / 320 / 0 | 21 | 78.1% |

The smallest demonstrated useful baseline in this recipe is **501 parameters**: both Tiny initializations beat Tracker, and the selected Tiny beats Micro 238–82. Small beats Tiny 219–101, Medium beats Small 167–69–84, and Large beats Medium 189–28–103 on the same paired test seeds. A network's size is not a reliable human difficulty label: Small's self-mirror scored rallies are short, while larger networks produce many draws. Human enjoyment has not been established by these numerical results.

These first networks mainly improve prediction and survival. All selected size-study networks lose every final rally to the attacking teacher. Altering an energy input can change their actions, but such sensitivity is not proof of useful resource management. Own-energy counterfactuals show only partial agreement with the teacher, and the teacher itself rarely changes decisions when opponent energy changes. At constant outgoing speed, part of its energy-dependent reach estimate subtracts the same quantity from every candidate placement. It does not supply a dependable multi-exchange strategy target.

`release/confirmation.json` retains all ten network evaluations against five reference opponents and themselves, plus selected crossplay and burst-action ablations. `release/manifest.json` freezes checkpoint/export hashes and selection before these tests.

## Separate burst-focused follow-up

One bounded follow-up continued the selected Large network with six epochs, six rounds of replay, and oversampling of actual game observations where the teacher requested burst. These extra observations come from learner/teacher trajectories; they are not synthetic battery edits. There is no PPO stage. Optimizer/replay/teacher mixing restart, and a fresh development block chooses the checkpoint. This experiment is **not** included in the equal-budget size comparison.

The run executed 60 additional rounds, 983,040 active environment steps, and 3,440,640 labels including repeated burst samples. Round 48 was selected on development results before another fresh final block. The exported network, named **Focus**, still has 18,309 learned parameters. Its selected weights incorporate 48 continuation rounds; later rounds were executed but not selected.

| Same fresh 320-rally test set | Wins / losses / draws |
| --- | --- |
| Focus vs attacking teacher | 44 / 255 / 21 |
| Original Large vs attacking teacher | 0 / 320 / 0 |
| Focus, burst actions mapped to ordinary movement, vs attacker | 0 / 320 / 0 |
| Focus vs original Large | 25 / 197 / 98 |
| Focus vs safer defender | 0 / 217 / 103 |
| Original Large vs safer defender | 0 / 49 / 271 |
| Focus vs itself | 114 / 114 / 92 |

This is concrete evidence that learned burst use can improve outcomes against one strong opponent. It is also a substantial tradeoff: Focus loses to the original Large and worsens against safer defense. It is a specialist, not a universal upgrade. Its self-mirror scored rallies have a median 22 returns and 28.8% draws.

The rule study supplies a specific pressure sequence ending at tick 883. The defender has energy 3 and cannot reach the next ball even with the best legal actions. Holding physical coordinates fixed and replacing energy with 96 creates a catchable counterfactual. Focus catches that full-energy shot; all five initial size-study models miss it. All models fail in the recorded depleted state, as required. The full-energy state is a legal local counterfactual, not a claim that the recorded action prefix produced it.

An earlier branch at tick 727 admits a placement counter: a scripted choice of contact center 19 avoids the eventual loss. Focus instead contacts at 16 and still loses. Thus rescue timing is learned, while the complete multi-exchange placement counter remains unlearned. `followup-release/tactic-probes.json` retains these continuations, and `followup-release/confirmation.json` retains the fresh paired outcomes.

## Seeds and verification

| Purpose | Seed range / generation |
| --- | --- |
| Training games | `[0, 1_000_000_000)`, collection RNG 619; follow-up 643 |
| Size development | NumPy RNG 20261211, eight seeds in `[2_810_000_000, 2_820_000_000)` |
| Size final | NumPy RNG 20261221, 32 seeds in `[2_900_000_000, 3_000_000_000)` |
| Size native replays | 3,010,000,007 and 3,010,000,043 |
| Follow-up development | NumPy RNG 20261224, 12 seeds in `[3_100_000_000, 3_110_000_000)` |
| Follow-up final | NumPy RNG 20261225, 32 seeds in `[3_300_000_000, 3_400_000_000)` |
| Follow-up native replays | 3,410,000,007 and 3,410,000,043 |

These ranges are disjoint from each other and the current rule-screen/confirmation ranges. The final blocks are consumed and must not select future improvements. Numeric export probes use their own seeded generators and test implementation, not model strength.

The five original exports agree with native ONNX on all 23,376 numeric/recorded-state probes, with maximum score difference 9.54e-6. Six complete scalar-reference/native-ONNX games contain 17,383 frames and 34,718 decisions. Focus agrees on all 8,192 additional numeric probes, with maximum score difference 1.91e-5; two complete native games contain 11,063 frames.

The integrated browser reproduced all eight games: **28,446 frames and 56,828 decisions**, with zero action or state disagreements, including energy. Maximum browser/native score difference was 1.53e-5. The [browser verification record](../browser-verification.json) also covers local ranking repeatability, saved-result restoration/export, contract switching, import races, ordinary HTTP LAN use, and emulated mobile controls. Physical-device performance and human enjoyment still require playtesting.

The scalar-reference hash changed between these two fixture-generation runs only because the rule identifier constant was added; physics is unchanged. The exact hash used for each is retained in its verification record. Canonical `burst_rules.py` behavior remained unchanged through training and evaluation.

## Reproduce

From the repository root with the pinned Python environment, use fresh output directories:

```sh
.venv/bin/python training/scripts/train_size_study.py \
  --out training/results/burst-study/models/reproduction-screen \
  --rules burst96-cost8-half --rounds 24 --envs 128 --horizon 128 \
  --synthetic 8192 --dev-seeds 8 --epochs 4 --eval-every 6
```

The exact historical trainer and rule source are in `screen/training-source.py` and `screen/rules-source.py`, matching the recorded hashes. The current trainer adds optional continuation controls without changing the default screen recipe. Set `PYTHONPATH=training/scripts` to execute a historical trainer snapshot directly.

The retained freeze/evaluation commands were:

```sh
.venv/bin/python training/scripts/evaluate_size_study.py --freeze
.venv/bin/python training/scripts/evaluate_size_study.py
```

They intentionally refuse to overwrite retained results. New experiments require new final seeds; do not remove the retained manifest to reuse them for selection.

The bounded follow-up command was:

```sh
.venv/bin/python training/scripts/train_size_study.py \
  --out training/results/burst-study/models/followup \
  --rules burst96-cost8-half --architectures large --init-seeds 641 --seed 643 \
  --rounds 60 --envs 128 --horizon 128 --synthetic 8192 --dev-seeds 12 \
  --epochs 6 --eval-every 12 --replay-rounds 6 \
  --initial-dir training/results/burst-study/models/release --focus-burst \
  --dev-seed-start 3100000000 --dev-rng 20261224
.venv/bin/python training/scripts/evaluate_size_followup.py
```

The following verification/report commands can be rerun without retraining or selecting new models:

```sh
.venv/bin/python training/scripts/verify_size_models.py
.venv/bin/python training/scripts/report_size_study.py
```

Public report: `/training/burst-models.html`. Raw plotted values, both initialization results and final summaries: `/training/burst-model-metrics.json`. The public native replay files are `/training/burst-onnx-replays.json` and `/training/burst-focus-replays.json`. The small `smoke/` run checks plumbing only and is excluded from model comparisons.

# Fully learned Pong v4 baselines

**Pacer v4** and **Apprentice v4** use raw six-input observations and a `6 → 128 → 128 → 3` Tanh MLP. Each export contains three learned Gemm layers and two Tanh nodes: 17,795 parameters, 17,536 multiply-accumulates and 71,658 bytes. There are no fixed trajectory features, geometric calculations or scripted decisions in inference.

Pacer is the continuation run's selected round-36 checkpoint. Apprentice is the initial run's round-six checkpoint, retained as an easier opponent. Both were frozen before the separate final model evaluation. Neither consistently beats the strongest adaptive scripts.

## Training method

`training/scripts/train_pace.py` implements supervised DAgger. The stage-aware geometric attacker from `pace_study.py` supplies labels for states encountered by the learner. The policy executes a mixture of teacher and student actions; teacher probability declines to 5% in each run. Opponents mix adaptive attack, safer defense, interception, old Contender, Tracker and Anticipator. Both players act on the same pre-tick state.

Each round collects 256 environments × 256 ticks = 65,536 active environment steps, and adds 16,384 synthetic component-valid observations labeled by the teacher. Synthetic observations cover integer positions and velocities; they are not asserted to be reachable game states. Training uses cross entropy, Adam, batch size 2,048, a learning rate decreasing from 0.001 toward 0.0004, and gradient norm clipping at 1. There is no PPO stage.

- Initial run: seed 411, 24 rounds, four epochs per round, replay from the latest four rounds.
- Continuation: seed 412, starts at the initial selected weights, 36 rounds, six epochs per round, latest eight rounds of replay. Optimizer and replay reset, and teacher mixing restarts.
- Total: 60 rounds, **3,932,160 active environment steps**, plus 983,040 supplemental labeled synthetic observations. Training repeatedly samples the replay buffer; these are not unique example counts.
- Every three rounds, deterministic first-argmax evaluation uses 32 development seeds and four opponents: adaptive attacker, defender, old Contender and Tracker. Mean point margin selects the strongest checkpoint.

The continuation improves development mean margin from about −0.052 to +0.093. This mean includes weaker opponents; it does not imply positive margin against either strong script. Curves show the underlying opponent results separately. Retained loss measurements are epoch averages at the logged evaluation rounds, not a separate validation loss.

## Untouched final evaluation

Each pairing uses 64 seeds × five serves × both side assignments = 640 rallies. These comparisons run the frozen PyTorch checkpoints in the verified vector environment. ONNX inference is checked separately using numeric probes, captured actions and complete native/browser matches; the table is not an all-ONNX evaluation. Model selection and the easier Apprentice checkpoint preceded this final set. `confirmation.json` retains every outcome; `manifest.json` freezes source/checkpoint/export hashes and final seeds before evaluation.

| Model | Opponent | Wins / losses / draws | Median returns in scored rallies |
| --- | --- | --- | ---: |
| Pacer | Pacer | 307 / 307 / 26 | 11 |
| Pacer | Apprentice | 397 / 186 / 57 | 8 |
| Pacer | Adaptive attacker | 163 / 439 / 38 | 9 |
| Pacer | Adaptive defender | 77 / 441 / 122 | 8 |
| Pacer | Old Contender | 512 / 112 / 16 | 8.5 |
| Pacer | Old Selfplay v2 | 490 / 113 / 37 | 9 |
| Pacer | Tracker | 615 / 25 / 0 | 2 |
| Pacer | Anticipator | 624 / 16 / 0 | 3 |
| Apprentice | Tracker | 517 / 123 / 0 | 2 |
| Apprentice | Anticipator | 394 / 246 / 0 | 3 |
| Apprentice | Old Contender | 299 / 300 / 41 | 5 |

All comparisons run under v4 physics. Older weights were trained under earlier rules, so these are transfer comparisons. Pacer's wins do not establish its standing on their original leaderboards. Strong scripts remain stronger; this is an initial learned baseline with a useful range of opponents. Mirror side-paired aggregate scores necessarily tie, but individual rallies can score.

## Seed separation

| Phase | Generation | Numeric range | Count |
| --- | --- | --- | ---: |
| Training games | Arena RNG seeded 411 / 412 | `[0, 1_000_000_000)` | Sampled at match reset |
| Model development | NumPy RNG 20261130, without replacement | `[1_900_000_000, 2_000_000_000)` | 32 |
| Rule screen | NumPy RNG 20261120, without replacement | `[2_300_000_000, 2_400_000_000)` | 32 |
| Rule confirmation | NumPy RNG 20261121, without replacement | `[2_400_000_000, 2_500_000_000)` | 64 |
| Model confirmation | NumPy RNG 20261203, without replacement | `[2_600_000_000, 2_700_000_000)` | 64 |
| Native ONNX parity | Explicit seeds | `2_710_000_007`, `2_710_000_043` | 2 |

Exact development lists are in run configurations, the final list in `manifest.json`, and rule lists in the parent study artifacts. `seed-check.json` verifies no final model seed overlaps development, either rule phase, training's interval, or the 2,881 seeds extracted from 195 pre-existing result JSON files outside this study. These confirmation sets are now consumed: future tuning needs fresh final seeds. Numeric probes use RNG 47019 and check implementation, not model selection.

## Artifacts and verification

- `dagger-s411/`, `dagger-continuation-s412/`: configurations, raw `metrics.jsonl`, selection, completion counts, checkpoints and original training source snapshots.
- `candidate/pacer/`, `candidate/apprentice/`: frozen PyTorch and ONNX exports; metadata declares `rules=pong-v4` and `fixed_features=none`.
- `learning-curves.png`, `.svg`: unsmoothed observed training/development curves.
- `verification.json`: ONNX checking, arena budget validation, 8,192 numeric probes per model, and agreement on all 6,401 captured game actions.
- `onnx-replays.json`: six native-ONNX matches, 10,904 frames, using independent `training/v4/reference.py`. Public copy: `public/training/pace-onnx-replays.json`.
- Public report: `public/training/pace-study.html`; raw plotted values and final summaries: `public/training/pace-metrics.json`.

Maximum native ONNX score error versus PyTorch on numeric probes was `1.33514404296875e-5` for Pacer and `4.76837158203125e-6` for Apprentice, with zero action disagreements. The verifier checks frozen hashes before and after running. Browser WASM parity also passed all six matches, 10,904 frames and 21,760 decisions: no action or state mismatches, maximum score difference `1.1444091796875e-5`. See `../browser-verification.json` for integration evidence and device-test limits.

## Reproduce

Run from the repository root with the project Python environment. The current trainer reproduces the method in fresh output directories:

```sh
.venv/bin/python training/scripts/train_pace.py \
  --out training/results/pace-study/reproduction/initial --seed 411 --rounds 24
.venv/bin/python training/scripts/train_pace.py \
  --out training/results/pace-study/reproduction/continuation --seed 412 \
  --rounds 36 --epochs 6 --replay-rounds 8 \
  --initial training/results/pace-study/reproduction/initial/best.pt
```

Original output directories were `training/results/pace-study/training/dagger-s411` and `dagger-continuation-s412`. Exact historical `training-source.py` snapshots match the SHA-256 recorded in each run's configuration. They differ because continuation support was added between runs; the canonical script now includes that support. To execute an archived source directly, set `PYTHONPATH=training/scripts` and supply a fresh `--out`. Single-thread CPU execution and fixed seeds control randomness; bitwise reproducibility across PyTorch/BLAS/platform versions is not guaranteed.

The retained freeze and evaluation commands were:

```sh
.venv/bin/python training/scripts/eval_pace_models.py --freeze
.venv/bin/python training/scripts/eval_pace_models.py
```

Those intentionally refuse to overwrite an existing manifest/final result. New weights need a separate run and fresh reserved seeds; do not delete retained evidence. Verification/report commands are safe to rerun:

```sh
.venv/bin/python training/scripts/verify_pace_models.py
.venv/bin/python training/scripts/pace_report.py
```

See the parent `README.md` for rule screening, exact transition verification and study limitations. Human keyboard/touch enjoyment has not been established by automated training or a delayed-observation controller.

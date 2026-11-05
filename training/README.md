# Train a Pong v3 paddle

Pong v3 keeps all v2 physics and the six-input/three-score interface. The rally deadline is now **1,600 ticks**. Fixed deterministic model pairs also draw when an exact post-contact ball/paddle state repeats. The arena labels loop and timeout draws separately; neither awards a point. Human matches use the full deadline.

Start with the self-contained [v3 contract](v3/README.md), also downloadable as `/training/pong-v3.zip`. It includes the engine, headless example, dependency-free Python reference, fixtures and checker. [V1](v1/README.md) and [v2](v2/README.md) remain frozen.

The latest [fully learned investigation](results/fully-learned/README.md) introduces **Contender**, a plain `6→64→64→3` Tanh actor trained through PPO and direct weight search. It passes three fresh test blocks against original v2 and the unchanged attacker, including majority ordinary five-rally game wins in each block. Its curves and final results are at `/training/fully-learned.html`. There are no fixed geometry or paddle-feedback layers in this export.

The [earlier fixed-rules investigation](results/training-improvement/README.md) diagnoses the v3 regression and records **Strategist**, a hybrid with fixed trajectory features and a learned target head. It does not satisfy the fully learned requirement. Its separate curves and results remain at `/training/improvement.html`; the original v3 recipe below also remains historical evidence.

## Why these rules

The [rule study](results/rule-study/README.md) compared 800/1,600/3,200-tick limits and ten physics configurations on matched serves. Longer limits let some genuinely unfinished contests score; exact cycles otherwise consume time without changing the outcome. The study favored retaining paddle speed 2 and horizontal shot speed 5–13. Its original report is `/training/rule-study.html`.

Loop termination is valid only when both policies are fixed, deterministic and stateless. A sampled training policy, a human, changing weights or untracked recurrent state can choose differently after a repeated physical position. Consequently **DAgger and PPO training disable loop exits**, while deterministic validation, final evaluation and browser model matches enable them. All modes use the 1,600-tick maximum. This is explicit in run configurations and the standalone specification.

## Setup and verification

Run from the repository root. Recorded training used Python 3.12 on Apple Silicon, with one Torch CPU thread per run. Dependencies are pinned in `requirements.txt`; `requirements-lock.txt` records the full environment.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r training/requirements.txt
python3 training/v1/check.py
python3 training/v2/check.py
python3 training/v3/check.py
.venv/bin/python training/scripts/check_env.py
```

V3 fixtures contain 2,446 full-match transitions, 1,765 synthetic trace transitions, 18 boundary cases, five detector histories and a restart reset check. They include deterministic loop draws and the same exchange continuing to timeout with detection disabled. The copied TypeScript engine is byte-identical to the live engine. The vector simulator passes 978 isolated fixture transitions, 32,000 randomized transitions, all 4,187 published action transitions, and 8,326 deterministic transitions against the scalar reference.

The headless example accepts `--no-loops`. For seed 24, both modes finish 1–1: 742 ticks with loops, 4,870 without. At a deadline, points take precedence; a repeated contact is labelled a loop ahead of a timeout. Instantiate/reset the detector on a new match or controller change. Its history resets at a new rally.

## Train and select

Three runs, seeds 7/17/29, each executed 10 DAgger rounds and 250 PPO updates: 8,847,360 decisions per run, 26,542,080 total. Concurrent runs completed in about 121 seconds each on this machine, excluding release evaluation. Architecture, reward, discount and other training hyperparameters were retained from the v2 recipe to limit confounding changes.

```sh
for seed in 7 17 29; do
  .venv/bin/python training/scripts/train.py \
    --out training/runs/v3-seed$seed --seed $seed \
    --dagger-rounds 10 --updates 250 --eval-every 25
done
.venv/bin/python training/scripts/release.py
```

This example loop runs sequentially; recorded runs used the same commands concurrently. Use fresh output directories. A smoke run can use `--envs 16 --horizon 16 --dagger-rounds 1 --updates 2 --eval-every 1`. There is no exact-resume facility: the inspection checkpoint does not retain every RNG/environment/opponent-pool state. Load only trusted PyTorch checkpoints.

| Seed | Selected PPO update | Validation score |
| --- | ---: | ---: |
| 7 | 225 | .78292 |
| 17 | 100 | .79125 |
| **29 · released** | **100** | **.81125** |

Selection maximizes mean `(rally wins + 0.5 × draws) / rallies` over five reference opponents on validation seeds only. Final tests do not select checkpoints. Apprentice is the selected PPO run's own best imitation warm start. Release retains each run's selected checkpoint pair, configuration, console log and learning log, plus commands and a selection manifest under `results/v3/`.

Training seeds are [0,999,999,999], fresh validation seeds [1,100,000,000,1,100,000,023], and fresh final tests [2,200,000,000,2,200,000,099]. Both sides and all five serves are evaluated. These sets differ from v2 validation/tests and the rule-study sets. V2's recorded validation score therefore is not directly comparable to the v3 validation score.

## Training method

Each actor is `6→64→64→3`, Tanh hidden layers, 4,803 parameters and 4,672 MACs/decision, approximately 20 KiB ONNX. A separate `6→64→64→1` critic is training-only. Export uses supported Gemm/Tanh operators, opset18/IR10, embedded float32 weights and `rules=pong-v3` metadata.

- DAgger collects 65,536 decisions per round from a mixture of teacher/learner actions. A predictive attacker labels learner-visited states. Teacher probability declines to zero over four rounds; six supervised epochs fit a four-round replay window, minibatch 2048, Adam .001.
- PPO uses 256 environments × 128 steps, four epochs, minibatch 2048, gamma .99, GAE lambda .95, clip .2, clipped value loss, gradient norm .5, Adam eps 1e-5, learning rate .00025→.00005, entropy .01 and approximate-KL early stop .03. A decaying imitation anchor starts at .05.
- Rewards remain +1 for scoring, −1 for conceding, and +.02 per return. These are training choices, not game rules. Return rewards and discounting can influence defensive behavior; this run does not test alternative rewards.
- Opponents include Tracker, Anticipator, Challenger, a predictive interceptor, a greedy attacking script and up to five historical actor snapshots. Episodes end per rally; all five openings are visited. Sampled training episodes end on points or the 1,600 deadline, with no bootstrap through termination. Validation uses frozen first-argmax policies and exact loop detection.
- Both scripts use only the six exposed observations. The attacker predicts flights and chooses a reachable contact placement by a greedy reachability margin; it is a strong reference, not an optimal-play proof or multi-exchange planner.

## Final exported-model results

Each row is 1,000 rallies / 200 five-rally games on identical held-out serves, both sides, using the actual ONNX files. Draws are explicitly split by termination reason.

| Opponent | Apprentice W / L / D | Selfplay W / L / D | Selfplay loop / timeout draws |
| --- | --- | --- | --- |
| Tracker | 807 /180 /13 | 951 /18 /31 | 31 /0 |
| Anticipator | 821 /156 /23 | 937 /33 /30 | 30 /0 |
| Challenger | 833 /160 /7 | 935 /23 /42 | 42 /0 |
| Interceptor | 320 /257 /423 | 445 /30 /525 | 363 /162 |
| Attacker | 60 /876 /64 | 203 /350 /447 | 447 /0 |

The new model is **not universally stronger**. V2 weights were reevaluated under the same v3 rules and exact same final-test serves, separating weight differences from the new deadline:

| Opponent | Previous Selfplay v2 W / L / D | New Selfplay v3 W / L / D |
| --- | --- | --- |
| Interceptor | 856 /73 /71 | 445 /30 /525 |
| Attacker | 111 /420 /469 | 203 /350 /447 |

The new policy wins more against the attacker but draws much more against the interceptor. Its mean `(wins + half draws)` score over the five reference opponents is 1.44 percentage points lower than the old weights on this test set. Head-to-head, new Selfplay scores **193 wins /428 losses /379 draws** against old Selfplay, with 229 loop draws and 150 timeouts. The old Selfplay remains selectable in the arena as **Selfplay v2**. Test results are reported, not used to retroactively select a different training run.

New Selfplay vs itself has 211/211/578 W/L/D, all 578 draws proven loops, and median 9 returns in scored rallies. Against its new Apprentice it scores 552/314/134, all 134 draws loops, with median 8 decisive returns. Longer scored rallies and draw rates are reported separately; neither proves universal strength.

Evaluation also reports paired ten-rally contests by aggregating the two side assignments for each seed. The browser still plays five-rally matches. Identical-policy paired contests necessarily tie. Reuse many seeds and varied opponents when comparing strength.

```sh
.venv/bin/python training/scripts/evaluate.py path/to/model.onnx \
  --seed-start 2200000000 --seeds 100 --output /tmp/model-test.json
```

Use fresh seeds for future development rather than tuning repeatedly against this published test set.

## Reports, exports and browser checks

The Model library links to `/training/report.html`: all three learning curves, final tests, loop/timeout counts and comparisons against old weights. Raw curves are unsmoothed. Final-minibatch losses are diagnostics, not full-dataset averages. Cards derive their architecture from the actual uploaded graph. Both new actors are available locally as `public/models/selfplay-v3.onnx` and `apprentice-v3.onnx`.

```sh
.venv/bin/python training/scripts/verify_export.py
node --import tsx training/scripts/check_replays.ts
.venv/bin/python training/scripts/report.py
python3 training/scripts/package.py
npm test
npm run build
```

Both actors agree with native ONNX on every action across 8,192 observations each. Ten complete ONNX matches reproduce exactly in live TypeScript: 17,536 frames, including 8 loop draws and one actual 1,600-tick model timeout. Browser WASM inference agrees on all 34,992 decisions and advances the same states; maximum score difference is 9.54e−6. Verification records are in `results/v3/`.

## Archives

V1/v2 contracts, ZIPs, original weights and original result directories are preserved. Their visual reports are `/training/v1/report.html` and `/training/v2/report.html`. V2 source and original documentation are at Git commit `517a730`; use a separate checkout to reproduce that experiment. The pre-v3 rule-study source checkpoint is `0b0cba1`; its historical checker continues to use the explicit v2 reference. Current training and release scripts target v3.

Methods: [PPO](https://arxiv.org/abs/1707.06347), [DAgger](https://arxiv.org/abs/1011.0686), [CleanRL PPO notes](https://docs.cleanrl.dev/rl-algorithms/ppo/). All training, inference and logging are local.

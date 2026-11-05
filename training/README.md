# Train a Pong v2 paddle

The current arena uses **Pong v2**, with maximum horizontal return speed 13 (previously 15). This is the only physics change. The six-input/three-score interface, paddle speed, aiming, serves, scoring and timeout are unchanged.

Start with the self-contained [v2 rules and reference](v2/README.md). It includes a headless example, standard-library Python port, four complete match traces and eleven boundary cases. The downloadable copy is `public/training/pong-v2.zip`, also linked in the Field guide. [V1](v1/README.md) remains frozen, including its original weights, test results and report.

## Why cap 13

We tested caps 15, 14, 13, 12, 11, 10 with identical serves and scripts that know each candidate's physics. We then confirmed caps 15, 13, 12 on an independent set of 2,000 rallies per matchup.

With two identical attackers, cap 13 changes median decisive returns from 1 to 5, first-receiver wins from 93.4% to 50.9%, and early finishes from 57.6% to 7.8%. All those v2 attacker-mirror rallies scored. Cap 12 instead drew 93.8%; caps 10 and 11 drew 100% in the initial scan. The attacker still scores 62.4% against the centering defender at cap 13.

See [balance evidence and reproduction commands](results/balance-v2/README.md). The fastest crossing is now 26 ticks (~0.87s), versus 22 (~0.73s). Human comfort still requires playtesting; the delayed-controller benchmark is a sensitivity check, not a human win-rate estimate. No hidden difficulty state or human-only modifier exists.

The later [timeout and rule study](results/rule-study/README.md) compares 800/1,600/3,200-tick limits, proves deterministic loops, and tests paddle/shot changes on matched serves. Its [visual report](../public/training/rule-study.html) is served at `/training/rule-study.html`. It recommends retaining v2 physics and considering a longer budget with model-loop termination; these are evaluated proposals, not changes to the current contract.

## Setup and parity

Run from the repository root. The recorded training used Python 3.12 on Apple Silicon and one Torch CPU thread per run. Direct dependencies are pinned; `requirements-lock.txt` records the original full environment.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r training/requirements.txt
python3 training/v1/check.py
python3 training/v2/check.py
.venv/bin/python training/scripts/check_env.py
```

V1 verifies 883 exact transitions; v2 verifies 980; each also checks 11 boundary cases and both float32 observations. The vector simulator checks 975 v2 action-bearing fixture/boundary cases plus 32,000 randomized transitions. Between-rally serve transitions contain no action. Check parity before training a port.

## Train and select

Three independent seeds each ran 10 DAgger rounds and 250 PPO updates, for 8,847,360 decisions per run (26,542,080 total). They completed in about 120 seconds while running concurrently on this machine, excluding dependency setup and final evaluations. Hardware and library versions affect timings and floating-point results.

```sh
for seed in 7 17 29; do
  .venv/bin/python training/scripts/train.py \
    --out training/runs/v2-seed$seed --seed $seed \
    --dagger-rounds 10 --updates 250 --eval-every 25
done
.venv/bin/python training/scripts/release.py
```

The loop above runs sequentially. The recorded jobs used the same commands concurrently. Always use fresh output directories; the trainer refuses to overwrite a metrics log. A smoke run can use `--envs 16 --horizon 16 --dagger-rounds 1 --updates 2 --eval-every 1` with a separate output directory. `release.py --runs path1 path2 path3 --out path` accepts custom run locations.

Training outputs include selected `imitation.pt/.onnx`, `ppo.pt/.onnx`, `last.pt`, `training-state.pt`, metrics, configuration and selection records. There is no exact-resume facility: the inspection checkpoint retains actor, critic and optimizer, but not the full RNG/environment/opponent-pool state. Load only trusted PyTorch checkpoints.

Release selects the highest validation score across runs **before** testing, then publishes `public/models/selfplay-v2.onnx` and that same run's imitation warm start as `apprentice-v2.onnx`. It retains every run's logs, configuration and selected PyTorch checkpoints under `training/results/v2/seed*/`. It evaluates the actual exported ONNX actors and both learned matchups. The root `training/results/v2/` contains the selected run plus final tests and selection manifest.

| Seed | Best PPO update | Validation score |
| --- | ---: | ---: |
| 7 | 125 | .77625 |
| **17 · selected** | **250** | **.80125** |
| 29 | 175 | .77792 |

The score is the mean of `(rally wins + 0.5 * draws) / rallies` across five reference opponents. It is a training diagnostic; match scoring remains one point per miss and zero for a timed-out rally. These three runs show variation, but the selected run's test results are not a multi-seed average or confidence interval.

## Training method

Each actor is `6→64→64→3`, Tanh hidden layers, 4,803 parameters and 4,672 MACs/decision, approximately 20 KiB ONNX. It outputs raw up/stay/down scores. A separate `6→64→64→1` critic is training-only.

- DAgger collects 65,536 decisions per round, labels states with a predictive attacker, and gradually removes teacher control over the first four rounds. Six supervised epochs fit each round; the replay window retains four rounds. Adam learning rate is .001.
- PPO uses 256 environments × 128 steps, four optimization epochs, 2,048-item minibatches, clipping .2, clipped value loss, GAE (gamma .99, lambda .95), normalized advantages, gradient norm .5, Adam eps 1e−5, learning rate .00025 declining toward .00005, and entropy coefficient .01. A mean approximate-KL threshold .03 stops further epochs.
- Rewards are +1 for scoring, −1 for conceding and +.02 per return. An imitation anchor starts at .05 and decays to zero. These are training choices outside the game rules.
- Opponents include Tracker, Anticipator, Challenger, a predictive interceptor, a placement-seeking attacker and up to five historical learner snapshots. Opponents are assigned per rally; the bounded historical pool rotates between updates. Each scored/drawn rally terminates the RL episode, with no bootstrap through terminal states; all five openings of each seeded match are used.
- The scripts use only the six policy inputs. They recover integer geometry, calculate wall reflections and discrete flight times; the attacker enumerates reachable contact positions using v2's actual shot formula. These are strong references, not optimal-play proofs.
- Training samples actions. Validation, final evaluation and the arena use deterministic first-argmax. Export serializes Linear/Tanh weights directly into Gemm/Tanh ONNX opset18, IR10, avoiding auxiliary operators outside the small submission contract. Critics and optimizers stay outside the files.

Training seeds are [0,999,999,999]. Validation uses [1,000,000,000,1,000,000,023], both sides and all five rallies (240 rallies/opponent/checkpoint). Balance studies use the 3,000,000,000 and 3,100,000,000 intervals. Final v2 evaluation uses [2,100,000,000,2,100,000,099]. The actual test range is recorded by `release.py` in the release manifest/configuration; it does not affect training. These sets are disjoint.

## Final exported-model results

Each row is 1,000 rallies / 200 five-rally matches, both sides, on the same unseen serves.

| Opponent | Apprentice W / L / D | Selfplay W / L / D | Selfplay matches W / L / D |
| --- | --- | --- | --- |
| Tracker | 849 / 144 / 7 | 898 / 18 / 84 | 200 / 0 / 0 |
| Anticipator | 789 / 139 / 72 | 912 / 39 / 49 | 199 / 0 / 1 |
| Challenger | 825 / 147 / 28 | 893 / 52 / 55 | 199 / 0 / 1 |
| Interceptor | 317 / 227 / 456 | 741 / 53 / 206 | 199 / 0 / 1 |
| Attacker | 65 / 752 / 183 | 111 / 359 / 530 | 22 / 141 / 37 |

The attacker remains stronger. Survival improved, with many draws; those are not wins. Against itself, Selfplay has median 7 returns in scored rallies and 41% timeouts. Against Apprentice, Selfplay wins 59.4%, draws 24%, and decisive rallies have median 5 returns. Compared with a simple defensive policy, trained agents offer a useful challenge, but no claim of optimal or state-of-the-art play is made.

To evaluate another submission:

```sh
.venv/bin/python training/scripts/evaluate.py path/to/model.onnx \
  --seed-start 2100000000 --seeds 100 --output /tmp/model-test.json
```

Use fresh seeds for future model development rather than repeatedly tuning to the published test set.

## Curves, export checks and the arena

The Model library links to `/training/report.html`, containing the balance comparison, all three training curves, raw selected-run opponent win rates, imitation loss, PPO entropy and final test tables. Additional policy/value losses, KL and clipping diagnostics are in the JSONL records. Final-minibatch losses are diagnostics, not dataset averages.

```sh
.venv/bin/python training/scripts/verify_export.py
node --import tsx training/scripts/check_replays.ts
.venv/bin/python training/scripts/report.py
python3 training/scripts/package.py
npm test
npm run build
```

The export check compares 8,192 observations per policy with PyTorch (identical actions; maximum score error below .000008). Eight complete native-ONNX matches reproduce exactly in the live TypeScript engine: 13,975 frames. App tests verify that the frozen v2 engine matches the live source and that v2 changes only horizontal contact speed relative to v1.

The arena starts with the new Selfplay vs Apprentice matchup. Import any compatible ONNX using **Import model**. Architecture cards derive widths, activations and operator graphs from the actual files. V1 inputs/outputs remain compatible, so existing imports still run; their performance must be re-evaluated under v2.

## V1 archive

`training/v1/`, `public/training/pong-v1.zip`, original `public/models/selfplay.onnx` / `apprentice.onnx`, and `training/results/seed7/` remain intact. The archived visual report is `/training/v1/report.html`. Original v1 training scripts and README are available at Git commit `a89c818`; use a separate checkout of that commit to reproduce the original experiment. Current scripts target v2.

Methods: [PPO](https://arxiv.org/abs/1707.06347), [DAgger](https://arxiv.org/abs/1011.0686), [CleanRL PPO notes](https://docs.cleanrl.dev/rl-algorithms/ppo/). The implementation is local and purpose-built for this arena; no remote training or logging service is used.

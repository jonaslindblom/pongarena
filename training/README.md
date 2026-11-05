# Train a Rally Lab paddle

The active arena uses **Classic v4** (`pong-v4`), with Pacer vs Apprentice. Start with its self-contained [six-input/three-action game specification](v4/README.md), downloadable as `/training/pong-v4.zip`. The immutable package retains its historical general-graph export section; the current dense-only [competition contract](COMPETITION.md) takes precedence for active submissions. It includes the engine, a headless example, dependency-free Python reference, fixtures and checker. Ordinary movement and contact-position aiming keep the game close to Pong; Burst is archived.

**Play yourself** starts Human vs Tracker. Use keyboard, the touch direction buttons or court dragging. Simulation remains at 30 decisions per second; display interpolation smooths animation without changing physics. Human matches retain the 1,600-tick deadline and do not use the deterministic loop shortcut. Physical-phone performance and touch feel still need testing.

The [competition protocol](COMPETITION.md) requires a plain dense neural network from the six raw observations to three scores: one to three fixed affine layers, zero to two hidden layers with uniform ReLU or Tanh, and a linear output. Graph programs, physics features, branches, shared parameters, computed weights, and input/output processing are ineligible. Smaller eligible networks may enter larger parameter classes, regardless of how their weights were created. The app runs and stores local leagues; public standings will require an independent evaluator and reserved evaluation seeds. Archived Burst results cannot be restored as Classic rankings.

The latest [Classic tactical test](results/classic-tactics/README.md), with replays and curves at `/training/classic-tactics.html`, separates legal placement opportunities from learned performance. It finds many scripted loss-to-win alternatives, but two controlled imitation continuations fail to produce an overall stronger model. The game rules and active weights are unchanged; see the retained methods and negative results before using those tactical labels for another experiment.

## Current Classic v4 workflow

Pong v4 builds speed over a rally. On each successful contact, let `s = min(13, abs(incoming_vx) + 1)` and `d` be ball Y minus the updated paddle center. Outgoing horizontal velocity is `±s`, away from the paddle; vertical velocity is `floor(9 * s * d / 195 + 0.5)`. The first return has horizontal speed 6, independently of contact offset. Placement sets approximately the same aiming angle as the ball accelerates, and a center return cannot reset speed. Its six-input/three-score interface has no burst resource.

The [gradual-speed study](results/pace-study/README.md) documents tuning, rule-aware opponents and the `pace_study.py` / `train_pace.py` reproduction workflow. Its report is `/training/pace-study.html`. **Pacer v4** and **Apprentice v4** are standalone, fully learned `6→128→128→3` Tanh actors trained through DAgger. The geometric teachers' trajectory calculations are not included in the exported networks. These two models form the default arena matchup.

**Vector Nano v4** adds an eligible `6→28→9→3` ReLU network with 487 parameters and 447 MACs. It encodes a physically derived opponent-side court-end contact goal in fixed dense weights, without training or learned-source transfer. It passes three fresh blocks against Pacer, Prism Nano, and the earlier Nano references, but fails against Prism Small and both adaptive scripts. The [Vector dense study](results/vector-dense-v4/README.md), `/training/vector-nano-v4.html`, and `vector_dense_v4.py` retain its exact construction, results, and forecast/angle/recovery limits.

**Prism Nano v4** adds an eligible `6→29→9→3` ReLU catalog comparison with 503 parameters and 462 MACs. Its weights are hand-constructed from approximate anticipation and conservative aiming priors, with no training or external trajectory helper. It passes three fresh blocks against Tracker, Anticipator, and Challenger, but fails the stronger Pacer/attacker target. The [analytic dense study](results/analytic-dense-v4/README.md) retains its exact export and full results; `/training/prism-nano-v4.html` provides the public report and download. `analytic_dense_v4.py` reproduces the fixed dense weights.

**Prism Small v4** is an eligible `6→42→33→3` ReLU model with 1,815 hand-constructed parameters and 1,737 MACs. A cap-speed forecast and coherent half-integer contact targets are encoded entirely in its dense matrices. It passes three fresh native blocks against Pacer and Prism Nano, while failing heavily against adaptive attack and defense. See the [robust dense study](results/robust-dense-v4/README.md), `/training/prism-small-v4.html`, and the exact constructor `robust_dense_v4.py`; these are physical priors without training or learned-source transfer.

The [compact handcrafted study](results/compact-handcrafted-v4/README.md) and [attacker-counter study](results/counter-attacker-v4/README.md) are ineligible archives. Their scalar arithmetic, trajectory features, tables, and planning circuits did not meet the intended dense-network challenge. They and hybrid Strategist have been removed from the active catalog; their results are not dense-network comparisons. The model contract is now `observation-6/action-scores-3/dense-mlp-v2`, so older general-graph rankings do not restore. Physics and historical evidence remain unchanged.

V4 fixtures contain 3,545 full-match transitions, 1,958 trace transitions, 20 boundary cases, five detector histories and a restart-reset check. `training/v4/game.ts` is its frozen packaged engine. Run `node --import tsx training/scripts/check_replays.ts` for its saved model replays. Earlier plain dense six-input weights remain eligible, but evaluate them again under v4 before comparing strength. The [v1](v1/README.md), [v2](v2/README.md) and [v3](v3/README.md) contracts remain immutable archives.

The [dense weights-only restart](results/dense-weight-transfer-v4/README.md) audits learned feature sensitivities and tests direct pruning, feature merging, and source-weight linearization in fixed Nano/Small MLPs without additional training. None of those candidates is an upgrade; all failed controls and strict dense exports are retained, and none was added to the catalog.

## Current setup and verification

Run from the repository root. The PyTorch scripts use the pinned dependencies in `requirements.txt`; `requirements-lock.txt` records the environment.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r training/requirements.txt
python3 training/v4/check.py
npm test
npm run build
```

Use `--no-loops` with the standalone headless example for human or stochastic-controller semantics. Only fixed deterministic stateless policy pairs may use the exact-loop shortcut. **The older `env.py`, `train.py` and `release.py` scripts target v3**, not Burst or Classic v4.

## Archived Burst experiment

This experiment is no longer an active arena mode. Its weights, scripts, reports and standalone [package](burst/README.md) remain available for reproduction. The eight-input/five-output weights cannot enter Classic matches.

Burst has eight observations: the same six ball/paddle values as Classic, followed by own and opponent energy divided by 96. Its five scores select up, stay, down, burst up or burst down. Normal movement covers two units; an affordable burst covers four and costs eight energy. Normal actions recharge one energy, while an exhausted burst request moves at normal speed without recharging. Every serve resets energy to 96. Horizontal speed increases by half a unit per return, capped at 13. Exact loop detection includes both energy values. The [contract](burst/README.md) defines all update order, rounding and terminal precedence.

The [rule study](results/burst-study/rules-README.md), available at `/training/burst-rules.html`, measures longer rallies and a pressure sequence with reproducible defensive counterfactuals. The [model-size study](results/burst-study/models/README.md), available at `/training/burst-models.html`, contains the training commands, learning curves, frozen selection manifests and final evaluations. `train_size_study.py` trains plain Tanh networks through pooled DAgger; the geometric teachers are training tools and do not appear in exported inference graphs.

| Model | Architecture | Parameters | MACs / decision | Smallest class |
| --- | --- | ---: | ---: | --- |
| Micro | `8→16→5` | 229 | 208 | Nano |
| Tiny | `8→16→16→5` | 501 | 464 | Nano |
| Small | `8→32→32→5` | 1,509 | 1,440 | Small |
| Medium | `8→64→64→5` | 5,061 | 4,928 | Standard |
| Large / Focus | `8→128→128→5` | 18,309 | 18,048 | Standard |

The first five sizes share data and optimization-pass budgets, with two initialization seeds each. **Tiny's 501 parameters** are the smallest useful learned starter demonstrated in this recipe: both initializations beat Tracker. This is not a minimum-size proof or proof of tactical mastery. All five initial models lose every final rally to the attacking teacher.

**Focus** is a separate continuation of Large that oversamples observed burst decisions. On a fresh 320-rally block it wins 44, loses 255 and draws 21 against that teacher, while Large loses all 320. Mapping Focus's burst actions to normal moves removes those wins. However, Focus loses to Large 25 / 197 / 98 and worsens against safer defense. It is a specialist, not an overall upgrade. The report distinguishes learned rescue timing from the multi-exchange placement counter that these networks have not learned.

Burst's standalone fixtures contain 102 isolated transitions and three complete five-rally matches with 11,905 frames. Inside a standalone copy of `training/burst/`, `npm install && npm run check` also checks the TypeScript implementation. The size-study report retains native ONNX parity and full-match replay checks, with separate manifests for the controlled size comparison and Focus follow-up. See that report's reproduction commands before training; retained final test blocks must not be reused for model selection.

## Archived v3 training and evaluation

The remainder records the original v3 experiment. Its results and verification counts apply to v3, not Burst or Classic v4. Reproducing its browser/release checks requires a separate checkout of the v3 source, for example commit `623e3a3`. The current replay checker can also verify its saved fixtures against the frozen v3 engine with `node --import tsx training/scripts/check_replays.ts --v3`.

Pong v3 keeps all v2 physics and the six-input/three-score interface. The rally deadline is now **1,600 ticks**. Fixed deterministic model pairs also draw when an exact post-contact ball/paddle state repeats. The arena labels loop and timeout draws separately; neither awards a point. Human matches use the full deadline.

The historical [v3 contract](v3/README.md), also downloadable as `/training/pong-v3.zip`, includes its engine, headless example, dependency-free Python reference, fixtures and checker.

The v3 [fully learned investigation](results/fully-learned/README.md) introduced **Contender**, a plain `6→64→64→3` Tanh actor trained through PPO and direct weight search. Under v3 it passed three fresh test blocks against original v2 and the unchanged attacker, including majority ordinary five-rally game wins in each block. Its curves and final results are at `/training/fully-learned.html`. There are no fixed geometry or paddle-feedback layers in this export.

The [earlier fixed-rules investigation](results/training-improvement/README.md) diagnoses the v3 regression and records **Strategist**, a hybrid with fixed trajectory features and a learned target head. It does not satisfy the fully learned requirement. Its separate curves and results remain at `/training/improvement.html`; the original v3 recipe below also remains historical evidence.

### Why v3 used these rules

The [rule study](results/rule-study/README.md) compared 800/1,600/3,200-tick limits and ten physics configurations on matched serves. Longer limits let some genuinely unfinished contests score; exact cycles otherwise consume time without changing the outcome. The study favored retaining paddle speed 2 and horizontal shot speed 5–13. Its original report is `/training/rule-study.html`.

Loop termination is valid only when both policies are fixed, deterministic and stateless. A sampled training policy, a human, changing weights or untracked recurrent state can choose differently after a repeated physical position. Consequently **DAgger and PPO training disable loop exits**, while deterministic validation, final evaluation and browser model matches enable them. All modes use the 1,600-tick maximum. This is explicit in run configurations and the standalone specification.

### V3 setup and verification

Run from the repository root. Recorded training used Python 3.12 on Apple Silicon, with one Torch CPU thread per run. Dependencies are pinned in `requirements.txt`; `requirements-lock.txt` records the full environment.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r training/requirements.txt
python3 training/v1/check.py
python3 training/v2/check.py
python3 training/v3/check.py
.venv/bin/python training/scripts/check_env.py
```

V3 fixtures contain 2,446 full-match transitions, 1,765 synthetic trace transitions, 18 boundary cases, five detector histories and a restart reset check. They include deterministic loop draws and the same exchange continuing to timeout with detection disabled. The copied TypeScript engine matches the archived v3 engine, not the current v4 engine. The historical vector simulator passed 978 isolated fixture transitions, 32,000 randomized transitions, all 4,187 published action transitions, and 8,326 deterministic transitions against the v3 scalar reference.

The headless example accepts `--no-loops`. For seed 24, both modes finish 1–1: 742 ticks with loops, 4,870 without. At a deadline, points take precedence; a repeated contact is labelled a loop ahead of a timeout. Instantiate/reset the detector on a new match or controller change. Its history resets at a new rally.

### V3 training and selection

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

### V3 training method

Each actor is `6→64→64→3`, Tanh hidden layers, 4,803 parameters and 4,672 MACs/decision, approximately 20 KiB ONNX. A separate `6→64→64→1` critic is training-only. Export uses supported Gemm/Tanh operators, opset18/IR10, embedded float32 weights and `rules=pong-v3` metadata.

- DAgger collects 65,536 decisions per round from a mixture of teacher/learner actions. A predictive attacker labels learner-visited states. Teacher probability declines to zero over four rounds; six supervised epochs fit a four-round replay window, minibatch 2048, Adam .001.
- PPO uses 256 environments × 128 steps, four epochs, minibatch 2048, gamma .99, GAE lambda .95, clip .2, clipped value loss, gradient norm .5, Adam eps 1e-5, learning rate .00025→.00005, entropy .01 and approximate-KL early stop .03. A decaying imitation anchor starts at .05.
- Rewards remain +1 for scoring, −1 for conceding, and +.02 per return. These are training choices, not game rules. Return rewards and discounting can influence defensive behavior; this run does not test alternative rewards.
- Opponents include Tracker, Anticipator, Challenger, a predictive interceptor, a greedy attacking script and up to five historical actor snapshots. Episodes end per rally; all five openings are visited. Sampled training episodes end on points or the 1,600 deadline, with no bootstrap through termination. Validation uses frozen first-argmax policies and exact loop detection.
- Both scripts use only the six exposed observations. The attacker predicts flights and chooses a reachable contact placement by a greedy reachability margin; it is a strong reference, not an optimal-play proof or multi-exchange planner.

### V3 final exported-model results

Each row is 1,000 rallies / 200 five-rally games on identical held-out serves, both sides, using the actual ONNX files. Draws are explicitly split by termination reason.

| Opponent | Apprentice W / L / D | Selfplay W / L / D | Selfplay loop / timeout draws |
| --- | --- | --- | --- |
| Tracker | 807 /180 /13 | 951 /18 /31 | 31 /0 |
| Anticipator | 821 /156 /23 | 937 /33 /30 | 30 /0 |
| Challenger | 833 /160 /7 | 935 /23 /42 | 42 /0 |
| Interceptor | 320 /257 /423 | 445 /30 /525 | 363 /162 |
| Attacker | 60 /876 /64 | 203 /350 /447 | 447 /0 |

The v3 model was **not universally stronger**. V2 weights were reevaluated under the same v3 rules and exact same final-test serves, separating weight differences from the new deadline:

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

### V3 reports, exports and browser checks

The historical v3 report is `/training/report.html`: all three learning curves, final tests, loop/timeout counts and comparisons against old weights. Raw curves are unsmoothed. Final-minibatch losses are diagnostics, not full-dataset averages. Cards derive their architecture from the actual uploaded graph. Those actors remain available locally as `public/models/selfplay-v3.onnx` and `apprentice-v3.onnx`.

```sh
.venv/bin/python training/scripts/verify_export.py
node --import tsx training/scripts/check_replays.ts
.venv/bin/python training/scripts/report.py
python3 training/scripts/package.py
npm test
npm run build
```

At the v3 release, both actors agreed with native ONNX on every action across 8,192 observations each. Ten complete ONNX matches reproduced exactly in the then-live v3 TypeScript engine: 17,536 frames, including 8 loop draws and one actual 1,600-tick model timeout. Browser WASM inference agreed on all 34,992 decisions and advanced the same states; maximum score difference was 9.54e−6. These historical verification records are in `results/v3/`.

## Archives

V1/v2/v3 contracts, ZIPs, original weights and original result directories are preserved. Their visual reports are `/training/v1/report.html`, `/training/v2/report.html` and `/training/report.html`. V2 source and original documentation are at Git commit `517a730`; use a separate checkout to reproduce that experiment. The pre-v3 rule-study source checkpoint is `0b0cba1`; its historical checker continues to use the explicit v2 reference. The legacy `env.py`, `train.py` and `release.py` target v3; Classic v4 uses `pace_study.py` and `train_pace.py`, and Burst uses `train_size_study.py` as documented above.

Methods: [PPO](https://arxiv.org/abs/1707.06347), [DAgger](https://arxiv.org/abs/1011.0686), [CleanRL PPO notes](https://docs.cleanrl.dev/rl-algorithms/ppo/). All training, inference and logging are local.

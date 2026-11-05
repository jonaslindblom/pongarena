# Offensive-balance comparison

No candidate from this study is promoted to the arena. The current Pong v3 physics, frozen packages and public model presets remain unchanged. Open `/training/offense-study.html` on the local app for synchronized replay panels, final comparisons, learning curves and the relationship to classic Pong.

## What was compared

All candidates keep the existing court, paddle movement, horizontal return speeds, seeded serves, six observations, three actions, 1,600-tick limit and exact model-loop rule. Only the vertical return law changes:

```
vy = floor(V * contact_offset / 15 + 0.5) + K * actual_paddle_movement_direction
```

| Candidate | V | K | Meaning |
| --- | ---: | ---: | --- |
| Baseline | 5 | 0 | Current v3 |
| Angle8 | 8 | 0 | Steeper placement angles |
| Angle9 | 9 | 0 | Still steeper placement angles |
| Motion2 | 5 | 2 | Current angle plus contact-motion deflection |

[RULES.md](RULES.md) is a standalone specification, including serve generation, negative rounding, clamping, simultaneous decisions, wall/contact ordering and draw precedence. There is no persistent spin or new hidden state. The shortest cross-court flight remains 26 ticks, or 0.87 seconds at 30 Hz.

## Final matched-serve results

The final run freezes source and model hashes in `manifest.json` before evaluating the 64 reserved seeds in `protocol.json`. Each matchup has 640 rallies: five serves per seed, with both side assignments. These seeds are separate from training/development and were checked against archived JSON seed lists. The final run was not used to select checkpoints or retune opponents.

| Rules | Strong attacker mirror: scored | Median returns in scored rallies | First receiver wins among scores | Attacker vs defender W/L/D |
| --- | ---: | ---: | ---: | --- |
| Baseline | 0 / 640 | — | — | 0 / 0 / 640 |
| Angle8 | 0 / 640 | — | — | 0 / 0 / 640 |
| Angle9 | 200 / 640 | 1 | 90.0% | 88 / 75 / 477 |
| Motion2 | 92 / 640 | 1 | 93.5% | 43 / 11 / 586 |

The strong scripts plan contact position and recovery using the candidate's rules, with exact discrete feasible contact paths. They are finite-horizon heuristics, not optimal game solvers. W/L/D is from the first player's perspective. Mirrored same-policy matches have equal total side-paired points by construction; the individual rally statistics still reveal whether scoring happens. Neither stronger candidate produced a scored attacker-mirror rally lasting six or more returns on this final set. The newly scored points favor the first receiver heavily rather than showing sustained expert exchanges.

Motion2 does reveal a useful skill distinction in a separate **diagnostic** matchup: its attacker beats simple interception in 314/320 rallies, with a median of 15 returns among scores. Against deliberate safe-return selection it gets 12 wins, 2 losses and 306 draws on those same diagnostic seeds. This is a promising mechanism at intermediate controller strength, but it does not resolve the strong-play serve bias. See [evaluation notes](evaluation/README.md).

## Angle8 still has guaranteed defense

The earlier scan found 36 arbitrary angle8 returns from which recovery into the old universal safe band failed. That did not prove the defender had to choose those returns. A controlled-invariant calculation now selects safe returns as well as recovery positions.

For every retained paddle position at an opponent contact height, **every** geometrically legal opposing return permits **some** exactly reachable own return that allows recovery into the retained set before the next opponent contact. Iteration continues to equality, not a chosen iteration limit. The baseline and angle8 both retain safe positions at all 153 contact heights, and all 1,050 serve height/slope cases enter this invariant in both roles. Thus either side can avoid conceding from every standard serve against arbitrary legal opponent play under these two rules.

This is independently checked through a different movement/shot implementation and corroborated with a constructive stateless defender. It is a diagnostic controller, never part of a learned submission. Angle9 and motion2 do not satisfy this sufficient proof from every serve; that failure does not prove an attacker can force a win.

## Four plain-network adaptation runs

Each rule received 20 generations of direct PyTorch weight search, starting from the same Contender network, with the same random mutations, seed sets, population size and opponent mix. Total active training decisions: **84,721,236**. This is a local fine-tuning probe rather than a full PPO/self-play campaign or an estimate of the learning ceiling. Details, unsuccessful pilot outputs and reproduction commands are in [training/README.md](training/README.md).

The selected export remains a plain `6 → 64 → 64 → 3` Tanh network, with 4,803 learned parameters and no fixed trajectory calculations. The final comparisons below are against the stronger rule-aware scripts, not the original scripted attacker.

| Rules | Opponent | Starting Contender W/L/D | Selected network W/L/D |
| --- | --- | --- | --- |
| Baseline | Adaptive attack | 0 / 96 / 544 | 0 / 12 / 628 |
| Baseline | Adaptive defense | 0 / 30 / 610 | 0 / 9 / 631 |
| Angle8 | Adaptive attack | 0 / 403 / 237 | 0 / 329 / 311 |
| Angle8 | Adaptive defense | 0 / 179 / 461 | 0 / 164 / 476 |
| Angle9 | Adaptive attack | 0 / 525 / 115 | 0 / 502 / 138 |
| Angle9 | Adaptive defense | 0 / 538 / 102 | 0 / 415 / 225 |
| Motion2 | Adaptive attack | 1 / 591 / 48 | 1 / 526 / 113 |
| Motion2 | Adaptive defense | 1 / 207 / 432 | 1 / 239 / 400 |

The changes mainly reduce concessions; they do not increase scoring against either script. Motion2 also trades worse defense against one opponent for improvement against the other. A scalar average fitness can accept such trades. Learned mirror matches scoring more frequently is insufficient evidence of strategic depth when their models still lose heavily to scripts.

`evaluation/confirmation.json` also retains matchups against frozen Contender, original v2 and each selected network itself, per-rally outcomes, return counts, point timing, loop/timeout reasons and whole-seed bootstrap confidence intervals. Public replays show the first reserved seed, all five serves, rather than selecting favorable examples.

## Distance from classic Pong

Angle-only changes retain hit-position aiming, described by Nolan Bushnell as a core Pong tradeoff in [this design interview](https://www.raspberrypi.com/news/the-history-of-pong-code-the-classics/). Motion2 adds a contact-timing mechanic but keeps the same controls and straight trajectories. It is a recognizable Pong variant, with one more rule to learn.

The arena's bounded movement and five seeded, time-bounded rallies are already custom competition choices. Original Pong used knob controllers, and Al Alcorn describes adding three horizontal speed increases as rallies continued in his [Smithsonian oral history](https://mads.si.edu/mads/id/NMAH-AC1498_Transcript_AllanAlcorn). Testing a gradual rally speed increase would therefore be historically grounded, but any counter or speed stage governing future physics must be included in the reproducible state, observations and loop key. It has not been implemented here.

## Verification and limits

- Exhaustive scalar/vector checks cover 120,258 contact transitions per candidate, 51,200 trajectory transitions, 2,048 masked symmetry transitions and 44 boundary/termination cases.
- An independent TypeScript implementation reproduces 44 fixtures and 12 complete Python trajectories, 2,156 transitions in total; baseline trajectories also match released `src/game.ts` exactly.
- Contact planning is checked over 4,530,504 feasibility entries; both adaptive scripts catch all 47,916 sampled exactly catchable flights.
- Independent invariant closure verifies 204,483 baseline and 182,299 angle8 state/shot cases, plus every standard serve case in both roles. Constructive defenders conceded zero in 3,200 diagnostic rallies across the two rules.
- Every export has three Gemm and two Tanh operations, with exact checkpoint parameters and zero action mismatches on 8,192 PyTorch/native-ONNX numeric probes per rule. Actual replay-input checks are recorded separately in `training/native-replay-verification.json`.
- The app's 19 tests pass with none skipped; the production build passes. Browser checks cover replay play/pause, stepping, scrubbing, serve/pair switching, loaded charts and desktop/320-pixel layouts.

**No real human playtest was performed.** The delayed-observation experiment is an uncompensated sensor-lag stress test: it keeps current own-paddle position and does not extrapolate the known observation age. It cannot estimate human win rates. No optimal-play claim is made for angle9 or motion2. No experimental model is promoted to the arena.

## Reproduce

```sh
.venv/bin/python training/scripts/check_offense_rules.py
node --import tsx training/scripts/check_offense_fixtures.ts
.venv/bin/python training/scripts/offense_invariant.py
.venv/bin/python training/results/offense-study/evaluation/verify_invariant.py
.venv/bin/python training/scripts/offense_eval.py --count 32
.venv/bin/python training/scripts/offense_report.py
```

`train_offense.py` supports fresh output directories as documented in the training notes. `confirm_offense.py --freeze` records hashes before the reserved run, and `confirm_offense.py` verifies them before and after evaluation. Both intentionally refuse to overwrite an existing final manifest/result; use a new study and fresh reserved seeds for a new selection experiment. Retain the frozen original results when rerunning diagnostics.

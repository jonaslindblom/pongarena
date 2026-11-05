# Can legal placement improvements transfer to learned Classic policies?

The game remains frozen Classic v4 (`pong-v4`). This experiment changes training labels, not physics or inference inputs. The result is **not an overall policy improvement**: the selected tactical continuation improves modestly against the defensive script while regressing clearly against the attacking script. It also fails to improve actual winning placements in the final local probes. No new model is recommended for promotion.

## Controlled training

Both arms begin from the same frozen Pacer checkpoint, a plain `6 → 128 → 128 → 3` Tanh network with 17,795 learned parameters. Exported ONNX graphs contain three Gemm nodes and two Tanh nodes. They have no trajectory calculations, rollout planner, lookup table or fixed feedback features.

The probe generator found legal paths that turn a scripted loss into a win when the same original scripts resume after the altered contact. Its training split supplies 16,898 action examples from 414 such states and 16 serve seeds. Those examples contain 6,608 numerically distinct float32 observations. There are 39 observations with conflicting desired actions, affecting 206 rows; the intervention labels are therefore not a globally consistent stateless target. The model sees six physical observations, without script identity, an intended contact target, elapsed rally ticks or intervention history.

Each training round collects a **shared** pool of game observations from both current learners against adaptive attack, adaptive defense, interception, frozen Pacer and Tracker. Both player perspectives receive labels from the original adaptive attacking teacher. Both arms also receive the same broad synthetic observations and the same sampled tactical-path observations, actual minibatch permutations, replay length and optimization-pass count.

- **Control:** the original adaptive attacking teacher labels every observation, including tactical-path states that originated from a defensive script. This is Pacer's original teacher, not necessarily that snapshot's controlled-side script.
- **Tactical:** ordinary and broad observations keep the original teacher labels; oversampled tactical paths use the selected loss-to-win action labels.

This isolates the label intervention within each run. Cross entropy is optimized by Adam, minibatch 2,048, four epochs per round, four rounds of replay, gradient clipping at one. Training disables deterministic loop exits. Development/full-match evaluation uses first-argmax actions and ordinary Classic loop rules. Geometry appears only in training supervision and diagnostic scripts.

## Two bounded attempts, retained separately

`run/` is the first attempt: seed 701, 24 rounds, 64 environments × 128 ticks per round, initial learning rate 0.0003, 8,192 tactical examples and 8,192 broad examples per round. It adds 196,608 shared environment steps and 786,432 labels per arm, counting sampled repetitions. Every evaluated trained checkpoint regresses against the strong opponents, so both arms correctly retain the unchanged round-zero Pacer checkpoint. At the fixed final training round, tactical wins 65 of 207 development placement opportunities versus Pacer's 31, but full-game performance is worse. Local placement improvement did not transfer to stronger play.

`anchored/` is one predeclared lower-drift follow-up, started before any final outcomes were generated: seed 709, 48 rounds, initial learning rate 0.00005, 1,024 tactical samples per round, and the same environment/broad/replay/epoch counts. Both arms add a KL penalty with coefficient **4**, temperature one, toward the initial Pacer output probabilities on every shared observation. This adds 393,216 environment steps and 1,228,800 labels per arm, including repeated samples. It is a separate attempt, not part of an equal-budget comparison with the first run.

Both learning-rate schedules decline by multiplying the initial rate by `0.4 + 0.6 * (1 - (round-1)/rounds)`. Checkpoints are evaluated every four rounds on 16 fixed development serve seeds. Selection maximizes mean point margin against **adaptive attack and adaptive defense only**; frozen Pacer is reported as a secondary regression check. The first checkpoint wins an exact tie. The follow-up control retains round zero; tactical selects round 44, with development margin −0.59375 versus the initial −0.60000.

Both attempts, source snapshots, curves' raw values, configuration hashes, selected checkpoints and fixed last-round checkpoints are retained. There was no third search after seeing final results.

## Frozen final full-match evaluation

`evaluation/manifest.json` freezes the selected exports/checkpoints, source hashes, probe protocol and final seeds before final outcomes are generated. Each pairing uses **64 fresh serve seeds × five rallies × both sides = 640 rallies**. Reported full-match outcomes use frozen PyTorch weights in the verified vector engine; native ONNX agreement is checked separately.

| Policy | Adaptive attack W / L / D | Adaptive defense W / L / D | Frozen Pacer W / L / D |
| --- | --- | --- | --- |
| Frozen Pacer / selected control | 156 / 453 / 31 | 78 / 450 / 112 | 310 / 310 / 20 |
| Selected tactical, round 44 | 114 / 508 / 18 | 94 / 432 / 114 | 301 / 320 / 19 |
| Control final round 48, secondary diagnostic | 81 / 545 / 14 | 66 / 425 / 149 | 285 / 318 / 37 |

The control's selected checkpoint is byte-identical to the initial Pacer checkpoint. The trained control-final export is retained separately because selecting round zero otherwise hides what the unsuccessful training actually did. It was not retroactively promoted.

Pairing the same complete serve seeds, tactical minus frozen Pacer has mean point-margin changes:

- Against attack: **−0.1516**, 95% bootstrap interval **[−0.2141, −0.0891]**.
- Against defense: **+0.0531**, interval **[+0.0063, +0.1000]**.
- Against Pacer: **−0.0297**, interval **[−0.0985, +0.0391]**.

These are changes in `(wins − losses) / rallies`, not percentage-point win-rate changes. Bootstrap resamples whole seeds, retaining both sides and all five rallies together. The ten rallies within one seed are not independent samples. `confirmation.json` retains per-seed records, all pairings and paired comparisons against both selected control and frozen Pacer.

## Final placement probes and state overlap

The final probe set uses 16 separately reserved seeds and 1,950 reachable flight starts. Exhaustive contact-center enumeration finds 410 states where the original scripted continuation loses and a legal changed placement wins. A learned model takes over only until its next contact, then the same original scripts resume. Every branch retains the original elapsed time and 1,600-tick deadline; no loop shortcut crosses this scheduled intervention. These are local placement tests, not matches played throughout by the network.

| Policy | Winning placements among 410 opportunities | Among 103 opportunities with a start absent from training tactical paths |
| --- | ---: | ---: |
| Frozen Pacer / selected control | 55 | 26 |
| Selected tactical | 54 | 25 |
| Control final round 48 | 51 | 28 |

The original full scripted continuation loses in every opportunity in this table. Thus some learned placements already beat the script locally, but this training intervention does not increase their count. Agreement with the retained path labels rises from **80.30% to 81.76%**, while actual winning placements do not improve. Exact agreement with one chosen action path is not the objective: any legal winning contact center receives credit.

Fresh serve seeds do not guarantee new physical states. The independent numerical audit, with signed zeros canonicalized, finds **12,162 of 16,616 final path samples (73.2%)** and **381 of 1,950 flight-start observations** in the training tactical-path data. Of the 410 strict opportunities, 307 have seen starts and 103 have unseen starts. The table above uses those canonicalized partitions; see [the independent audit](../verification.json). The raw evaluator uses byte keys that distinguish signed zero and records one fewer seen path sample and two fewer seen starts. That representation detail does not change any model outcome or conclusion. The seen/unseen labels concern the **tactical path dataset only**, not every pooled game or synthetic training observation.

All local probe outcomes, contact centers, winning-center sets, full action continuations, checkpoint hashes, label agreement and seen/unseen partitions are retained in `evaluation/tactic-probes.json`. The first and anchored runs' development probes are retained separately. No held-out tactical or match result was used to alter the selected weights.

## Verification and limits

All three exported roles pass ONNX validation and have 17,795 parameters. Native ONNX agrees with PyTorch on **8,192 numeric observations per export** and **12,452 additional actual game observations**, with zero action disagreements. Maximum actual-game score difference is approximately 1.21e−5. Six complete native-ONNX/scalar-reference games contain **12,476 frames**, retained in `evaluation/onnx-replays.json`. The integrating browser also reproduced all six games and **24,904 decisions** without action or state disagreements; see [browser verification](../browser-verification.json). These checks establish implementation agreement, not policy strength.

The local branch generator proves a concrete weakness in the finite scripted decision, not a forced win against every possible defense. Supervised imitation of a winning intervention can damage interception or recovery elsewhere. The experiment does not establish a neural-network ceiling, optimal play, human enjoyment or a minimum network size.

## Reproduce the retained experiment

Use the pinned project Python environment from the repository root. Existing output directories deliberately refuse overwrite. The following commands record the historical sequence; use fresh output paths for a replication and preserve the exact archived source when reproducing the first run.

```sh
.venv/bin/python training/scripts/train_classic_tactics.py \
  --dataset training/results/classic-tactics/probes/train.npz \
  --out training/results/classic-tactics/learning/run
.venv/bin/python training/scripts/eval_classic_tactics.py --development-probes

.venv/bin/python training/scripts/train_classic_tactics.py \
  --dataset training/results/classic-tactics/probes/train.npz \
  --out training/results/classic-tactics/learning/anchored \
  --rounds 48 --learning-rate 0.00005 --anchor-weight 4 \
  --tactical-samples 1024 --seed 709
.venv/bin/python training/scripts/eval_classic_tactics.py \
  --source training/results/classic-tactics/learning/anchored --development-probes
.venv/bin/python training/scripts/eval_classic_tactics.py \
  --source training/results/classic-tactics/learning/anchored --freeze
.venv/bin/python training/scripts/classic_tactics.py --phase final
.venv/bin/python training/scripts/eval_classic_tactics.py
.venv/bin/python training/scripts/eval_classic_tactics.py --final-probes
```

The failed first freeze invocation supplied a relative source path to an absolute-path conversion. It failed before writing a manifest or reading final outcomes; the path handling was corrected and only its incomplete copied artifacts were removed before the successful freeze. No final test was consumed by that plumbing error.

All final seeds are now consumed. Reproducing these numbers does not make them fresh for selecting another model. New tuning requires new reserved evaluation seeds. Numeric/native replay seeds check inference agreement and do not select model strength.

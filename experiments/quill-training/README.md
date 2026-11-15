# Training Quill's architecture

This experiment tests an independently trained **6 → 12 → 4 → 3 → 3 ReLU**
network: exactly Quill's architecture, **163 parameters and 141 MACs**. The
student begins with random weights and receives the ordinary six observations.
It has no extra inference features, memory, simulator access or action processing.

Quill and Counter provide no training weights, labels, opponents or checkpoint
selection. They are evaluated only after the candidate files and primary choice
are frozen. The preselected student is shipped as **Small** in the model roster.

## Result

**The programmed models remain stronger in this attempt.** None of the 13 frozen
candidate checkpoints scored above 50% against either held-out opponent. These
include repeated policies retained when a refinement failed to improve selection.
This is evidence about the tested training recipes, not a capacity bound.

The preselected model is `local-ppo-large-943`: Large distillation seed 943,
PPO refinement, then CMA-ES seed 2943 with sigma 0.003. The selected checkpoints
are imitation round 40, PPO round 96, and search generation 16.

| Opponent | Paired-contest score (95% interval) | Rally score (95% interval) |
| --- | ---: | ---: |
| Quill | 0.8% (0.3–1.5%) | 21.8% (20.8–22.8%) |
| Counter | 42.7% (38.6–46.6%) | 46.9% (45.6–48.3%) |

The large difference between Quill's contest and rally scores means that the
student wins some rallies but almost always loses their ten-rally paired contest.
The primary had one contest win, six draws and 505 losses against Quill; against
Counter it had 182 wins, 73 draws and 257 losses.

Distillation did not show a consistent advantage over the two script controls.
Its four starts had held-out rally scores of 14.1–32.9% against Quill, versus
20.9–24.2% for script imitation. Those small samples do not establish a general
ranking of training methods. The strongest development candidate was also not
the strongest Quill opponent: `large-942` scored 32.9% of Quill rally points, but
was not selected. Optimizing this development field therefore leaves an important
opponent-generalization problem. No post-test candidate substitution was made.

The result supports the project-specific observation that finding strong weights
for extremely small networks has been difficult with the methods tested. Quill's
weights demonstrate that its architecture can express Quill-level play; these
experiments do not establish a capacity bound. The programmed models benefit from
geometric insight and Codex-assisted formula revisions guided by simulated match
feedback. The comparison holds architecture fixed, not development effort, prior
knowledge or optimization budget. Large already beats Quill.

## Training effort

- Four Large-distillation starts: seeds 941–944. The teacher is the independently
  trained, frozen Large ONNX already in the repository. Temperature-2 soft action
  labels are collected on learner/teacher rollouts and synthetic contact states.
- Two script-imitation controls: seeds 941–942, with the same student, state
  collection, training budget and selection field. Their labels are the existing
  attack script's actions. Its integer-rounded forecasts are retained, so this
  is a comparison with the existing script, not an improved oracle.
- PPO refinement of the two strongest distilled students (942, 943) and the
  strongest script student (942). Each receives 8,388,608 environment steps,
  terminal win/loss rewards, and an equal mixture of scripts and a pool of up to
  eight recent learned opponents. The independent critic is discarded on export.
- CMA-ES refinement of the strongest PPO candidate from each label source:
  96 generations × 24 candidates, eight fresh shared paired serve sets per
  generation, against three scripts and Large. Both broad (`sigma=0.08`) and
  local (`sigma=0.003`) searches are recorded. A training-only perturbation probe
  motivated the smaller scale; no held-out opponent was consulted.

Imitation totals 9,437,184 environment steps across six runs; PPO totals
25,165,824 across three. Four searches evaluate 9,216 candidate vectors over
2,949,120 rallies. These are training budgets, not independent test observations.
All candidates have the same architecture and use CPU training.

Checkpoint selection maximizes mean paired-contest score against **attack,
defend, interceptor and Large**, on the existing 32 development seeds. Mean rally
score breaks ties. CMA-ES's training fitness is contest score plus 0.25 × rally
score; its saved checkpoints use the same development selection as the others.

## Evidence and replay

[results.json](results.json) records every run's configuration, selected stage,
development score, frozen hashes and held-out results. The primary model is
[public/models/small.onnx](../../public/models/small.onnx), also available in
PongArena's Models tab.
Held-out comparisons use 512 fresh shared serve sets drawn after freezing, both
sides and five rallies per side: **5,120 rallies per opponent**.

Contest score is `(paired wins + paired draws / 2) / 512`. Rally score separately
counts tied rallies as half. Intervals resample complete paired serve sets;
they describe serve uncertainty, not variability across training runs. The
primary candidate is selected before the final matches; other candidates are
reported as ablations and are not promoted based on their held-out scores.

From the repository root, using Python 3.12:

```sh
python3.12 -m venv training/.venv
training/.venv/bin/pip install -r training/requirements-search.txt
training/.venv/bin/python -m unittest discover -s training -p 'test_*.py'
training/.venv/bin/python experiments/quill-training/evaluate.py --output /tmp/quill-student-replay.json
npm ci
node --import tsx experiments/quill-training/verify.mjs
```

Use the Node version in `.nvmrc`. The verifier checks the strict model contract,
217 runtime probes, and exact Python/WASM paired-margin agreement on the first
16 held-out sets per opponent (64 games). The Python command replays all 512 sets.

The training recipes are [quill-distill.json](../../training/configs/quill-distill.json)
and [quill-ppo.json](../../training/configs/quill-ppo.json). Each run requires a
fresh output directory. Raw checkpoints and trajectories stay in ignored
`training/runs/`; the roster holds the frozen student and this directory holds
the experiment evidence.

Reconstruct the selected model, from the repository root:

```sh
training/.venv/bin/python training/distill.py --seed 943 --out training/runs/reproduce-distill-943
training/.venv/bin/python training/distill.py --config training/configs/quill-ppo.json --seed 943 --resume training/runs/reproduce-distill-943 --out training/runs/reproduce-ppo-943
training/.venv/bin/python training/search.py --seed 2943 --sigma 0.003 --resume training/runs/reproduce-ppo-943 --out training/runs/reproduce-search-943
cmp training/runs/reproduce-search-943/model.onnx public/models/small.onnx
```

The reference platform is Python 3.12.11 on macOS ARM64, one CPU thread per
process, with the pinned training dependencies and CMA 4.4.0. Other platforms can
produce different floating-point training trajectories. The frozen model hash is
`5e72e00a62404e78974a33351667d5899efd8b93ae2ff09970ff8aa23e5eb82f`.

At the experiment freeze, all 16 Python tests and all 145 application tests passed,
with no skips; TypeScript checking passed. PyTorch/ONNX export parity and the
existing physics fixtures passed. All 512 paired sets per opponent replayed
exactly. [verification.json](verification.json) records browser-contract checks,
217 WASM probes and 64 games matching the Python margins. UI end-to-end tests
were not run for the original training-only experiment.

[reproduction.json](reproduction.json) records byte-for-byte reconstruction of
all three selected checkpoints. Imitation and PPO were repeated completely;
the duplicate search was stopped after reproducing the selected generation-16
checkpoint. The original search completed all 96 generations; its full budget
was not repeated a second time.

To reconstruct the controls, use `--labels script --seed 941` or `942` with
`distill.py`; the other distilled starts use seeds 941, 942 and 944. PPO parents
and every search seed/configuration are recorded in `results.json`. Preserve the
label-source flag when resuming script-trained policies. Future tuning informed
by these results needs a new independent final evaluation.

## Current roster comparison

The [current comparison](current-comparison.json) evaluates Small, Quill, Counter,
Large and OneShot on another shared set of 512 paired serves. It records the
current model hashes and replayable outcomes. The study results above retain
their original held-out seeds; this separate comparison puts the shipped models
on one common field without changing the study's selected candidate.

The [method-label revision](../model-metadata.json) and
[development-description revision](../programmed-method-metadata.json) updated
file hashes for the programmed opponents in the saved records. They changed no
graph, tensor or other model field. The original freeze timestamps, selected
student, serve seeds and match outcomes are preserved. The audits record the
earlier file identities and verify that the model payloads are identical.

From the repository root:

```sh
training/.venv/bin/python experiments/quill-training/current-comparison.py evaluate experiments/quill-training/current-comparison-plan.json /tmp/current-comparison.json
cmp /tmp/current-comparison.json experiments/quill-training/current-comparison.json
node --import tsx scripts/generate-findings.ts --check
```

[freeze.json](freeze.json) preserves source hashes at the time of the study.
The published-model checker and Large trainer CLI have since been updated for
the current roster; their source hashes are not assertions about today's files.
The training algorithms, pinned recipes, selected student's bytes and export
metadata are unchanged. The student's ONNX metadata retains the experiment name
“Quill-sized student” so the frozen bytes remain reproducible.

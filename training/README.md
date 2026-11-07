# Training

Two ordinary PyTorch networks learn Pong from a geometric script teacher. Both
start from random weights. Handwritten models are excluded from training,
opponents and checkpoint selection, so they remain independent comparisons.

## Run

Python 3.12 and a CPU are enough. From this folder:

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python check.py
.venv/bin/python train.py --size small --out runs/small
.venv/bin/python train.py --size large --out runs/large
```

Use fresh output directories. Each run writes the selected `model.onnx`, a
PyTorch checkpoint, its exact configuration and training metrics. Add `--smoke`
for a quick pipeline check. Training takes minutes, not GPU hours.

## How it works

`game.py` implements the same physics as the browser. `teachers.py` contains
three geometric scripts: attack, defend and intercept. Training collects play
against these scripts, labels the learner's own positions with the attacker,
and mixes in synthetic positions. Cross-entropy fits the attacker’s three action
choices; Large also emphasizes positions near paddle contact. The two files in `configs/` define the whole
recipe; exported inference uses only six observations and dense network weights.

Every eight rounds, the learner plays five serves on both sides of eight fixed
development seeds. The best mean point margin against the three scripts selects
the checkpoint. Training, development and final evaluation use disjoint seed
ranges. Single-threaded CPU execution and fixed seeds make runs repeatable with
the pinned dependencies; floating-point results can vary across platforms.

## Check and compare

```sh
.venv/bin/python check.py --run runs/small
.venv/bin/python evaluate.py runs/small/model.onnx --output runs/small/evaluation.json
.venv/bin/python evaluate.py runs/large/model.onnx --seed 42 --output runs/large/fresh.json
```

The default evaluation uses 64 published reference seeds, paired sides and five
serves: 640 rallies per opponent. `--seed` creates a separate evaluation set.
After freezing a model, add `--opponent path/to/model.onnx` to compare another
model. Draws count as half a point in `rally_score`.

`check.py` verifies the shared physics fixtures, script decisions and optional
PyTorch/ONNX output parity. `--models ../public/models` checks the shipped Small
and Large files. This folder is self-contained: it needs no model downloads,
repository history or stored training data.

## Published result

| Recipe | Parameters | Rally score against Quill |
| --- | ---: | ---: |
| Small | 235 | 27.3% |
| Large | 34,307 | 39.8% |

These are the frozen recipes on 640 rallies each, counting draws as half a point.
[results.json](results.json) records the hashes, script comparisons and runtime.
Both full runs reproduced identical ONNX bytes from an isolated copy of this
folder. Quill is a separate, frozen comparison model.

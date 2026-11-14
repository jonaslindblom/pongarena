# Trained models

Small and Large start from random PyTorch weights. Small learns by imitating
geometric scripts; Large adds reinforcement learning through game outcomes.
Handwritten models supply no weights, labels, opponents or checkpoint-selection
results.

| Model and recipe | Layer widths | Training | Seed |
| --- | --- | --- | ---: |
| [Small](configs/small.json) | 6 → 16 → 6 → 3 | Imitation learning | 731 |
| [Large](configs/large.json) | 6 → 128 → 128 → 128 → 128 → 3 | Imitation + PPO | 733 |

Both use ReLU hidden layers and a linear output. The linked recipes contain all
training settings. Imitation uses the attacking script's actions as labels;
Large emphasizes positions near paddle contact. [PPO](https://arxiv.org/abs/1707.06347)
then rewards rally wins against scripts and frozen snapshots of the learned
policy. Only the actor network is exported.

`train.py` selects the checkpoint with the best mean point margin against three
scripts on 32 development seeds. Training, selection and final evaluation use
disjoint seed ranges. The [model contract](../docs/RULES.md#observations-and-actions)
defines the six inputs and three actions.

## Reproduce the shipped models

The shipped weights use the original `pong` rules; the current checkout trains
with `pong-v3`. Use this [pinned training revision](https://github.com/jonaslindblom/pongarena/commit/e006aa9ac64afe8e866c61a433077fb0561010d2)
to reproduce the shipped weights and their recorded results. It contains the
matching simulator, recipes, dependencies and reference model files.

From the repository root:

```sh
git fetch origin e006aa9ac64afe8e866c61a433077fb0561010d2
git worktree add --detach ../pongarena-training-baseline e006aa9ac64afe8e866c61a433077fb0561010d2
cd ../pongarena-training-baseline/training
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python check.py --models ../public/models
.venv/bin/python train.py --size small --out runs/small
.venv/bin/python train.py --size large --out runs/large
cmp runs/small/model.onnx ../public/models/small.onnx
cmp runs/large/model.onnx ../public/models/large.onnx
```

Use fresh output directories. Each run saves its selected ONNX model, checkpoint,
configuration and metrics. `cmp` verifies byte-for-byte reconstruction. The
reference platform and hashes are in [results.json](results.json); floating-point
differences may affect training on other platforms.

## Reproduce the recorded scores

In the same pinned checkout, evaluate the shipped files on the saved final seeds:

```sh
.venv/bin/python evaluate.py ../public/models/small.onnx --opponent ../public/models/quill.onnx --output runs/small-v-quill.json
.venv/bin/python evaluate.py ../public/models/large.onnx --opponent ../public/models/quill.onnx --output runs/large-v-quill.json
.venv/bin/python evaluate.py ../public/models/large.onnx --opponent ../public/models/small.onnx --output runs/large-v-small.json
```

| Model | Parameters | Rally score vs Quill (95% interval) |
| --- | ---: | ---: |
| Small | 235 | 24.4% (22.4–26.4%) |
| Large | 50,819 | 60.5% (58.2–62.9%) |

Large scores **88.1% against Small** (86.3–89.8%). Each comparison uses 128 seeds,
both sides and five rallies per side: 1,280 rallies per opponent. Draws count as
half a point. Intervals resample whole paired serve sets and describe seed-set
uncertainty, not variation between training runs.

These scores use `pong`. For the current `pong-v3` comparison, see the
[reference league](../public/league.md); scores across rule versions are not
directly comparable.

## Train with current rules

From `training/` in the current checkout:

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python check.py
.venv/bin/python train.py --size small --out runs/small
.venv/bin/python train.py --size large --out runs/large
.venv/bin/python check.py --run runs/large
.venv/bin/python evaluate.py runs/large/model.onnx --seed 42 --output runs/large/fresh.json
```

Python 3.12 and a CPU are enough. Add `--smoke` to a training command for a quick
pipeline check, or `--seed` to use a different training initialization. Evaluate
new models on fresh seeds after freezing their weights. `check.py --run` verifies
PyTorch/ONNX agreement; it does not compare new training against the shipped model.

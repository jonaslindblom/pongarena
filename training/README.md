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

Training and evaluation use the current `pong-v3` rules. `train.py` selects the
checkpoint with the best mean point margin against three scripts on 32 development
seeds. Training, selection and final evaluation use disjoint seed ranges.
The [model contract](../docs/RULES.md#observations-and-actions) defines the six inputs
and three actions.

## Reproduce the shipped weights

From the repository root:

```sh
cd training
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python check.py --models ../public/models
.venv/bin/python train.py --size small --out runs/small
.venv/bin/python train.py --size large --out runs/large
cmp runs/small/model.onnx ../public/models/small.onnx
cmp runs/large/model.onnx ../public/models/large.onnx
.venv/bin/python check.py --run runs/small
.venv/bin/python check.py --run runs/large
```

Use fresh output directories. Each run saves its selected ONNX model, checkpoint,
configuration and metrics. `cmp` verifies byte-for-byte reconstruction;
`check.py --run` verifies PyTorch/ONNX agreement. The reference platform and hashes are in
[results.json](results.json); floating-point differences may affect training on
other platforms. Python 3.12 and a CPU are enough.

## Reproduce the recorded scores

From `training/`, evaluate the shipped files on the saved final seeds:

```sh
.venv/bin/python evaluate.py ../public/models/small.onnx --opponent ../public/models/quill.onnx --output runs/small-v-quill.json
.venv/bin/python evaluate.py ../public/models/large.onnx --opponent ../public/models/quill.onnx --output runs/large-v-quill.json
.venv/bin/python evaluate.py ../public/models/large.onnx --opponent ../public/models/small.onnx --output runs/large-v-small.json
```

| Model | Parameters | Rally score vs Quill (95% interval) |
| --- | ---: | ---: |
| Small | 235 | 23.9% (21.7–26.1%) |
| Large | 50,819 | 74.8% (72.3–77.2%) |

Large scores **83.0% (80.7–85.3%) against Small**. Each comparison uses 128
seeds, both sides and five rallies per side: 1,280 rallies per opponent. The final
seeds were drawn after the model weights were frozen and are saved in
[results.json](results.json). Draws count as half a point. Intervals resample whole
paired serve sets and describe seed-set uncertainty, not variation between
training runs.

The [published league](https://jonaslindblom.github.io/pongarena/#league) compares
all models on its own shared serves. Compare scores within the same opponent field
and seed set.

## Train a new model

Use `--seed` to change the training initialization, or edit a recipe. Add `--smoke`
for a quick pipeline check. After freezing new weights, evaluate them on fresh
serves by choosing a new RNG seed:

```sh
.venv/bin/python evaluate.py runs/large/model.onnx --seed 42 --output runs/large/fresh.json
```

Keep the seed and output record with the model so others can replay the comparison.

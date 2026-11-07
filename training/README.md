# Training

Small and Large start from random PyTorch weights. Small learns from geometric
scripts; Large then improves through game outcomes. Handwritten models never
supply weights, labels, opponents or checkpoint-selection results.

## Run

Python 3.12 and a CPU are enough. From this folder:

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python check.py
.venv/bin/python train.py --size small --out runs/small
.venv/bin/python train.py --size large --out runs/large
```

Use fresh output directories. Each run saves its selected `model.onnx`, PyTorch
checkpoint, configuration and metrics. Add `--smoke` for a quick pipeline check
or `--seed 734` to try another initialization. Full standalone runs took about
49 seconds for Small and 4.4 minutes for Large on the reference machine.

## How it works

1. **Imitate.** Play against attack, defend and intercept scripts. Label the
   learner's own positions with the attacking teacher and mix in synthetic
   positions. Cross-entropy trains the three action scores. Large gives extra
   attention to positions near paddle contact.
2. **Improve Large.** [PPO](https://arxiv.org/abs/1707.06347) rewards rally wins
   (+1), losses (−1) and draws (0). Opponents are scripts and frozen snapshots of
   the learned policy. A separate critic helps training; only the actor is exported.
3. **Select.** At configured intervals, play five serves on both sides of 32
   development seeds against all three scripts. Keep the best mean point margin
   across checkpoints, including the imitation stage. Compare three initialization
   seeds before freezing the shipped model.

`configs/` defines both recipes. `game.py` matches the browser physics;
`teachers.py`, `train.py` and `ppo.py` implement the stages. Inference uses only
six observations and dense weights. This folder needs no model downloads,
repository history or stored training data.

## Check and compare

```sh
.venv/bin/python -m unittest discover -p 'test_*.py'
.venv/bin/python check.py --run runs/large
.venv/bin/python evaluate.py runs/large/model.onnx --output runs/large/evaluation.json
.venv/bin/python evaluate.py runs/large/model.onnx --seed 42 --output runs/large/fresh.json
```

Default evaluation uses 128 published seeds, paired sides and five serves:
**1,280 rallies per opponent**. After freezing a model, add
`--opponent path/to/model.onnx` for a head-to-head comparison. Draws count as
half a point. Use `--seed` for fresh evaluation; published seeds are public and
should not guide further tuning if you intend to claim independent results.

Training, development and final evaluation use disjoint seed ranges.
`check.py` verifies physics, script decisions and optional PyTorch/ONNX parity;
`--models ../public/models` also checks the shipped model and recipe hashes.
Fixed seeds and single-threaded CPU runs reproduced both ONNX files byte for byte
from an isolated copy of this folder. Floating-point results can vary across platforms.

## Results

| Model | Parameters | Rally score vs Quill (95% interval) |
| --- | ---: | ---: |
| Small | 235 | 24.4% (22.4–26.4%) |
| Large | 50,819 | 60.5% (58.2–62.9%) |

Large scores **88.1% against Small** (86.3–89.8%). Each comparison uses 1,280
rallies after freezing the models. Intervals resample whole paired serve sets;
they describe serve-set uncertainty, not variation between training runs.
Across three training seeds, development scores averaged over the scripts range
from 22.4–32.2% for Small and 74.2–78.8% for Large.

[results.json](results.json) records seeds, hashes, resource counts and scores.
These are reproducible baselines within this arena, not a claim of optimal
training. Quill remains an independent 163-parameter handwritten comparison.

# Trained models

Small and Large begin with random PyTorch weights. Large learns from geometric
scripts and game outcomes. Small distills the independently trained Large policy,
then uses PPO and CMA-ES refinement. Programmed models supply no weights, labels,
training opponents or checkpoint-selection results.

| Model | Layer widths | Parameters | Training |
| --- | --- | ---: | --- |
| Small | 6 → 12 → 4 → 3 → 3 | 163 | Large distillation → PPO → CMA-ES |
| Large | 6 → 128 → 128 → 128 → 128 → 3 | 50,819 | Script imitation → PPO |

Both use ReLU hidden layers and a linear output. Only the actor network is
exported. Small has exactly Quill's architecture; the
[training study](../experiments/quill-training/README.md) records its six random
starts, refinements, controls and held-out results. The shipped Small is the
preselected `local-ppo-large-943` candidate, not the candidate that happened to
score best against Quill after testing.

Training and evaluation use `pong-v3`. Large selects the best mean point margin
against three scripts on 32 development seeds. Small selects mean paired-contest
score against those scripts and Large, with rally score breaking ties. Training,
selection and final evaluation use disjoint seed ranges. The
[model contract](../docs/RULES.md#observations-and-actions) defines the six inputs
and three actions.

## Reproduce the shipped weights

From the repository root, using Python 3.12:

```sh
python3.12 -m venv training/.venv
training/.venv/bin/pip install -r training/requirements-search.txt
training/.venv/bin/python training/check.py --models public/models

# Large: the independent teacher used by Small
training/.venv/bin/python training/train.py --size large --out training/runs/large
cmp training/runs/large/model.onnx public/models/large.onnx
training/.venv/bin/python training/check.py --run training/runs/large

# Small: selected checkpoints feed the next stage
training/.venv/bin/python training/distill.py --seed 943 --out training/runs/small-distill
training/.venv/bin/python training/distill.py --config training/configs/quill-ppo.json --seed 943 --resume training/runs/small-distill --out training/runs/small-ppo
training/.venv/bin/python training/search.py --seed 2943 --sigma 0.003 --resume training/runs/small-ppo --out training/runs/small-search
cmp training/runs/small-search/model.onnx public/models/small.onnx
training/.venv/bin/python training/check.py --run training/runs/small-search
```

Use fresh output directories. Each run saves its selected ONNX model, checkpoint,
configuration and metrics. `cmp` checks byte-for-byte reconstruction;
`check.py --run` checks PyTorch/ONNX agreement. The reference platform is Python
3.12.11 on macOS ARM64 with one CPU thread and the pinned dependencies. Other
platforms may produce different floating-point training trajectories.

[results.json](results.json) records the shipped hashes and recipes. Large's
recipe is [large.json](configs/large.json); Small uses
[quill-distill.json](configs/quill-distill.json),
[quill-ppo.json](configs/quill-ppo.json), and the search settings above. The
[reproduction record](../experiments/quill-training/reproduction.json) confirms
byte-for-byte reconstruction of Small's three selected checkpoints: complete
imitation and PPO runs, followed by search through the selected generation 16.
The original search completed all 96 generations; the duplicate full search was
not completed.

## Reproduce the recorded scores

Small's held-out study uses 512 paired serve sets against Quill and Counter:

```sh
training/.venv/bin/python experiments/quill-training/evaluate.py --output /tmp/small-replay.json
```

Large's original result in [results.json](results.json) uses 128 paired serve sets
against scripts and Quill. It scores 74.8% of rally points against Quill, with a
95% interval of 72.3–77.2%. Replay that record with:

```sh
training/.venv/bin/python training/evaluate.py public/models/large.onnx --opponent public/models/quill.onnx --output /tmp/large-replay.json
```

Each serve set contains five rallies from each side. Draws count as half a point;
intervals resample complete paired serve sets and describe serve uncertainty, not
variation across training runs. Compare models on the same opponent field and
seed set; the [current comparison](../experiments/quill-training/current-comparison.json)
and the published league provide shared comparisons.

## Train a new model

Use `--seed` to change the initialization, or edit a recipe. `distill.py --labels
script` runs the script-imitation control with Small's architecture. Add `--smoke`
to `train.py` or `distill.py` for a quick pipeline check. After freezing new
weights, choose fresh evaluation serves:

```sh
training/.venv/bin/python training/evaluate.py training/runs/small-search/model.onnx --seed 42 --output training/runs/small-search/fresh.json
```

Keep the seed and output record with the model so others can replay the comparison.
Tuning informed by the published results needs a new independent final evaluation.

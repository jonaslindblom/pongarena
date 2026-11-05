# Teacher and distillation experiments

These experiments changed training while keeping Pong v3 physics, observations,
actions, deadline and deterministic loop detection fixed. Run commands below
from the repository root with the environment in `training/requirements.txt`.
Use fresh output directories; the scripts refuse to overwrite learning logs.

The teacher predicts reachable incoming contacts and moves toward a minimax
recovery position while the ball travels away. It consumes only the six public
observations. Restricting its desired incoming contact offset to 13 leaves two
physical units of contact margin. The restriction changes supervision, not the
game's allowed edge hits. `runs/teacher-experiment/` retains the exact teacher
comparisons, including their 60 dispersed development seeds.

## Retained runs

Each run directory contains its original configuration, all completed learning
rows, console log, selected checkpoint and ONNX export. Available export checks
and interruption metadata are retained unchanged. `last.pt`, optimizer state and
replay buffers are not archived; these trainers do not implement exact resume.

| Run | Method | Completed / planned rounds | Selected round | Attacker W/L/D | v2 W/L/D |
| --- | --- | ---: | ---: | ---: | ---: |
| `distill-71` | Raw-input action classification | 20 / 20 | 17 | 56/198/66 | 90/132/98 |
| `distill-73` | Raw-input action classification | 40 / 40 | 31 | 13/115/192 | 49/112/159 |
| `target-81` | Raw-input target regression + CE | 24 / 24 | 20 | 10/206/104 | 69/207/44 |
| `target-feature-83` | Frozen geometry + learned target + CE | 16 / 24 | 12 | 43/0/277 | 149/0/171 |
| `target-feature-89` | Frozen geometry + learned target + CE | 24 / 24 | 14 | 29/0/291 | 113/19/188 |
| `target-mse-83` | Frozen geometry + learned target, pure MSE | 24 / 24 | 23 | 24/0/296 | 144/0/176 |

These are checkpoint-selection results on the same 32 development seeds, five
openings and both sides (320 rallies per opponent). Checkpoint selection maximizes
the smaller `(wins + half draws) / rallies` score against attacker and v2, breaking
ties by their mean. These rows are not independent final tests. Main experiment
records report subsequent development confirmation and held-out evaluation.

The raw-input variants failed the competition objective and remain here as
negative results. Geometry reduced the approximation problem. Pure MSE improved
target error but did not produce the highest competition score. The CE83 run was
interrupted after round 16 to test pure MSE; its selected round 12 checkpoint had
already been written. See its `termination.json` for the recorded reason.

## Reproduce training

```sh
.venv/bin/python training/scripts/train_distill.py --seed 71 --rounds 20 --out training/runs/reproduce-distill-71
.venv/bin/python training/scripts/train_distill.py --seed 73 --rounds 40 --out training/runs/reproduce-distill-73
.venv/bin/python training/scripts/train_target.py --seed 81 --rounds 24 --max-offset 15 --classification-weight .01 --out training/runs/reproduce-target-81
.venv/bin/python training/scripts/train_target.py --seed 83 --rounds 24 --features --max-offset 13 --classification-weight .01 --out training/runs/reproduce-target-feature-83
.venv/bin/python training/scripts/train_target.py --seed 89 --rounds 24 --features --max-offset 13 --classification-weight .01 --out training/runs/reproduce-target-feature-89
.venv/bin/python training/scripts/train_target.py --seed 83 --rounds 24 --features --max-offset 13 --classification-weight 0 --out training/runs/reproduce-target-mse-83
```

Historical CE configurations describe `.01` in their `loss` field but predate the
explicit `classification_weight` field. Supply `.01` as above: the current default
is zero. The raw target run predates `features` and `max_offset` configuration
fields; its effective values were false and 15. For CE83, keep `--rounds 24` even
when comparing only its first 16 completed rounds: this value controls the learning
rate schedule. A full rerun can select a later checkpoint; the archived round 12
weights are the exact selected candidate for this experiment.

All runs use one Torch CPU thread, seeded NumPy/Torch generators, 256 environments,
256 decisions per round, six optimizer epochs, minibatches of 2048, a four-round
replay window and Adam learning rate .001→.0003. Incoming states within 12 ticks of
contact receive three times the loss weight. The teacher-action probability is
`max(0, 1 - (round - 1) / 8)` and one percent of actions are random. Rollout partners
are attacker, v2, interceptor and the original minimax teacher with offset limit 15,
including when the labeling teacher uses limit 13. Sampled collection disables
loop exits; deterministic evaluation enables them. Both retain the 1600-tick limit.
Pinned dependencies and the same platform are needed for closest numerical
reproduction; no claim of bit-for-bit training across platforms is made.

The action model is `6→128→128→64→3` with Tanh hidden layers. Target models predict
one normalized paddle center and use a fixed disclosed feedback branch to emit
`[-error, 1.001/80, error]`. Feature variants contain a frozen ReLU network computing
normalized next-contact landing Y and time, followed by a learned
`8→128→128→64→1` Tanh head. The full exported graph still has only six inputs and
three outputs. Its fixed physics weights are a disclosed prior; the model is not
a fully learned raw-input MLP. It contains 48,063 parameters, requires 45,152 MACs,
and occupies 208,140 bytes. All three feature exports matched Torch actions on
8,192 reachable observations; see each run's `export-check.json`.

`teacher_probe_mae` and `near_contact_mae` are in physical paddle-coordinate units,
measured on up to 8,192 teacher-visited development observations. They are diagnostics,
not a bound on every live position. `training_accuracy` is measured on the current
classification replay window. Logged loss is the final minibatch, not a full-data
average. Wall times include concurrent work and are not benchmarked throughput.

## Reproduce teacher comparisons

```sh
.venv/bin/python training/scripts/teacher_experiment.py --seed 1217 --count 60 --out training/runs/reproduce-teacher/recovery.json
```

The conservative-offset comparison uses the same sampled seeds and the public
functions in `teacher_experiment.py`:

```sh
.venv/bin/python - <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, 'training/scripts')
from env import scripted
from policy import Actor, actions
from teacher_experiment import evaluate_pair, target_teacher
torch.set_num_threads(1)
seeds = np.random.default_rng(1217).choice(100_000_000, 60, replace=False) + 1_200_000_000
v2 = Actor()
v2.load_state_dict(torch.load('training/results/v2/ppo.pt', weights_only=True))
v2.eval()
opponents = {'attacker': lambda x: scripted(x, True), 'v2': lambda x: actions(v2, x)}
rows = []
for limit in (15, 14, 13):
    for name, opponent in opponents.items():
        result = evaluate_pair(lambda x: target_teacher(x, limit), opponent, seeds)
        rows.append(dict(max_offset=limit, opponent=name, **result))
output = Path('training/runs/reproduce-teacher/conservative.json')
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(dict(seeds=seeds.tolist(), rows=rows), indent=2) + '\n')
PY
```

No final holdout is used by these commands. Use new development seeds for future
tuning rather than reusing published final tests.

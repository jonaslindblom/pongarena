# Quill Speed and Quill Geometry

These two variants extend [Quill](QUILL.md) using weight programming, without
classic training. All weights come from explicit formulas and geometry. There
is no optimizer, gradient descent, imitation dataset, learned checkpoint or
runtime controller outside the dense ONNX graph. We designed a finite set of
hypotheses, evaluated their play, and manually selected two candidates before
drawing fresh evaluation serves. This is evaluated engineering, not a zero-shot
construction claim.

| Model | Class | Architecture | Stored parameters | MACs |
| --- | --- | --- | ---: | ---: |
| Original Quill | Small | `6 → 12 → 4 → 3 → 3` | 163 | 141 |
| Quill Speed | Small | `6 → 18 → 4 → 3 → 3` | 229 | 201 |
| Quill Geometry | Large | `6 → 43 → 729 → 6 → 467 → 3` | 41,430 | 40,182 |

The original Quill and both trained models retain their original bytes. The new
models are selectable and downloadable in Models. Geometry spends substantially
more capacity than Quill Speed and cannot enter the Small division.

## From the Weights experiments to hypotheses

The Weights tab demonstrates three useful building blocks: boundary detection,
behavior switching and target adjustment. These experiments use the same ideas
to address a specific limitation: Quill estimates flight time using maximum
horizontal speed even when the ball is moving more slowly.

The [replay script](../experiments/quill/run.py) compares **16 named designs**,
including the original Quill, against the seven original models on **32 shared
exploratory serve sets**. Every matchup swaps sides and replays all five serves.
The [complete report](quill-exploration.json) retains design settings, seeds,
model hashes, source hashes and results for every hypothesis, including failures.
The designs were developed iteratively from formulas and exploratory feedback;
the replay freezes the full matrix rather than searching its parameters.
Quill's self-match is a draw in this exploratory reference field; the fresh
comparison below instead uses six common opponents with no self-matches.

| Hypothesis | Exploratory finding |
| --- | --- |
| More accurate multiplication within 256 parameters | Helps some matches, but slower-ball timing remains approximate. |
| Bounded extra lead for slower incoming balls | The 20-tick cap produced the selected Quill Speed; doubling it hurt play. |
| Speed-specific flight banks | Better timing than assuming speed 13; removing aiming or tick correction hurt play. |
| Exact discrete flight timing | Improves on a generic half-tick correction without needing a division or ceiling operator at runtime. |
| Opponent-contact geometry for recovery | Selected for Quill Geometry; restricting it to speed 13 regressed. |
| Integer direct and bank-shot targeting | Some strong Quill matchups, but worse against trained Large; adding recovery did not fix that regression. |

These observations guided selection and are not independent evidence of strength.
No training code, trained weights or teachers supply either candidate's policy.
The existing vectorized physics is reused for exploratory evaluation only.

## Quill Speed: a bounded correction in six neurons

The [constructor](../handwritten/quill_speed.py) retains Quill's original square
approximation, aiming and direction gate. Six new first-layer hinges add extra
anticipation on slower approaches, within the same 256-parameter budget.

With normalized distance `d = (x − 15)/180`, vertical velocity `v = vy/6`, and
incoming speed deficit `w = 13 − |vx|`, define:

```text
F(t,v) = ReLU(t + 1.5v) − ReLU(t − 1.5v) − 1.5v
C = F(d,v) + F(0.04w,v) − F(d + 0.04w,v)
```

For incoming balls, this adds a signed, bounded correction to the landing
forecast. It vanishes at maximum speed, zero vertical velocity and contact.
Its magnitude is at most `20·|vy|` court units, also bounded by distance and
speed deficit. It is a timing heuristic, not exact division by horizontal speed.
All three F terms expand to ordinary ReLU neurons and fixed linear weights.

## Quill Geometry: compile more of the physics

The [constructor](../handwritten/quill_geometry.py) writes five dense layers:

| Layer | Role |
| --- | --- |
| `6 → 43` | Detect distance from each legal incoming/outgoing speed; carry position signals. |
| `43 → 729` | Gate speed-specific square-interpolation banks and direction-specific carriers. |
| `729 → 6` | Form forecasts for our contact and the opponent's contact, plus paddle and aiming signals. |
| `6 → 467` | Fold wall crossings and apply a piecewise-linear geometric recovery map. |
| `467 → 3` | Compare the combined paddle target with our position and apply a movement deadband. |

**Discrete flight time.** Horizontal speeds are integers from 5 to 13. Returns
start at a contact plane, 330 units from the other plane. If `s = |vx|`, every
pre-contact position in that flight satisfies:

```text
ticks_left = distance/s + ceil(330/s) − 330/s
```

The ceiling correction is a constant for each speed, calculated when writing
the weights. Speed-5 serves start 165 units away, and both 165/5 and 330/5 are
integers. No division or ceiling operator appears in the submitted graph.

**Vertical travel.** A matching speed bank approximates distance × vertical
velocity with the difference-of-squares identity used by Quill, on a finer
piecewise-linear grid. Inactive banks vanish. On 26,892 tested states per flight
direction, the maximum unfolded forecast error was below **0.751 court units**.
The wall-folding hinges support up to five crossings in the reachable range.

**Recovery.** While the ball heads away, forecast the opponent's contact height.
Paddle bounds constrain its legal return offsets. The constructor enumerates
possible integer vertical velocities under maximum-speed return physics, reflects
their destinations, and centers their range. A one-unit interpolation of this
map is compiled into ReLU hinges. No table lookup or geometric search runs during
play. Near-wall contact can therefore call for recovery away from center.

Recovery and shot aiming still approximate lower-speed return choices. The
network is stateless, and a single minimax target does not solve every opponent
or guarantee a catch. More accurate interception alone did not produce the best
results; recovery mattered in these experiments.

## Fresh evaluation

Both models were frozen before the saved **128 fresh shared serve sets** were
drawn. The sets do not overlap the exploratory or public development seeds. All
nine models then played **9,216 games** through the unchanged single-threaded
ONNX Runtime Web CPU/WASM engine, using `pong-v3`. There are 1,280 rallies per
matchup. The [saved evaluation](quill-evaluation.json) includes the complete plan,
seeds, identities, per-seed scores and bootstrap intervals.

| Model | Rally score vs Quill | Paired contests vs Quill: W / D / L | Rally score vs trained Large |
| --- | ---: | ---: | ---: |
| Original Quill | — | — | 35.35% |
| Quill Speed | 57.62% | 79 / 20 / 29 | 43.32% |
| Quill Geometry | **67.70%** | **110 / 8 / 10** | **46.29%** |

Rally score counts drawn rallies as half a win. Geometry's contest score against
Quill is **89.06%**, with a 95% whole-serve-set bootstrap interval of
**83.98–93.75%**. Its rally-score interval is **65.47–69.96%**.

A direct win over Quill does not establish improvement across other opponents.
For that comparison, every model faces the same six original opponents:
**Large, Small, Counter97, Attack75, Predict75 and Follow**. Quill and both
proposed variants are excluded from this common field.

| Model | Mean common-field rally score | Mean common-field contest score |
| --- | ---: | ---: |
| Original Quill | 79.69% | 85.87% |
| Quill Speed | 75.98% | 84.96% |
| Quill Geometry | **82.93%** | **89.39%** |

Geometry improves over Quill by **3.24 percentage points in rally score**
(95% paired bootstrap interval **2.47–4.02 points**) and **3.52 points in contest
score** (**1.95–5.08 points**). Resampling retains each whole shared serve set,
both sides and all six opponents together.

**Geometry is the supported broader improvement, using a larger resource
budget. Speed is a matchup-specific Small alternative.** Although Speed beats
Quill directly and improves against Large, its common-field rally score falls
by 3.70 points (95% interval **−4.50 to −2.94**); its contest change is
−0.91 points (**−2.54 to +0.65**). The fresh evaluation therefore does not support
calling Speed an overall Quill replacement.

Geometry still trails trained Large: **31 wins, 40 draws and 57 losses** in
paired contests, a 39.84% contest score (95% interval **33.19–46.88%**).
Its 46.29% rally score has interval **43.95–48.63%**. This is an improvement over
Quill's Large matchup, not evidence that Geometry surpasses Large.

Final model SHA-256 identities:

```text
quill_speed     2d44a094f4c69efe8e0aae9509658a44b6765bf03913e9c34f4a0395a20563c5
quill_geometry  46a4bc79c474354844723d029199859c443bb1fe6ef2e47c197c391e07bdb932
```

The separate research evaluation uses a saved 128-set plan. Normal model PR
publication still requires the repository's independently drawn 512-set gate.
Serve sets are the independent sampling units; replaying a match adds no samples.
Strength claims are limited to this opponent field, rules and scoring protocol.

## Reproduce

Install the [experiment requirements](../experiments/quill/requirements.txt) in
a local Python environment. The saved exploration report records these package
versions. The playable constructors need only NumPy and ONNX, without PyTorch.

```sh
python -m pip install -r experiments/quill/requirements.txt
python handwritten/quill_speed.py --out /tmp/quill_speed.onnx
python handwritten/quill_geometry.py --out /tmp/quill_geometry.onnx
cmp /tmp/quill_speed.onnx public/models/quill_speed.onnx
cmp /tmp/quill_geometry.onnx public/models/quill_geometry.onnx

python experiments/quill/run.py --report /tmp/quill-exploration.json
python -m unittest discover -s experiments/quill -p 'test_*.py'

npm run models:check
npm test
npm run league:check
npm run build
npx playwright test
```

The research WASM wrapper partitions independent serves without changing game,
inference or scoring code. Its [equivalence record](quill-wasm-equivalence.json)
checks exact serial/parallel results and Markdown on two frozen serve sets.
The full saved plan can also be replayed by the original serial league command.

Extract the recorded plan and replay it without drawing replacement serves:

```sh
python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('docs/quill-evaluation.json').read_text())
Path('/tmp/quill-plan.json').write_text(json.dumps(report['plan']))
PY

node --import tsx experiments/quill/wasm-evaluate.ts \
  --plan /tmp/quill-plan.json --workers 4 --output /tmp/quill-holdout-replay

python experiments/quill/summarize.py \
  --league /tmp/quill-holdout-replay/league.json \
  --plan /tmp/quill-plan.json --out /tmp/quill-evaluation-replay.json

# Equivalent serial replay through the original runner:
npm run league -- --plan /tmp/quill-plan.json --output /tmp/quill-serial-replay
```

The parallel wrapper saves validated checkpoints and resumes the same plan if
interrupted. Checkpoint resume was checked against the complete holdout and
produced byte-identical league JSON, Markdown and plan files. Replay requires
the exact nine-model roster and evaluator recorded in the plan; future roster
or rules changes require isolating the historical inputs or drawing a new
evaluation plan for the changed benchmark.

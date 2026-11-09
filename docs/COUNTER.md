# Counter: prediction, aiming and recovery under 100 parameters

Counter is a separate handwritten model with **97 parameters** and **81 MACs
per decision**. Its layout is **6 → 6 → 4 → 3 → 3**, with ReLU hidden layers
and three linear action scores. Predict and Attack remain unchanged.

The [constructor](../handwritten/counter.py) combines Attack's bounded
forecast and corner selection with Quill's direction gate and center recovery.
Manual designs at 75 parameters or fewer sacrificed too much interception
quality. The extra layer allows these behaviors to coexist while staying below
100 stored parameters, including every zero and bias. This does not establish
that 75 parameters is an absolute limit.

All weights come from fixed formulas and manual play evaluation. There is no
training, optimizer, automated fitting, learned checkpoint or runtime helper.

## What the layers do

| Layer | Role |
| --- | --- |
| 6 → 6 | Two forecast hinges, ball and paddle carriers, an outgoing-velocity gate and an opponent-position ramp. |
| 6 → 4 | Combine and gate the forecast; carry the paddle, incoming indicator and corner-selection hinge. |
| 4 → 3 | Remove the carry offset, fold the prediction at the walls and form the paddle target. |
| 3 → 3 | Choose movement or stay using a 1.1-unit deadband. |

While the ball approaches, the unfolded forecast in court units is:

```text
ball_y + sign(ball_vy) · min(0.4 · distance, 18 · |ball_vy|) + 0.75 · ball_vy
```

Here `distance` is distance to our contact plane. Counter extends Attack's
12-tick lead cap to 18 ticks; the 0.75-tick term accounts approximately for
discrete contact. The network reflects the prediction at normalized heights
`±0.95`. Its incoming target retains Attack's aiming law:

```text
s = clip(80 · opponent_input + 4 · ball_y_input, −1, 1)
target = (83 / 78) · reflected_forecast + (19 / 312) · s
```

The ramp chooses the corner away from the opponent, with ball position favoring
a cross-court return when the opponent is nearly centered. The aiming constants
come from the game's contact-offset law and the 26-tick flight at speed 13.

While the ball departs, `ReLU(vx_input)` suppresses the forecast. A separate
incoming indicator removes the positive carry offset only on approach, so a
suppressed forecast becomes center rather than a false wall reflection.
Coupled aiming weights shrink the remaining opponent-dependent recovery offset
as horizontal speed rises. Its magnitude is at most
`(190 / 39) · (1 − speed / 13)` court units: below three units at speed 5,
and zero at speed 13.

## Fresh evaluation

After freezing the model, 128 new shared serve sets were drawn and all seven
models played both sides: **5,376 games** under `pong-v3`, using ONNX Runtime Web
CPU/WASM with one thread. No weights changed after seeing these results. The
[saved summary](counter-evaluation.json) includes the complete plan, seeds,
model hashes, evaluator identity and counts.

Against Attack, Counter won **127 paired contests, drew one and lost none**.
It won **1,130 of 1,280 rallies**, with 150 losses and no draws: **88.3%**.

Comparison against all five opponents shared by both models:

| Opponent | Attack rally score | Counter rally score |
| --- | ---: | ---: |
| Follow | 77.5% | 85.4% |
| Quill | 9.1% | 15.2% |
| Small | 69.5% | 56.8% |
| Large | 3.5% | 15.7% |
| Predict | 92.0% | 91.0% |
| Mean | **50.3%** | **52.8%** |

Rally score counts draws as half a win. The mean gain is **2.5 percentage
points**, with an approximate 95% paired bootstrap interval of **1.5–3.4
points**, resampling whole shared serve sets with all five opponents and both
sides together. These results support a strong direct advantage over Attack
and a modest improvement across this field, with a substantial regression
against Small. Quill and Large still beat Counter overall.

Counter uses horizontal velocity for recovery, but still approximates flight
time instead of dividing distance by actual incoming speed. Difficult fast
returns and multiple real wall bounces remain weaknesses. Bot results do not
establish strength against human players.

## Reproduce

The standalone constructor needs NumPy and ONNX, not PyTorch:

```sh
python handwritten/counter.py --out public/models/counter.onnx
npm run models:check
npm run league
npm test
npm run league:check
npm run build
npx playwright test
```

Expected model SHA-256:

```text
5f7a480b98c982cf403f2f8849fac1e5e02503ff12e6b0b5d1e377856534e35c
```

Replay the separate 128-set evaluation without overwriting development results:

```sh
node --input-type=module <<'JS'
import { readFileSync, writeFileSync } from 'node:fs'
const { plan } = JSON.parse(readFileSync('docs/counter-evaluation.json', 'utf8'))
writeFileSync('/tmp/counter-plan.json', JSON.stringify(plan))
JS
npm run league -- --plan /tmp/counter-plan.json --output /tmp/counter-replay
```

The saved plan requires the exact seven-model roster and evaluator. If future
models are added, isolate the saved roster as shown in
[Attack's historical replay](ATTACK.md#reproduce). Future weight changes
require a new evaluation seed set.

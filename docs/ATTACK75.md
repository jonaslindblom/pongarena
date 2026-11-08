# Attack75: corner aiming in 75 parameters

Attack75 is an aggressive handwritten fork of [Predict75](PREDICT75.md).
Both retain the **6 → 6 → 3 → 3** ReLU layout, **75 parameters** and **63 MACs
per decision**. Predict75's constructor and ONNX bytes are unchanged.

Attack75 trades two of Predict75's forecast hinges for opponent-dependent
shot placement. The [constructor](../handwritten/attack75.py) writes fixed
formulas. Designs were adjusted manually and compared through play; no
optimizer, training data, learned checkpoint or automated weight search was
used. Every decision runs through the ordinary dense ONNX layers.

## How it plays

The first two neurons compute `ReLU(d + v)` and `ReLU(d − v)`, where
`d = (ball_x − 15) / 180` is distance to our paddle's contact plane and
`v = ball_vy / 6`. Their difference minus `v` equals
`sign(v) · min(d, |v|)` for legal distances. This produces the unfolded
forecast, expressed here in court units:

```text
ball_y + sign(ball_vy) · min(0.4 · distance, 12 · |ball_vy|) + 0.75 · ball_vy
```

Two more neurons carry ball height and our paddle position. The final pair
forms `s = clip(80 · opponent_input + 4 · ball_y_input, −1, 1)`. It selects
the corner away from the opponent. When the opponent is almost centered,
the ball-position term favors a cross-court return instead of a flat shot.

The second layer reflects the forecast at the walls, at normalized heights
`±0.95`, and compares our paddle with this offset target:

```text
target = (83 / 78) · reflected_forecast + (19 / 312) · s
```

The aiming coefficients come from the game's return law: at horizontal speed
13, a return reaches the other paddle after 26 ticks, and each unit of
contact offset shifts its unfolded landing by `26 · 9 / 15 = 78 / 5` units.
Moving slightly off the predicted landing therefore sends the ball toward
the selected corner. The output keeps Predict75's 1.1-unit movement deadband.

## Evaluation

After freezing the model, 128 fresh shared serve sets were drawn and all six
models played both sides: **3,840 games** under `pong-v3`, using ONNX Runtime
Web CPU/WASM with one thread. The saved [evaluation summary](attack75-evaluation.json)
includes the exact plan, seeds, evaluator identity, model hashes and shot counts.
These seeds were not used for subsequent weight changes.

Attack75 won **128 of 128 paired contests against Predict75**. Across their
1,280 rallies it won 1,123, lost 71 and drew 86: a **91.1% rally score**,
counting a draw as half a win.

Against the same four other opponents:

| Opponent | Predict75 rally score | Attack75 rally score |
| --- | ---: | ---: |
| Follow | 60.3% | 78.7% |
| Quill | 3.6% | 9.5% |
| Small | 66.6% | 70.0% |
| Large | 0.0% | 3.8% |
| Mean | **32.6%** | **40.5%** |

The mean improvement is **7.9 percentage points**, with an approximate 95%
paired bootstrap interval of **6.8–8.9 points**, resampling whole serve sets
with all opponents and both sides together.

A second WASM replay measured returns against those same four opponents.
Straight shots (`vy = 0`) fell from **62.3% to 20.0%**. Projected corner
landings (`y < 28` or `y > 132`, after wall reflection) rose from **18.3% to
34.9%**. These are pooled fractions of each model's own returns; match lengths
and shot opportunities differ. They describe placement, not guaranteed winners.

Attack75 still loses to Quill and Large overall. Its bounded forecast is an
approximation, it folds only one predicted wall bounce, and it retains
Predict75's lack of horizontal-speed awareness and center recovery while the
ball travels away. The results measure these bots and serves, not human strength.

## Reproduce

Only NumPy and ONNX are needed to rebuild the weights; PyTorch is not required:

```sh
python handwritten/attack75.py --out public/models/attack75.onnx
npm run models:check
npm run league
npm test
npm run league:check
npm run build
npx playwright test
```

Expected Attack75 SHA-256:

```text
646d1a1365e3e307bd4431fc73930c5efa19de16b7cc22bbb9200608c2b720f8
```

To replay the separate 128-set evaluation without overwriting the development
standings, copy its historical six-model roster into a temporary directory.
This keeps the saved comparison reproducible after new models are added:

```sh
node --input-type=module <<'JS'
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
const { plan } = JSON.parse(readFileSync('docs/attack75-evaluation.json', 'utf8'))
const directory = mkdtempSync(join(tmpdir(), 'attack75-replay-'))
mkdirSync(join(directory, 'models'))
for (const { id } of plan.entrants)
  copyFileSync(`public/models/${id}.onnx`, join(directory, 'models', `${id}.onnx`))
const roster = plan.entrants.map(({ id, name }) => ({ id, name, submittedBy: 'jonaslindblom' }))
writeFileSync(join(directory, 'roster.ts'), `export const presets = ${JSON.stringify(roster)}`)
writeFileSync(join(directory, 'plan.json'), JSON.stringify(plan))
execFileSync(process.execPath, ['--import', 'tsx', 'scripts/generate-league.ts',
  '--plan', join(directory, 'plan.json'), '--models-dir', join(directory, 'models'),
  '--roster', join(directory, 'roster.ts'), '--output', join(directory, 'results')], { stdio: 'inherit' })
console.log(directory)
JS
```

The evaluator rejects changed models or rules. Future tuning must use new
evaluation serves rather than treating this published seed set as a holdout.

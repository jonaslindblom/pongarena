# Predict: prediction in 75 parameters

Predict is a handwritten dense network with layout **6 → 6 → 3 → 3**.
The two hidden layers use ReLU; the output contains three linear action
scores. It stores **75 parameters** and uses **63 MACs per decision**, including
the weights that are zero. It belongs to the Small class.

## What the neurons do

The first hidden layer uses four ReLU hinges to approximate the product of
horizontal distance and vertical velocity. Two more neurons carry ball height
and paddle position. The forecast assumes a horizontal speed of 13 and includes
a small correction for discrete contact timing.

The second hidden layer carries the forecast minus paddle position and two
hinges for a predicted wall bounce. The output combines them as
`z − 2·ReLU(z − .95) + 2·ReLU(−z − .95)`, compares the reflected target with
the paddle position, and applies a 1.1-unit movement deadband.

All decisions are represented by the submitted dense weights. No prediction,
reflection or other position helper runs outside the network during play.

## Tradeoffs

Predict uses 54% fewer parameters than Quill. It anticipates incoming balls
and reflects one predicted wall bounce, but its coarse flight approximation
loses accuracy. It ignores the opponent position and horizontal velocity input,
so it has no opponent-dependent aiming or direction-dependent center recovery.
It can keep following a forecast while the ball travels away.

This is the frozen 75-parameter candidate from the model-size exploration.
Bot evaluation does not establish that it beats human players.
Its weights are unchanged; the checked-in league evaluates them under the
current `pong-v3` centered-serve rules.

## Reproduce

Install the Python dependencies from `training/requirements.txt`, then run:

```sh
python handwritten/predict.py --out public/models/predict.onnx
npm run models:check
```

The constructor preserves the original exploration graph name
`Predictive_ungated2` and metadata so its exact model identity stays unchanged:

```text
0d405fe22a0a5f568c1d23d76356893e12aae1ef1c9a7f26150e8f35520b3cad
```

The displayed name is Predict. Its submitter is recorded in the model roster.

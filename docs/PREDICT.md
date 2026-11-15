# Predict: prediction in 75 parameters

Predict is a programmed dense network with layout **6 → 6 → 3 → 3**.
The two hidden layers use ReLU; the output contains three linear action
scores. It stores **75 parameters** and uses **63 MACs per decision**, including
the weights that are zero. It belongs to the Small class.

## What the neurons do

The first hidden layer uses four ReLU hinges to approximate the product of
distance along the court to our paddle and across-court velocity. Two more neurons
carry the ball's across-court position and our paddle position. The forecast assumes
an along-court speed of 13 units per tick and includes
a small correction for discrete contact timing.

The second hidden layer carries the forecast minus paddle position and two
hinges for a predicted wall bounce. Positions use the
[centered, normalized coordinates](RULES.md#observations-and-actions), with side
walls at ±0.95. The output folds the normalized forecast `z` as
`z − 2·ReLU(z − .95) + 2·ReLU(−z − .95)`, compares the reflected target with
the paddle position, and applies a 1.1-unit movement deadband.

All decisions are represented by the submitted dense weights. No prediction,
reflection or other position helper runs outside the network during play.

## Tradeoffs

Predict uses 54% fewer parameters than Quill. It anticipates incoming balls
and reflects one predicted wall bounce, but its coarse flight approximation
loses accuracy. It ignores the opponent position and along-court velocity input,
so it has no opponent-dependent aiming or direction-dependent center recovery.
It can keep following a forecast while the ball travels away.

This is the frozen 75-parameter candidate from the model-size exploration.
Bot evaluation does not establish that it beats human players.
Its weights are unchanged; the checked-in league evaluates them under the
current `pong-v3` centered-serve rules.

## Reproduce

Use the [shared NumPy/ONNX setup and byte-for-byte rebuild check](../programmed/README.md)
for [Predict's constructor](../programmed/predict.py). No training dependencies are needed.

The constructor preserves the original exploration graph name
`Predictive_ungated2` and all weights. The descriptive method label was updated
in the [metadata revision](../experiments/model-metadata.json), which records
the old and new file hashes and verifies an identical model payload.
The current model SHA-256 is:

```text
2253a9dc4b461771f94e6592109c3a2113aa773059e3f46ab8bb1230f5db2bf6
```

The displayed name is Predict. Its submitter is recorded in the model roster.

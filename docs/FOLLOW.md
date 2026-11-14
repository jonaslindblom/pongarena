# Follow: understand every weight

Follow is a playable **13-parameter** handwritten network: `6 → 1 → 3`, with one Tanh neuron and 9 multiply-accumulates per decision. It follows the ball's current position. No training or optimizer is involved.

The six inputs describe the ball's position and velocity along and across the court, then our paddle and the opponent's paddle across the court. Follow uses only the ball's and our paddle's across-court positions. In the formula below, `ballY` and `paddleY` are the [centered, normalized inputs](RULES.md#observations-and-actions), not raw court units; increasing Y moves right on screen.

```text
error = ballY − paddleY
h = tanh(error)
scores = [−h, δ, h]              # left, stay, right
δ = tanh(1.5 / 80) ≈ 0.0187478
```

Tanh preserves the error's sign while bounding its size. The largest score wins. The positive stay score keeps the paddle still when it is within roughly 1.5 simulation units of the ball, avoiding overshooting. The stored bias is the float32 value of the formula above.

Every stored value is visible here:

| Tensor | Values | Count |
| --- | --- | ---: |
| First-layer weights | `[0, 1, 0, 0, −1, 0]` | 6 |
| First-layer bias | `[0]` | 1 |
| Output weights | `[−1, 0, 1]` (one per action) | 3 |
| Output biases | `[0, δ, 0]` | 3 |

Only five values are nonzero, but **all 13 count** toward the size budget. A single hidden neuron shares the position error across the outputs; a direct `6 → 3` dense layer would store 21 parameters. Follow is the smallest architecture permitted by this arena's dense contract.

Zero weights ignore distance to the paddle, velocity and the opponent. Follow cannot anticipate a bounce or aim a return. That makes it a useful teaching model and a modest opponent, rather than a claim that handwritten weights outperform training.

## Rebuild it

Use the [shared NumPy/ONNX setup and byte-for-byte rebuild check](../handwritten/README.md)
for the standalone [constructor](../handwritten/follow.py). It needs no training
code, checkpoints or PyTorch. During play, the ONNX weights perform the entire policy.

[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) explores constructing neural weights directly. Follow demonstrates the basic idea with a policy small enough to explain completely; it does not reproduce the article's LLM experiments.

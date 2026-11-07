# Follow: understand every weight

Follow is a playable **13-parameter** handwritten network: `6 → 1 → 3`, with one Tanh neuron and 9 multiply-accumulates per decision. It follows the ball's current position. No training or optimizer is involved.

The six inputs are ball X, ball Y, velocity X, velocity Y, our paddle Y and the opponent's paddle Y. Positions are centered and normalized; simulator Y runs left to right on the displayed court.

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

From the repository root, using Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/pip install numpy==2.5.4 onnx==1.23.2
.venv/bin/python handwritten/follow.py --out public/models/follow.onnx
npm run league
npm test
```

The [constructor](../handwritten/follow.py) is standalone and deterministic. It needs no training code, checkpoints or PyTorch. During play, the ONNX weights perform the entire policy.

[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) explores constructing neural weights directly. Follow demonstrates the basic idea with a policy small enough to explain completely; it does not reproduce the article's LLM experiments.

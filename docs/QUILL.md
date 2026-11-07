# Quill: handwritten neural weights

Quill plays Pong with **163 parameters** in a `6 → 12 → 4 → 3 → 3` ReLU network. Its weights are written directly from geometric formulas. It uses no optimizer, training dataset or learned checkpoint. Every decision runs through the same dense ONNX operations as the trained models.

The [constructor](../handwritten/quill.py) is independent of the PyTorch training code. Small and Large learn from scripted teachers, without Quill's weights, outputs or matches during training and selection. The frozen models meet in the final league so the two methods can be compared.

The project is inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab. Their article explores reasoning directly about neural weights, including constructing game-playing networks. Quill is a small, inspectable example of expressing a control policy in weights. Its design was evaluated and adjusted through play; it is not a claim to reproduce the article's zero-shot experiments.

## What the network computes

The six inputs describe the ball's position and velocity, our paddle and the opponent's paddle. Each player receives the same view with its own paddle on the left. The app rotates this view into the vertical court.

| Layer | Job |
| --- | --- |
| 6 → 12 | Build ReLU hinges for a piecewise linear square approximation; carry ball height, paddle positions and travel direction. |
| 12 → 4 | Combine the hinges into an approximate landing forecast and choose recovery behavior when the ball travels away. |
| 4 → 3 | Reflect the forecast at a wall and shift the contact target to aim away from the opponent. |
| 3 → 3 | Turn target error into left/stay/right scores, with a small deadband to prevent overshooting. |

The key product is distance × sideways velocity. A square approximation supplies it through the identity `(d + a)² − (d − a)² = 4da`. ReLU hinges make the approximation expressible using ordinary dense layers. Another pair of hinges folds the predicted landing at a wall.

All 163 stored values count toward its Small class budget, including zeros and biases. Inference uses 141 multiply-accumulates per decision. The constructor's Python arithmetic only writes those values; no geometric helper runs alongside the network during play.

## Rebuild it

From the repository root, use Python 3.12 and install only NumPy and ONNX:

```sh
python3.12 -m venv .venv
.venv/bin/pip install numpy==2.5.4 onnx==1.23.2
.venv/bin/python handwritten/quill.py --out public/models/quill.onnx
npm run league
npm test
```

The constructor includes its ONNX exporter and requires no training files or PyTorch. Rebuilding is deterministic and replaces the chosen output file.

Quill approximates travel time using the maximum horizontal speed and models one wall reflection. Its small network cannot represent exact planning for every position or speed. Compare it with the trained models in the [reference league](../public/league.md); those results measure this opponent field and serve set rather than universal strength.

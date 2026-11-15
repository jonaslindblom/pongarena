# Quill: programmed neural weights

Quill plays Pong with **163 parameters** in a `6 → 12 → 4 → 3 → 3` ReLU network. Its weights come from geometric formulas that Codex developed and revised using simulated match feedback. Every decision runs through the same dense ONNX operations as the trained models.

Codex inspected existing models and constructors, wrote candidate formulas, evaluated them in simulated matches and revised the design from those results. This was iterative design optimization. No gradient training or conventional numerical parameter-fitting algorithm was used to produce Quill's weights. **Programmed** describes how the weights were constructed, not an absence of feedback or optimization.

The [constructor](../programmed/quill.py) is independent of the PyTorch training code. [Small](../experiments/quill-training/README.md) uses the same architecture and learns through distillation from Large, PPO and CMA-ES. Large learns from scripted teachers and game outcomes. Neither uses programmed models' weights, outputs or matches during training and selection. The frozen models meet in the final league so the methods can be compared.

The project is inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab. Their article explores reasoning directly about neural weights, including constructing game-playing networks. Quill is a small, inspectable example of expressing a control policy in weights. Its feedback-guided development differs from the article's zero-shot experiments.

## What the network computes

The six inputs describe the ball's position and velocity, our paddle and the opponent's paddle. Both players receive an own-paddle-relative view: along-court position and velocity are mirrored as needed, so negative along-court velocity always means approach. Across-court position and velocity retain the same left-to-right direction for both players. See the [coordinate and observation contract](RULES.md#coordinates-and-player-identity).

| Layer | Job |
| --- | --- |
| 6 → 12 | Build ReLU hinges for a piecewise linear square approximation; carry the ball's across-court position, paddle positions and travel direction. |
| 12 → 4 | Combine the hinges into an approximate landing forecast and choose recovery behavior when the ball travels away. |
| 4 → 3 | Reflect the forecast at a wall and shift the contact target to aim away from the opponent. |
| 3 → 3 | Turn target error into left/stay/right scores, with a small deadband to prevent overshooting. |

The key product is distance along the court to our paddle × across-court velocity. A square approximation supplies it through the identity `(d + a)² − (d − a)² = 4da`, where `d` and `a` are scaled versions of that distance and velocity. ReLU hinges make the approximation expressible using ordinary dense layers. Another pair of hinges folds the predicted landing at a wall.

All 163 stored values count toward its Small class budget, including zeros and biases. Inference uses 141 multiply-accumulates per decision. The constructor's Python arithmetic only writes those values; no geometric helper runs alongside the network during play.

## Rebuild it

Use the [shared NumPy/ONNX setup and byte-for-byte rebuild check](../programmed/README.md).
The constructor includes its ONNX exporter and needs only NumPy and ONNX.
Rebuilding is deterministic and replaces the chosen output file. It reproduces
the final network, not the Codex iteration and match evaluation that produced its design.

Quill approximates travel time using the maximum along-court speed and models one wall reflection. Its small network cannot represent exact planning for every position or speed. Compare it with the trained models in the [reference league](https://jonaslindblom.github.io/pongarena/league.md); those results measure this opponent field and serve set rather than universal strength.

# PongArena

Play Pong against tiny neural networks, watch them compete, and inspect their neurons as they play. Everything runs locally in the browser through ONNX Runtime Web.

PongArena compares two ways to build neural controllers: **Quill**, a network with 163 handwritten parameters, and **Small** and **Large**, trained independently with PyTorch. Small learns from scripted teachers; Large adds reinforcement learning from game outcomes against scripts and past learned policies. Quill expresses prediction, wall reflection and aiming directly in dense weights. The trained models use no handwritten weights, targets or opponents during training or selection; Quill enters the final comparison after they are frozen.

**Follow** is a playable teaching example with just **13 handwritten parameters and 9 MACs per decision**. [Understand every weight](docs/FOLLOW.md), then inspect its single hidden neuron in the arena.

**Predict75** uses **75 handwritten parameters and 63 MACs per decision**, with a
`6 → 6 → 3 → 3` layout. It predicts landings and a wall bounce while omitting
Quill’s recovery and opponent-dependent aiming. [Construction and limitations](docs/PREDICT75.md).

The project is inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), which explores reasoning directly about neural weights. [How Quill works](docs/QUILL.md) explains its construction and limitations. The league explores how this small handwritten policy compares with learned ones.

[League standings](public/league.md) · [Train the models](training/README.md) · [Contribute a model](CONTRIBUTING.md) · [Rules](docs/RULES.md)

## Run and play

Use Node.js 22 and npm:

```sh
npm ci
npm run dev
```

Open the localhost URL and press **P** to play. **P** also pauses, resumes, or plays again after game over. Move with **A/D** or **←/→**; on a touch screen, hold the left or right half of the court. **R** restarts with the same serves. **F** toggles full screen; **Esc** exits it.

New matches and **Play again** use fresh serve sets. Each rally starts at the center of the circle, alternates the receiving end, and launches at a gentle 10–15° angle toward it. Opening serves reach the paddle without a wall bounce; returns build the difficulty. Restart replays the current set for practice.

Faint arrows and shaded halves show the touch controls directly on the court. Hold either half to move, slide across to change direction, and release to stop. The held half lights up gently; no separate movement buttons take space from the game.

Choose an opponent or switch to **Watch models** to compare two networks. **Brain** starts open and shows live activations. **Models** provides reference ranks, downloads and local ONNX imports. **Weights** explains Follow through an interactive view of its actual weights and decisions. A match contains five rallies, each lasting at most 1,600 ticks at 30 decisions per second.

To play from a phone on the same Wi-Fi, use `npm run dev -- --host 0.0.0.0` and open the printed Network URL.

## League

The deployed League page uses verified PR evaluation results: **512 unique shared serve sets**, sampled after model hashes are frozen and played from both sides. That is **10,240 games** across Follow, Predict75, Quill, Small and Large. Saved seeds and hashes make the results replayable. The checked-in [standings](public/league.md) remain a fast 16-seed development benchmark: **320 games**.

The current `pong-v3` benchmark uses centered, gentle serves with the same frozen model files; its scores are separate from earlier `pong` and `pong-v2` results.

Standings show approximate 95% score intervals, calculated by resampling entire serve sets with both sides and all opponents together. No model retraining is needed.

Each pair of five-rally games forms one contest. Its winner earns one league point; a draw earns half. Equal totals share rank. Compare scores within the same opponent field and serve set.

Local leagues can include imported models. Size classes apply equally to handwritten and trained weights:

| Class | Maximum parameters | Maximum MACs per decision |
| --- | ---: | ---: |
| Small | 256 | 1,024 |
| Large | 65,536 | 65,536 |

Both limits apply; Small models may also compete in Large. See the [model contract and scoring](docs/RULES.md).

```sh
npm run models:check  # validate every model and finite WASM output
npm run league        # regenerate public/league.json and public/league.md
npm run league:check  # replay and verify both committed files
```

Model PRs include regenerated development standings. A separate [submission gate](.github/workflows/model-evaluation.yml) validates models and evaluates the full proposed roster on fresh serves. Retries reuse the saved plan; model changes get a new set. On merge, Pages verifies and publishes the matching results. See [contributing](CONTRIBUTING.md).

## Develop

```sh
npm test
npm run build
npm run preview
```

The static build is written to `dist/`. Python is only needed to [train models](training/README.md) or rebuild [Follow](docs/FOLLOW.md#rebuild-it), [Quill](docs/QUILL.md#rebuild-it) and [Predict75](docs/PREDICT75.md#reproduce).

- `src/`: game, browser interface, model validation and league logic.
- `public/models/`: the bundled ONNX files.
- `public/league.json` and `public/league.md`: reference results.
- `handwritten/`: standalone handwritten constructors for Follow and Quill.
- `training/`: standalone PyTorch imitation, reinforcement learning and evaluation.
- `scripts/`: league generation for local development and CI.
- `tests/`: game, model and competition checks.

See [publishing notes](docs/PUBLISHING.md) for GitHub Actions and Pages.

## License

PongArena’s original source code, documentation and bundled ONNX model weights
are licensed under the [MIT License](LICENSE). Third-party dependencies and
assets retain their own licenses. Contributions, including model submissions,
are accepted under MIT; see [contributing](CONTRIBUTING.md).

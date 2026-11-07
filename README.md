# PongArena

Play Pong against tiny neural networks, watch them compete, and inspect their neurons as they play. Everything runs locally in the browser through ONNX Runtime Web.

PongArena compares two ways to build neural controllers: **Quill**, a network with 163 handwritten parameters, and **Small** and **Large**, trained independently with PyTorch using scripted teachers. Quill expresses prediction, wall reflection and aiming directly in dense weights. The trained models use no Quill weights, targets or opponents during training or selection; Quill enters the final comparison after they are frozen.

**Follow** is a playable teaching example with just **13 handwritten parameters and 9 MACs per decision**. [Understand every weight](docs/FOLLOW.md), then inspect its single hidden neuron in the arena.

The project is inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), which explores reasoning directly about neural weights. [How Quill works](docs/QUILL.md) explains its construction and limitations. The league explores how this small handwritten policy compares with learned ones.

[League standings](public/league.md) · [Train the models](training/README.md) · [Contribute a model](CONTRIBUTING.md) · [Rules](docs/RULES.md)

## Run and play

Use Node.js 22 and npm:

```sh
npm ci
npm run dev
```

Open the localhost URL and press **P** to play. **P** also pauses, resumes, or plays again after game over. Move with **A/D** or **←/→**; on a touch screen, hold the left or right half of the court. **R** restarts with the same serves. **F** toggles full screen; **Esc** exits it.

Choose an opponent or switch to **Watch models** to compare two networks. **Brain** starts open and shows live activations. **Models** provides reference ranks, downloads and local ONNX imports. **Weights** explains Follow through an interactive view of its actual weights and decisions. A match contains five rallies, each lasting at most 1,600 ticks at 30 decisions per second.

To play from a phone on the same Wi-Fi, use `npm run dev -- --host 0.0.0.0` and open the printed Network URL.

## League

The League page opens with the checked-in [standings](public/league.md). Follow, Quill, Small and Large face each other on 16 fixed serve sets, with both side assignments: **192 games**. [league.json](public/league.json) contains every result, seed and model hash.

Each pair of five-rally games forms one contest. Its winner earns one league point; a draw earns half. Equal totals share rank. These public seeds are a reproducible development benchmark; use fresh seeds for independent performance claims.

Local leagues can include imported models. Size classes apply equally to handwritten and trained weights:

| Class | Maximum parameters | Maximum MACs per decision |
| --- | ---: | ---: |
| Small | 256 | 1,024 |
| Large | 65,536 | 65,536 |

Both limits apply; Small models may also compete in Large. See the [model contract and scoring](docs/RULES.md).

```sh
npm run league        # regenerate public/league.json and public/league.md
npm run league:check  # replay and verify both committed files
```

Model PRs include regenerated standings. The [Check and rank workflow](.github/workflows/check.yml) verifies them, displays the table in its summary, and uploads results for review.

## Develop

```sh
npm test
npm run build
npm run preview
```

The static build is written to `dist/`. Python is only needed to [train models](training/README.md) or rebuild [Follow](docs/FOLLOW.md#rebuild-it) and [Quill](docs/QUILL.md#rebuild-it).

- `src/`: game, browser interface, model validation and league logic.
- `public/models/`: the four bundled ONNX files.
- `public/league.json` and `public/league.md`: reference results.
- `handwritten/`: standalone handwritten constructors for Follow and Quill.
- `training/`: independent PyTorch training and evaluation using scripted teachers.
- `scripts/`: league generation for local development and CI.
- `tests/`: game, model and competition checks.

See [publishing notes](docs/PUBLISHING.md) for GitHub Actions and Pages. A source and model-weight license must be selected before publication.

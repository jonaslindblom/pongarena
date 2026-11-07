# PongArena

Play Pong against small neural networks and deterministic scripts, watch them compete, and inspect a model's live activations. Inference runs locally on the CPU through ONNX Runtime Web. Models, fonts and WebAssembly assets are served with the app.

[Reference league](public/league.md) · [Contribute a model](CONTRIBUTING.md) · [Game and competition contract](training/COMPETITION.md)

## Run locally

Use Node.js 22 and npm:

```sh
npm ci
npm run dev
```

Open the localhost URL printed by Vite. The arena starts ready for **You play** against **Tracker**; press **Play** to begin.

To try a phone on the same Wi-Fi, run `npm run dev -- --host 0.0.0.0` and open the printed Network URL.

## Play, watch and inspect

- Move with **A/D** or **←/→**. On a touch screen, hold the left or right half of the court.
- Press **R** or **Restart** to immediately replay the same five serves. **Pause** stops the match; **Full screen** focuses on the court.
- Choose another opponent, or use **Play** from the league and **Play against** from a model card.
- Switch to **Watch models** to choose both controllers. The initial matchup is Prism Small v4 versus Quill Plus v4.
- Open **Show brain** to see a weight model's hidden activations and chosen action while it plays.
- **Models** holds downloads and local ONNX imports. **Guide** explains the game and includes the optional device test.

Classic v4 runs at 30 decisions per second with five rallies per match. Rallies last at most 1,600 ticks; automated players also draw when an exact state repeats. Both paddles use the same observations and movement limits. Scripts are labelled separately from weight models.

## League

The **League** page loads the checked-in [reference standings](public/league.md) immediately. All six bundled weight models and three scripts compete on the same 16 fixed serve sets, with both side assignments: 1,152 games in total. Full scores, controller hashes and seeds are in [league.json](public/league.json).

A contest combines two five-rally games. Its winner earns one league point, a tie earns half, and equal totals share rank. Scores describe this field and seed set; the published seeds are a development benchmark, not a private training holdout.

Expand **Run your own league** to compare imported models or change the field. **All controllers** includes scripts and weights; the size classes accept weights only:

| Class | Maximum parameters | Maximum MACs per decision |
| --- | ---: | ---: |
| Small | 256 | 1,024 |
| Large | 65,536 | 65,536 |

Both limits apply. Quill Plus and Tracker fit Small; the other bundled weight models fit Large. Small models may compete in Large; handwritten and trained weights follow the same contract.

```sh
npm run league        # regenerate public/league.json and public/league.md
npm run league:check  # replay the league and verify both committed files
```

The included GitHub Actions workflow runs tests, regenerates the league, checks the committed snapshot, and uploads standings for review. Model PRs include both refreshed files; the workflow does not push changes. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Model contract

Submit one opset 18 ONNX file, at most 256 KiB, with float32 `observation` input `[1, 6]` and `action_scores` output `[1, 3]`. Use a sequential dense network with zero to four hidden layers, uniform ReLU or Tanh activation, and a linear output. Graph structure, finite parameters and resource usage are validated.

The simulation's legacy actions `up, stay, down` display as `left, stay, right`. The [competition contract](training/COMPETITION.md) gives the exact interface; the [standalone Classic v4 package](training/v4/README.md) contains the game and reference implementations.

## Development and research

```sh
npm test
npm run build
npm run preview
```

The static build is written to `dist/`. For phone measurements, use a production preview with `-- --host 0.0.0.0`; open **Guide → Device test** and compare reports with matching workload versions and model hashes. Browser emulation does not establish physical-device performance.

[Training instructions](training/README.md), the [model catalog](training/catalog.json) and [retained evidence](training/evidence.md) document the active models and historical experiments. Retired model bytes remain available for compatibility tests and reproducing old comparisons.

## Publishing

[Publishing notes](docs/PUBLISHING.md) cover free public GitHub Actions compute, the manual Pages workflow and publishing the main branch while keeping local checkpoints private. A source and model-weight license still needs to be selected before release.

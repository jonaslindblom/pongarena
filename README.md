# PongArena

**How small can a neural network be and still play Pong?**

Play against tiny neural networks, watch them compete, and see their neurons fire as they make each move. Everything runs on your device in the browser with ONNX Runtime Web.

PongArena is open source. Submit your weights to enter the league.

[**Play in your browser**](https://jonaslindblom.github.io/pongarena/) · [Explore the findings](https://jonaslindblom.github.io/pongarena/findings/) · [Build a model](CONTRIBUTING.md)

![PongArena showing a match and the neural network’s live activations](docs/media/pongarena.png)

## What happens when the architecture stays the same?

**Quill, Small and OneShot share the same 163-parameter ReLU network: `6 → 12 → 4 → 3 → 3`.** Quill's programmed weights are calculated from deliberately designed formulas, developed with Codex and refined through play. Small learns through distillation from Large, reinforcement learning and direct weight search. OneShot's weights came from a single model reply given the game rules and a tactical brief. Programmed, trained and prompted describe how the weights are produced. All three run as neural networks, receiving the same six observations, choosing from the same three actions and using the same dense ONNX operations.

[**Findings**](https://jonaslindblom.github.io/pongarena/findings/) connects those methods to frozen models and reproducible match results. Across six initial training runs and subsequent refinements, none of the training experiment's saved candidates beat Quill or Counter on the held-out comparisons. That makes learning a strong controller at this size an interesting challenge. It does not establish an architecture limit: Quill's own weights demonstrate what the network can express, and the much larger trained model is competitive with the programmed opponents. See the [training experiment](experiments/quill-training/README.md) for methods, selection rules and results.

The [prompting experiment](https://jonaslindblom.github.io/pongarena/findings/oneshot/) asks a different question: can a model write working weights directly, without match feedback? Explore OneShot's prompt and decisions, or [rebuild the exact weights and replay the results](experiments/one-prompt-weights/follow-up/README.md). These are a few controlled attempts, not a general reliability estimate for prompting.

## League models

The roster spans a 13-parameter ball follower through a 50,819-parameter learned policy. Every model uses the same observations and action contract.

| Model | How it is made | Layer widths | Parameters | MACs / decision |
| --- | --- | --- | ---: | ---: |
| [Follow](docs/FOLLOW.md) | Programmed ball follower | `6 → 1 → 3` | 13 | 9 |
| [Predict](docs/PREDICT.md) | Programmed landing and bounce prediction | `6 → 6 → 3 → 3` | 75 | 63 |
| [Attack](docs/ATTACK.md) | Programmed prediction and opponent-aware corner aiming | `6 → 6 → 3 → 3` | 75 | 63 |
| [Counter](docs/COUNTER.md) | Programmed prediction, corner aiming and center recovery | `6 → 6 → 4 → 3 → 3` | 97 | 81 |
| [Quill](docs/QUILL.md) | Programmed prediction, recovery and aiming | `6 → 12 → 4 → 3 → 3` | 163 | 141 |
| [Small](experiments/quill-training/README.md) | Trained with Large distillation, PPO and CMA-ES | `6 → 12 → 4 → 3 → 3` | 163 | 141 |
| [Large](training/README.md) | Trained with imitation learning and PPO | `6 → 128 → 128 → 128 → 128 → 3` | 50,819 | 50,304 |
| [OneShot](experiments/one-prompt-weights/follow-up/README.md) | Prompted with game rules and a tactical brief | `6 → 12 → 4 → 3 → 3` | 163 | 141 |

All stored weights and biases count, including zeros. A MAC is a multiply-accumulate. Programmed weights do not supply training labels, opponents or checkpoint-selection results for the learned models; they meet after the models are frozen. League results compare these particular controllers and serves; they do not establish performance against human players.

The project is inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). Explore [Follow’s 13 parameters](https://jonaslindblom.github.io/pongarena/how-it-thinks/), or [switch off a Counter neuron](https://jonaslindblom.github.io/pongarena/how-it-thinks/counter/) to see how its decision changes. Both lessons run the actual ONNX weights and show their inputs, activations and moves.

## Play and explore

- **Jump in:** the arena opens with Quill playing Counter and keeps rematching while untouched. Choose **Take over & play** to start a fresh game from the bottom paddle, with a three-second countdown. Interacting stops automatic rematches. Watch links autoplay their selected models; human match links wait for you to start.
- **Play:** choose an opponent, click the playing field, then move your mouse or trackpad left/right to steer without holding the button, even outside the field. Click outside the field or press **Esc** to release mouse control, or use **A/D** or **←/→** to take control. On a phone, hold either half of the court, slide across to change direction, and release to stop.
- **Watch models:** put two networks head to head. Open **Brain** to inspect their live activations.
- **Findings:** compare programmed, trained and prompted weights, with the evidence and limitations beside the results.
- **How it thinks:** follow the computation through actual weights, or switch off a neuron and watch the decision change.
- **Models:** inspect layer layouts, see linked submitters, download the exact ONNX files, or import your own locally.
- **League:** compare published standings or run a local tournament with imported models.

**P** starts, pauses, resumes or plays again. **R** restarts with the same serves. **F** toggles full screen; **Esc** exits it.

Human games are first to three points; draws lead to another serve. Spectator matches have five rallies, each lasting up to 90 seconds, with explained transitions for time-limit or repeating-loop draws. The League retains five rallies and its 1,600-tick rally limit. All modes run at 30 decisions per second. Serves start at the center, alternate ends and use gentle, varied angles; returns build the difficulty. New matches draw fresh serves, while restart replays the current set for practice.

### Share a match or lesson

Arena links accept `?model=counter` to choose a human opponent, or
`?mode=watch&model=counter&opponent=quill` to watch two models.
The explainers and findings have direct page URLs: [Follow](https://jonaslindblom.github.io/pongarena/how-it-thinks/),
[Counter](https://jonaslindblom.github.io/pongarena/how-it-thinks/counter/),
[training findings](https://jonaslindblom.github.io/pongarena/findings/) and
[prompting findings](https://jonaslindblom.github.io/pongarena/findings/oneshot/).
Older hash links to these pages still work. Use `#play`, `#league`, `#models`
or `#rules` at the site root for the other tabs.
Add `&fullscreen=1` to open the court in a viewport-filling layout. Watch links
start automatically; human match links wait for Play (or P). Model and fullscreen
changes update the URL for copying. Imported models are available only in browsers where they were saved;
unavailable linked models fall back to bundled models with a notice.
Human games show a three-second countdown before every serve.

## Run locally

Use **Node.js 26 LTS** and npm. The tested version is in [`.nvmrc`](.nvmrc);
with nvm, run `nvm install` and `nvm use` from the repository root.
Python is needed only for training or reconstructing weights.

```sh
git clone https://github.com/jonaslindblom/pongarena.git
cd pongarena
npm ci
npm run dev
```

Open the URL printed by Vite. To play from a phone on the same Wi-Fi, run `npm run dev -- --host 0.0.0.0` and open the printed Network URL.

```sh
npm run models:check  # validate ONNX graphs, budgets and finite WASM output
npm run typecheck     # check app, scripts and test types
npm test              # game, model and league checks
npm run league:check  # replay the generated development standings
npm run build         # production files in dist/
npm run preview       # serve the production build locally
```

Browser tests exercise the production app on desktop and mobile:

```sh
npx playwright install chromium
npm run test:e2e
```

## Submit a model

[Open a model PR](CONTRIBUTING.md) with just one ONNX file containing a short description and creation method. CI generates the model card and league results. Training scripts and constructors are optional. Community model cards credit the submitter with a link to their GitHub profile.

The contract is float32 `observation [1, 6] → action_scores [1, 3]`, through one to five dense layers with uniform ReLU or Tanh hidden activations and a linear output. Files may be at most 256 KiB.

| Class | Maximum parameters | Maximum MACs / decision |
| --- | ---: | ---: |
| Small | 256 | 1,024 |
| Large | 65,536 | 65,536 |

Both limits apply. Small models can also enter Large. CI validates every submission’s graph, resource counts and runtime output before evaluation; see the [full contract](docs/RULES.md).

## How the league works

Published standings use **512 fresh shared serve sets**, drawn after model hashes are frozen. Every pair plays from both sides, for **28,672 games** with eight models. Retries reuse their saved serve plan. On merge, Pages publishes the verified results matching those exact models and rules.

A contest combines two five-rally games with swapped sides. A win earns one league point; a draw earns half. Approximate 95% score intervals resample whole serve sets, preserving both sides and all matchups together. Compare results within the same opponent field, rules and serve set.

Automatically generated development standings use **16 fixed public serve sets / 896 games** with the eight core models for fast, reproducible checks. Local builds and CI discover ONNX files and generate the catalog and standings; these derived files are not committed. They are separate from the larger published evaluation. Current rules are `pong-v3`. See [scoring and reproducibility](docs/RULES.md).

## Find your way around

- [Contributing](CONTRIBUTING.md): submit a model and run the PR checks.
- [Training](training/README.md): independent PyTorch recipes and evaluation.
- [Training experiment](experiments/quill-training/README.md): the search for a learned controller in Quill's exact architecture.
- [Prompting experiment](experiments/one-prompt-weights/follow-up/README.md): prompts, generated weights and frozen evaluations.
- [Programmed models](programmed/README.md): rebuild Follow, Predict, Attack, Counter and Quill with NumPy and ONNX.
- [Rules](docs/RULES.md): observations, actions, physics, model limits and scoring.
- [Publishing](docs/PUBLISHING.md): GitHub Actions, required checks and Pages.
- [Screenshots](docs/media/README.md): refresh the README image from the running app.
- `src/`, `scripts/`, `tests/`, `e2e/`: the app, evaluators and automated checks.

## License

PongArena’s original source code, documentation and bundled ONNX model weights use the [MIT License](LICENSE). Contributions, including model submissions, are accepted under MIT. Third-party dependencies and assets retain their own licenses; their [notices](public/THIRD_PARTY_NOTICES.txt) ship with the website.

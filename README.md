# PongArena

A local React and TypeScript arena for watching Pong policies play and inspecting the action scores behind each move. ONNX Runtime Web runs inference on the CPU. Models, fonts, and WebAssembly assets are served from the app.

## Run locally

With Node.js 22 and npm installed, run these commands from the project directory.

```sh
npm ci
npm run dev
```

Open the localhost URL printed by Vite. The default match pairs **Pacer v4** and **Apprentice v4** under the frozen **Classic v4** rules. The active arena uses ordinary Pong movement and contact-position aiming.

To try it on a phone connected to the same Wi-Fi, start Vite with:

```sh
npm run dev -- --host 0.0.0.0
```

Open the **Network** URL printed by Vite on the phone. The default `127.0.0.1` URL is accessible only on the computer running the app. Playing, importing and ranking have been checked on an ordinary HTTP LAN origin, including its SHA-256 fallback; mobile layout uses browser emulation. Physical-device performance and touch feel still need testing.

## Play and inspect

- Every match uses a top/bottom court, including matches between automated controllers. Pick an opponent for **Player 02**, then press **Play yourself** to face it (if P2 is Human, this uses Tracker). Or press **Play against** on any library card. Your paddle is at the bottom in solo play. Use A/D or the left/right arrows. With two humans, P1 is at the bottom with A/D and P2 is at the top with the arrows. W/S and up/down arrows remain aliases.
- **Attacker**, **Defender**, and **Interceptor** are deterministic scripts, marked in amber. They are browser ports of the Classic v4 evaluation opponents in `training/scripts/pace_study.py` (`pace13-v9-scaled`), using the same six observations, physics, and action limits as models. Scripts can play either side against a human, model, or another script. Their inspector shows the chosen action as 1, with the others 0; these are not learned confidence scores.
- Weight models have distinct **Nano**, **Small**, **Standard**, or **Open** badges with parameter counts. These are the same budget classes used by Rankings. Cards retain architecture and training-method details, so a hand-edited network and a learned network are both identifiable as weight models.
- On touch screens, hold the left half of the court to move left or the right half to move right; lift to stop. In fullscreen, these thumb zones cover the screen apart from the game controls. Holding both directions stops the paddle; lifting one thumb restores the remaining direction. Two humans use the bottom and top halves. The Left/Right buttons also support held input. Releasing, cancelling, pausing or restarting clears held controls.
- Press **Full screen** on the court to focus on the game with scores, pause, restart and exit controls. Page scrolling and gestures are locked until you exit. The game uses native fullscreen when available and fills the browser viewport otherwise. Press **Exit** or Escape to return to the workspace at your previous scroll position.
- Use **P1** or **P2** in Action scores to inspect that player's last decision. Pause, step, or scrub the timeline to see the exact scores and actions recorded during play. Selecting an event in the match log jumps to its frame.
- The simulation makes 30 decisions per second; court animation interpolates between those states at the display's refresh rate. Paused and stepped frames show exact recorded states. Animation does not change physics or model decisions, and switching away from the page pauses play.
- Opening or reloading the arena draws a fresh seed. Its five rallies start in five different bands across the court, and both paddles receive serves moving left and right. A point is worth one; a draw scores zero. Rallies have a 1,600-tick limit. Matches between models and/or scripts also end a rally when its exact ball/paddle state repeats after a return. Human matches use the full limit. The log distinguishes loops from timeouts. **Restart** repeats the set. **New serves** draws a fresh seed. Choosing different players keeps the current set for comparison.
- **Show brain** below playback controls opens a live activation graph for one paddle. Choose P1 or P2 to inspect a weight model: hidden neurons show real post-activation values, and three output nodes highlight the sampled decision. Samples update up to four times per second and retain their decision tick when paused. Positive values are green, negative values orange; brightness is scaled within each layer to emphasize small values. Small layers use larger nodes, and hidden-layer counts show how many values are nonzero. Human and scripted controllers have no neural activations. Inspection uses a separate session and preserves the competing model’s original bytes and scores.
- **Import model** assigns a local ONNX file to either paddle. Imported models are kept in this browser's IndexedDB when available. The model library contains downloadable baseline files. The field guide documents the submission contract.

## Measure a phone baseline

Use a production build for comparable performance measurements:

```sh
npm run build
npm run preview -- --host 0.0.0.0 --port 4173
```

Keep the Mac and phone on the same Wi-Fi, then open Vite's **Network** URL in Safari on iPhone or Chrome on Android. Open **Device test**, enter the phone model, label the visit, and press **Run device test**. Keep the page visible and the orientation unchanged. The test measures warmed-up inference followed by 15 seconds of animated play. Press **Copy report** and paste the result into the chat. If clipboard access is unavailable on the local HTTP connection, select and copy the report text manually.

Save the first report before reloading. Reload the same page and run again with the repeat-visit label to compare loading with a potentially warmer browser cache. The visit label is self-reported; the app does not claim to detect an empty cache. Keep battery-saving settings and other device conditions consistent across comparisons.

Reports include navigation-relative arena/library readiness, loading phases, observed startup visibility, browser/viewport information, a source fingerprint, model hashes, and the versioned workload. The benchmark uses the existing CPU WASM runtime and ordinary game/rendering code. Inference statistics describe **batch-average decision times**, not individual-call tail latency. Animation statistics describe browser animation-callback intervals, not physical display FPS or touch-to-display latency; intentional between-rally pauses are counted separately. Hidden or interrupted tests are cancelled rather than reported as complete.

Workload `rally-device-v2` measures Tracker, Pacer v4, and Prism Small v4. It replaces v1’s retired Strategist model; compare inference timings only between reports with matching workload versions and model hashes.

Startup measurements belong to that page load. Running the test again without reloading only repeats inference and animation. Development-server reports are labelled as development and should not be used for before/after comparisons; the production source fingerprint is calculated at build time. Phone viewport emulation is useful for UI checks but is not physical-device performance evidence. The Device test addition preserves the existing startup sequence so the initial phone reports can serve as a baseline before optimization.

## Rally balance

Two competitive paddles should be able to sustain a rally under pressure and still win the point through shot placement. To check a rules change, keep the attacking opponent and serve seeds fixed, swap sides, and compare a baseline defender with one that wins more often across several opponents. Track how often a rally ends before the second return, the median number of returns in rallies that end in a point, and both loop and timeout draws. Match score determines the winner.

Classic v4 remains frozen. Every return increases speed toward the opposing paddle by one, capped at 13; contact position controls sideways deflection. Its [gradual-speed study](training/archive/README.md) and `/training/pace-study.html` demonstrate placement/recovery scoring between strong diagnostic scripts, with a median 11 returns in scored attacker mirrors. This does not establish learned tactical mastery or human enjoyment.

The intended direction is simple Pong: ordinary movement, contact-position aiming, and gradual acceleration. A slower speed ramp without boost remains a separate, unevaluated candidate. Substantially faster paddles can make recenter-and-intercept defense sufficient, so speed must be evaluated together with reachable shots. The [burst rule study](training/results/burst-study/rules-README.md) and `/training/burst-rules.html` are retained as historical experiments; Burst is no longer an active arena mode.

The earlier [offensive-balance experiment](training/archive/README.md) compared steeper returns and contact-motion deflection with adaptive opponents and four equally budgeted learned-weight runs. Its archived `/training/offense-study.html` report explains why those settings were not promoted: the milder angle still admits guaranteed defense, while stronger candidates favor short first-receiver points between strong scripts.

## Model interface

The simulation and ONNX contract keep their original coordinates so existing models and training results remain compatible. The display rotates the court: simulation X runs between the top and bottom paddles, and simulation Y runs left to right. The legacy action names **up, stay, down** mean **left, stay, right** on screen. The standalone training package and historical studies describe simulation coordinates.

Submit one opset 18 ONNX file (up to 256 KiB) containing a plain dense neural network with embedded float32 weights and biases:

| Rules | `observation` input | `action_scores` output |
| --- | --- | --- |
| Classic v4 (`pong-v4`) | float32 `[1, 6]` | float32 `[1, 3]`: up, stay, down |

The six observations are centered ball X and Y, horizontal and vertical velocity divided by six, own paddle Y, and opponent paddle Y. The observing player's court is mirrored horizontally to place its paddle on the left. Output scores must be finite; the first maximum wins ties. Both players decide from the same pre-move state. Archived eight-input/five-output Burst files cannot be imported into the active game; previously saved Burst files remain in browser storage but are not loaded.

The active contract permits one to five sequential dense `Gemm` layers: zero to four hidden layers and a linear three-score output. Every hidden layer uses the same standard `Relu` or `Tanh` activation. Each dense layer has its own embedded fixed weight matrix and bias; `alpha` and `beta` are one, and the input is not transposed. MatMul/Add instruction graphs, branches, skip connections, shared parameters, computed weights, feature extraction, and input/output processing are rejected. Knowledge must reside in the dense weights and biases. The importer validates this structure as well as shapes, finite parameters, the 65,536-MAC budget, and file limits. Cards show the actual architecture and validated resource counts.

## Compare weights

The current nine-model roster, maintained training commands and checkpoint provenance are documented in [training/README.md](training/README.md). Robust Strong v4 is the retained full-size learned candidate; Pacer and Apprentice remain the default matchup. Completed experiments are compressed in the local training archive and public evidence downloads.

**Refined Nano v4** is the trained 163-parameter `6→16→3` Tanh preset (144 MACs). Its frozen small-step CMA candidate improves the unchanged starting model's six-opponent rally score from **26.62% to 30.46%** on 256 independent paired seeds (+3.84 points, 95% CI +3.04 to +4.61). Plain Nano remains stronger overall. On a separate matched test, Quill Plus averages **48.90% versus Refined Nano's 30.02%** and wins their direct matchup 434 games to 46, with 32 draws; Refined Nano performs better against Attacker. `/training/refined-nano-v4.html` and its evidence download retain the selected weights, protocols, checkpoint, comparisons and runtime checks.

**Quill Plus v4** is the retained handwritten 163-parameter `6→12→4→3→3` ReLU model (141 MACs), created without training or automated fitting. It changes only four of original Quill's weights. A longer-range forecast correction and speed-dependent recovery raise the equal-opponent rally score from **54.93% to 57.53%** against ten references on a separate 128-seed test; draws count as half a point. The paired gain's seed-bootstrap 95% interval is **+1.98 to +3.22 percentage points**. It improves against Pacer, Robust Strong, Prism Small and the adaptive scripts, but regresses against Plain Nano and Vector Nano. Directly against original Quill it records 562/603/115 rallies, with no demonstrated advantage. Quill Plus is the only active Quill preset; original Quill is retained as a test fixture and in the archived study evidence. The [constructor](training/scripts/quill_plus_dense_v4.py) and `/training/quill-plus-v4.html` report retain the development and confirmation evidence.

**Plain Nano v4** is the trained 447-parameter `6→16→12→8→3` ReLU preset (408 MACs). It publishes the plain-Nano study's preselected `precise-cma-7151` candidate, refined through local CMA weight search after student training. In the retained confirmation study it won 332 games, lost 145, and drew 35 against unchanged Pacer across 256 paired seeds. Defender remains a major weakness; this does not establish full-size playing strength. The built-in file is `public/models/plain-nano-v4.onnx`, SHA-256 `fd59c8f08927f3e0b9a196919ad80674dcd92bdc44087065e3d0a9fa54406e3c`.

**Rankings** runs a local round robin with a selected roster, common serve seeds and both side assignments. Each pair of five-rally games awards one league point for a win, half for a tie, and zero for a loss. Equal totals share rank; unfinished leagues remain unranked. Cancel, export the result, or repeat the exact benchmark. Rules, contracts, roster hashes and seeds identify each result; results from different benchmarks are not interchangeable ratings.

Scripts are available as reference opponents in the Arena, but do not enter weight-budget leagues or receive a model size class.

The active neural-network contract is `observation-6/action-scores-3/dense-mlp-v3`. Rankings saved under the earlier two-hidden-layer or general-graph contracts are not restored. Classic v4 physics is unchanged.

Classes require both limits: **Nano** ≤512 parameters / 2,048 MACs, **Small** ≤2,048 / 8,192, **Standard** ≤20,480 / 65,536, and **Open** ≤65,536 / 65,536. Smaller models may compete up. Training method is irrelevant. These are provisional budgets, and existing graph/file limits still apply. The [competition protocol](training/COMPETITION.md) explains scoring and reproducibility. A public leaderboard will need an independent evaluator; local browser results are not verified public standings.

## Verify

```sh
npm test
npm run build
npm run preview
```

The tests cover the game rules, perspective symmetry, finite output and tie handling, and the upload contract. The production build is written to `dist/`.

Script conformance tests compare every reachable contact path, observation probes, and 15 complete matches against independently generated Python fixtures. Regenerate them with `.venv/bin/python training/scripts/export_scripted_fixtures.py` after reviewing any change to the evaluated policies.

## Train your own weights

Start with the standalone [Classic v4 game specification](training/v4/README.md), downloadable as `/training/pong-v4.zip`. The immutable package retains its historical general-graph export section; the current [dense-network competition contract](training/COMPETITION.md) supersedes that section for active submissions. It includes the engine, a headless TypeScript example, a dependency-free Python reference, and deterministic match fixtures. Copy `training/v4/` to train independently of the app; run `python3 training/v4/check.py` to check the Python port. Enable loop detection only for fixed deterministic stateless policies; sampled training actions can escape a repeated position.

[Current training instructions](training/README.md) cover the nine retained models, full-size DAgger training, native evaluation on explicit seeds, and the handcrafted model constructors. [The model catalog](training/catalog.json) records deployed file hashes and selected checkpoint provenance. Robust Strong and Plain Nano still have weaknesses against adaptive scripts; retained study evidence is summarized in [training/evidence.md](training/evidence.md).

Completed experiments, superseded checkpoints and their original source snapshots are compressed in the [local experiment archive](training/archive/README.md). Public experiment reports describe historical results and contracts; they are not the current training instructions. Frozen v1/v2/v3 and Burst packages remain for compatibility tests.

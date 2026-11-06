# Rally Lab

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

Classic v4 remains frozen. Every return increases speed toward the opposing paddle by one, capped at 13; contact position controls sideways deflection. Its [gradual-speed study](training/results/pace-study/README.md) and `/training/pace-study.html` demonstrate placement/recovery scoring between strong diagnostic scripts, with a median 11 returns in scored attacker mirrors. This does not establish learned tactical mastery or human enjoyment.

The intended direction is simple Pong: ordinary movement, contact-position aiming, and gradual acceleration. A slower speed ramp without boost remains a separate, unevaluated candidate. Substantially faster paddles can make recenter-and-intercept defense sufficient, so speed must be evaluated together with reachable shots. The [burst rule study](training/results/burst-study/rules-README.md) and `/training/burst-rules.html` are retained as historical experiments; Burst is no longer an active arena mode.

The earlier [offensive-balance experiment](training/results/offense-study/README.md) compared steeper returns and contact-motion deflection with adaptive opponents and four equally budgeted learned-weight runs. Its archived `/training/offense-study.html` report explains why those settings were not promoted: the milder angle still admits guaranteed defense, while stronger candidates favor short first-receiver points between strong scripts.

## Model interface

The simulation and ONNX contract keep their original coordinates so existing models and training results remain compatible. The display rotates the court: simulation X runs between the top and bottom paddles, and simulation Y runs left to right. The legacy action names **up, stay, down** mean **left, stay, right** on screen. The standalone training package and historical studies describe simulation coordinates.

Submit one opset 18 ONNX file (up to 256 KiB) containing a plain dense neural network with embedded float32 weights and biases:

| Rules | `observation` input | `action_scores` output |
| --- | --- | --- |
| Classic v4 (`pong-v4`) | float32 `[1, 6]` | float32 `[1, 3]`: up, stay, down |

The six observations are centered ball X and Y, horizontal and vertical velocity divided by six, own paddle Y, and opponent paddle Y. The observing player's court is mirrored horizontally to place its paddle on the left. Output scores must be finite; the first maximum wins ties. Both players decide from the same pre-move state. Archived eight-input/five-output Burst files cannot be imported into the active game; previously saved Burst files remain in browser storage but are not loaded.

The active contract permits one to three sequential dense `Gemm` layers: zero, one, or two hidden layers and a linear three-score output. Every hidden layer uses the same standard `Relu` or `Tanh` activation. Each dense layer has its own embedded fixed weight matrix and bias; `alpha` and `beta` are one, and the input is not transposed. MatMul/Add instruction graphs, branches, skip connections, shared parameters, computed weights, feature extraction, and input/output processing are rejected. Knowledge must reside in the dense weights and biases. The importer validates this structure as well as shapes, finite parameters, the 65,536-MAC budget, and file limits. Cards show the actual architecture and validated resource counts.

## Compare weights

**Rankings** runs a local round robin with a selected roster, common serve seeds and both side assignments. Each pair of five-rally games awards one league point for a win, half for a tie, and zero for a loss. Equal totals share rank; unfinished leagues remain unranked. Cancel, export the result, or repeat the exact benchmark. Rules, contracts, roster hashes and seeds identify each result; results from different benchmarks are not interchangeable ratings.

Scripts are available as reference opponents in the Arena, but do not enter weight-budget leagues or receive a model size class.

The active neural-network contract is `observation-6/action-scores-3/dense-mlp-v2`. Rankings saved under the earlier general-graph contract are not restored. Classic v4 physics is unchanged.

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

[PyTorch training and results](training/README.md) documents the current Classic workflow and archived experiments. Candidate compact Classic networks include `6→16→3` (163 parameters), `6→16→16→3` (435), and `6→32→32→3` (1,379). These are architecture budgets, not freshly trained or evaluated policies. Removing energy management and extra actions simplifies the learning problem; tactical capability still needs measurement.

The `pace_study.py` and `train_pace.py` workflow produced **Pacer v4** and **Apprentice v4**, fully learned `6→128→128→3` Tanh networks trained through DAgger. The v1/v2/v3 packages, weights and results remain frozen; those older plain dense weights remain eligible for Classic, but previous results do not establish strength under v4.

The [Classic placement test](training/results/classic-tactics/README.md), visualized at `/training/classic-tactics.html`, finds legal alternatives that turn scripted losses into wins. Matched learned continuations do not turn those local opportunities into an overall stronger player: the selected model improves against the defender but regresses against the attacker. Both failed and selected training attempts, independent replay checks and fresh evaluations are retained; the active models remain unchanged.

**Vector Nano v4** is a `6→28→9→3` ReLU network with 487 hand-constructed parameters and no training. Its opponent-side court-end aiming beats Pacer, Prism Nano, and the three earlier Nano references across three fresh blocks, but loses to Prism Small and both adaptive scripts. Its cap-speed forecast and contact goal remain approximate; see the [Vector dense study](training/results/vector-dense-v4/README.md) and `/training/vector-nano-v4.html`.

**Prism Nano v4** is a catalogued `6→29→9→3` ReLU network with 503 hand-constructed parameters and no training. It beats the Nano references Tracker, Anticipator, and Challenger in all three fresh test blocks, but loses to Pacer and the adaptive scripts. Its approximate anticipation and conservative aiming are encoded entirely in ordinary dense weights. See the [analytic dense study](training/results/analytic-dense-v4/README.md) and `/training/prism-nano-v4.html` for the full results and limits.

**Prism Small v4** extends that approach to a plain `6→42→33→3` ReLU network with 1,815 fixed parameters. It beats Pacer and Prism Nano in all three fresh native test blocks, but still loses heavily to both adaptive scripts. Its hand-constructed cap-speed forecast and coherent contact targets remain limited by lower-speed approximation, movement reachability, and center recovery. The [robust dense study](training/results/robust-dense-v4/README.md) and `/training/prism-small-v4.html` retain those results and qualifications.

The [handcrafted graph investigation](training/results/compact-handcrafted-v4/README.md) and [attacker-counter investigation](training/results/counter-attacker-v4/README.md) are retained as ineligible archives. Their ONNX graphs encoded physics and planning operations rather than plain dense neural networks. Handcrafted Nano, Handcrafted Small, Counter Small, and the older hybrid Strategist are removed from the active catalog; their historical scores are not active dense-network standings.

The archived [Burst size comparison](training/results/burst-study/models/README.md), with curves at `/training/burst-models.html`, trained networks from 229 to 18,309 parameters. Its 501-parameter Tiny beat Tracker under Burst rules, and Focus learned useful rescue timing but not the full placement counter. Those results do not transfer to Classic, and the eight-input/five-output exports are no longer arena presets. Reports, training scripts, weights and the standalone Burst package remain available for reproducing that experiment.

The earlier **Contender** is fully learned: a plain `6→64→64→3` Tanh network with 4,803 learned parameters and no fixed trajectory or feedback layers. Under v3, PPO followed by direct weight search produced an export that passed three reserved blocks against v2 and the attacker, including winning a majority of ordinary five-rally games in every block. The [fully learned investigation](training/results/fully-learned/README.md) retains those historical results; open `/training/fully-learned.html` for its curves.

**Strategist** is an archived hybrid with fixed trajectory features and is ineligible under the dense-only contract. Its [earlier investigation](training/results/training-improvement/README.md) and `/training/improvement.html` retain the v3 diagnosis, separate results and learning curves.

The [balance study](training/results/balance-v2/README.md) explains the matched-serve comparison and why cap 13 was chosen. The report includes both scored-rally lengths and draw rates.

The [timeout and rule study](training/results/rule-study/README.md) provides the measurements behind the v3 deadline and loop decisions, plus comparisons with faster paddles, different ball speeds, and larger paddles. Its original experimental report remains at `/training/rule-study.html`.

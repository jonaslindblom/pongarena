# Rally Lab

A local React and TypeScript arena for watching Pong policies play and inspecting the action scores behind each move. ONNX Runtime Web runs inference on the CPU. Models, fonts, and WebAssembly assets are served from the app.

## Run locally

With Node.js 22 and npm installed, run these commands from the project directory.

```sh
npm ci
npm run dev
```

Open the localhost URL printed by Vite. The default match pairs Pacer v4 and Apprentice v4 under the Pong v4 rules.

To try it on a phone connected to the same Wi-Fi, start Vite with:

```sh
npm run dev -- --host 0.0.0.0
```

Open the **Network** URL printed by Vite on the phone. The default `127.0.0.1` URL is accessible only on the computer running the app. Mobile checks so far use browser emulation; physical-device performance and touch feel still need testing.

## Play and inspect

- Press **Play yourself** to start Human vs Tracker, or set either paddle to **Human** and press **Play match**. Use W/S or the up/down arrows. With two humans, the left player uses W/S and the right uses the arrows.
- On touch screens, drag the court to steer your paddle or hold the Up/Down buttons. One human can drag anywhere on the court; two humans use their respective halves. Touch targeting uses the same two-unit movement limit as keyboard and model actions. Releasing, cancelling, pausing or restarting clears held controls.
- Use **P1** or **P2** in Action scores to inspect that player's last decision. Pause, step, or scrub the timeline to see the exact scores and actions recorded during play. Selecting an event in the match log jumps to its frame.
- The simulation makes 30 decisions per second; court animation interpolates between those states at the display's refresh rate. Paused and stepped frames show exact recorded states. Animation does not change physics or model decisions, and switching away from the page pauses play.
- Opening or reloading the arena draws a fresh seed. Its five rallies start in five different height bands, and both paddles receive rising and falling serves. A point is worth one; a draw scores zero. Rallies have a 1,600-tick limit. Model-only matches also end a rally when its exact ball/paddle state repeats after a return. Human matches use the full limit. The log distinguishes loops from timeouts. **Restart** repeats the set. **New serves** draws a fresh seed. Choosing different players keeps the current set for comparison.
- **Import model** assigns a local ONNX file to either paddle. Imported models are kept in this browser's IndexedDB when available. The model library contains downloadable baseline files. The field guide documents the submission contract.

## Rally balance

Two competitive paddles should be able to sustain a rally under pressure and still win the point through shot placement. To check a rules change, keep the attacking opponent and serve seeds fixed, swap sides, and compare a baseline defender with one that wins more often across several opponents. Track how often a rally ends before the second return, the median number of returns in rallies that end in a point, and both loop and timeout draws. Match score determines the winner.

Pong v4 separates placement from speed. Every return increases horizontal speed by one, capped at 13; contact position controls vertical deflection. The first return has speed 6, and center hits never reset the speed. The [gradual-speed study](training/results/pace-study/README.md) records candidate selection, stronger defensive checks and matched-serve evaluation; open `/training/pace-study.html` for the report and learned models. Human enjoyment still requires actual playtesting.

The earlier [offensive-balance experiment](training/results/offense-study/README.md) compared steeper returns and contact-motion deflection with adaptive opponents and four equally budgeted learned-weight runs. Its archived `/training/offense-study.html` report explains why those settings were not promoted: the milder angle still admits guaranteed defense, while stronger candidates favor short first-receiver points between strong scripts.

## Model interface

Submit one opset 18 ONNX file (up to 256 KiB) with embedded float32 weights. The single input is `observation`, float32 with shape `[1, 6]`. It contains centered ball X and Y, horizontal and vertical velocity divided by six, own paddle Y, and opponent paddle Y. The observing player's court is mirrored horizontally to place its paddle on the left.

The output is `action_scores`, finite float32 values with shape `[1, 3]` ordered up, stay, down. Ties choose the first maximum. Each tick, the two players decide from the same pre-move state.

The supported stateless graphs use Gemm, MatMul, Add, Relu, and Tanh. The importer checks tensor shapes and embedded weights, with limits of 65,536 multiply–accumulates, 256 operators, and 262,144 total intermediate output elements per decision. The starter model is in `public/models/tracker.onnx`.

## Verify

```sh
npm test
npm run build
npm run preview
```

The tests cover the game rules, perspective symmetry, finite output and tie handling, and the upload contract. The production build is written to `dist/`.

## Train your own weights

The standalone [Pong v4 training contract](training/v4/README.md), downloadable as `/training/pong-v4.zip`, includes the frozen engine, a headless TypeScript example, a dependency-free Python reference, and deterministic match fixtures. Copy `training/v4/` to train independently of the app. Run `python3 training/v4/check.py` to verify the Python port. Enable loop detection only for fixed deterministic stateless policies; sampled training actions can escape a repeated position.

[PyTorch training and results](training/README.md) links to the current `pace_study.py` and `train_pace.py` workflow, learning curves, evaluation and standalone exports. **Pacer v4** and **Apprentice v4** are fully learned `6→128→128→3` Tanh networks trained through DAgger; no fixed trajectory calculations are embedded in their inference graphs. Cards show the actual network architecture and link to the report. The six-input/three-score model interface is unchanged. The v1/v2/v3 packages, weights and results remain frozen; old models are interface-compatible but their previous results do not establish strength under v4.

The earlier **Contender** is fully learned: a plain `6→64→64→3` Tanh network with 4,803 learned parameters and no fixed trajectory or feedback layers. Under v3, PPO followed by direct weight search produced an export that passed three reserved blocks against v2 and the attacker, including winning a majority of ordinary five-rally games in every block. The [fully learned investigation](training/results/fully-learned/README.md) retains those historical results; open `/training/fully-learned.html` for its curves.

**Strategist** remains a clearly labelled hybrid comparison, with fixed trajectory features. It does not meet the fully learned requirement. Its [earlier investigation](training/results/training-improvement/README.md) and `/training/improvement.html` retain the v3 diagnosis, separate results and learning curves.

The [balance study](training/results/balance-v2/README.md) explains the matched-serve comparison and why cap 13 was chosen. The report includes both scored-rally lengths and draw rates.

The [timeout and rule study](training/results/rule-study/README.md) provides the measurements behind the v3 deadline and loop decisions, plus comparisons with faster paddles, different ball speeds, and larger paddles. Its original experimental report remains at `/training/rule-study.html`.

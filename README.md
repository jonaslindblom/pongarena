# Rally Lab

A local React and TypeScript arena for watching Pong policies play and inspecting the action scores behind each move. ONNX Runtime Web runs inference on the CPU. Models, fonts, and WebAssembly assets are served from the app.

## Run locally

With Node.js 22 and npm installed, run these commands from the project directory.

```sh
npm ci
npm run dev
```

Open the localhost URL printed by Vite. The default match pairs Contender and Selfplay v2 under the unchanged Pong v3 rules.

## Play and inspect

- Set either paddle to **Keyboard**, then press **Play match**. Use W/S or the up/down arrows. With two humans, the left player uses W/S and the right uses the arrows.
- Use **P1** or **P2** in Action scores to inspect that player's last decision. Pause, step, or scrub the timeline to see the exact scores and actions recorded during play. Selecting an event in the match log jumps to its frame.
- Opening or reloading the arena draws a fresh seed. Its five rallies start in five different height bands, and both paddles receive rising and falling serves. A point is worth one; a draw scores zero. Rallies have a 1,600-tick limit. Model-only matches also end a rally when its exact ball/paddle state repeats after a return. Human matches use the full limit. The log distinguishes loops from timeouts. **Restart** repeats the set. **New serves** draws a fresh seed. Choosing different players keeps the current set for comparison.
- **Import model** assigns a local ONNX file to either paddle. Imported models are kept in this browser's IndexedDB when available. The model library contains downloadable baseline files. The field guide documents the submission contract.

## Rally balance

Two competitive paddles should be able to sustain a rally under pressure and still win the point through shot placement. To check a rules change, keep the attacking opponent and serve seeds fixed, swap sides, and compare a baseline defender with one that wins more often across several opponents. Track how often a rally ends before the second return, the median number of returns in rallies that end in a point, and both loop and timeout draws. Match score determines the winner.

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

The standalone [Pong v3 training contract](training/v3/README.md) includes the frozen engine, a headless TypeScript example, a dependency-free Python reference, and full deterministic match fixtures. Copy `training/v3/` to train independently of the app. Run `python3 training/v3/check.py` to verify the Python port. Enable loop detection only for fixed deterministic stateless policies; sampled training actions can escape a repeated position.

[PyTorch training and results](training/README.md) includes runnable DAgger + PPO self-play scripts, learning curves, held-out tests, and standalone ONNX exports. **Apprentice** and **Selfplay** are available in the Model library. Cards show the actual network architecture and link to the local training report. Pong v3 preserves all v2 physics and the six-input/three-score model interface, extends the rally limit from 800 to 1,600 ticks, and adds early loop draws for model matches. The frozen v1/v2 packages, weights and results remain available.

**Contender** is fully learned: a plain `6→64→64→3` Tanh network with 4,803 learned parameters and no fixed trajectory or feedback layers. PPO followed by direct weight search produced an export that passed three reserved blocks against v2 and the attacker, including winning a majority of ordinary five-rally games in every block. The [fully learned investigation](training/results/fully-learned/README.md) includes reproducible scripts, unsuccessful variants and final results; open `/training/fully-learned.html` for its curves. Rules and upload limits are unchanged.

**Strategist** remains a clearly labelled hybrid comparison, with fixed trajectory features. It does not meet the fully learned requirement. Its [earlier investigation](training/results/training-improvement/README.md) and `/training/improvement.html` retain the v3 diagnosis, separate results and learning curves.

The [balance study](training/results/balance-v2/README.md) explains the matched-serve comparison and why cap 13 was chosen. The report includes both scored-rally lengths and draw rates.

The [timeout and rule study](training/results/rule-study/README.md) provides the measurements behind the v3 deadline and loop decisions, plus comparisons with faster paddles, different ball speeds, and larger paddles. Its original experimental report remains at `/training/rule-study.html`.

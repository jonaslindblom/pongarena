# Rally Lab

A local React and TypeScript arena for watching Pong policies play and inspecting the action scores behind each move. ONNX Runtime Web runs inference on the CPU. Models, fonts, and WebAssembly assets are served from the app.

## Run locally

With Node.js 22 and npm installed, run these commands from the project directory.

```sh
npm ci
npm run dev
```

Open the localhost URL printed by Vite. The default match pairs the evolved Challenger with the hand-written Tracker.

## Play and inspect

- Set either paddle to **Keyboard**, then press **Play match**. Use W/S or the up/down arrows. With two humans, the left player uses W/S and the right uses the arrows.
- Use **P1** or **P2** in Action scores to inspect that player's last decision. Pause, step, or scrub the timeline to see the exact scores and actions recorded during play. Selecting an event in the match log jumps to its frame.
- Opening or reloading the arena draws a fresh seed. Its five rallies start in five different height bands, and both paddles receive rising and falling serves. A point is worth one and a timed-out rally scores zero. **Restart** repeats the set. **New serves** draws a fresh seed. Choosing different players keeps the current set for comparison.
- **Import model** assigns a local ONNX file to either paddle. Imported models are kept in this browser's IndexedDB when available. The model library contains downloadable baseline files. The field guide documents the submission contract.

## Rally balance

Two competitive paddles should be able to sustain a rally under pressure and still win the point through shot placement. To check a rules change, keep the attacking opponent and serve seeds fixed, swap sides, and compare a baseline defender with one that wins more often across several opponents. Track how often a rally ends before the second return, the median number of returns in rallies that end in a point, and the share that reach the 800-tick draw. Match score determines the winner.

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

The standalone [Pong v1 training contract](training/v1/README.md) includes the frozen engine, a headless TypeScript example, a dependency-free Python reference, and full deterministic match fixtures. Copy `training/v1/` to train independently of the app. Run `python3 training/v1/check.py` to verify the Python port.

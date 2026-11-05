# Rally Lab

A local React and TypeScript arena for watching Pong policies play and inspecting the action scores behind each move. ONNX Runtime Web runs inference on the CPU. Models, fonts, and WebAssembly assets are served from the app.

## Run locally

With Node.js 22 and npm installed, run these commands from the project directory.

```sh
npm ci
npm run dev
```

Open the localhost URL printed by Vite. The default match pairs Focus and Large under **Burst preview**. The **Game rules** selector also offers the frozen **Classic v4** game, with its own compatible models and rankings.

To try it on a phone connected to the same Wi-Fi, start Vite with:

```sh
npm run dev -- --host 0.0.0.0
```

Open the **Network** URL printed by Vite on the phone. The default `127.0.0.1` URL is accessible only on the computer running the app. Playing, importing and ranking have been checked on an ordinary HTTP LAN origin, including its SHA-256 fallback; mobile layout uses browser emulation. Physical-device performance and touch feel still need testing.

## Play and inspect

- Press **Play yourself** to start Human vs Tiny in Burst preview (Human vs Tracker in Classic v4), or set either paddle to **Human** and press **Play match**. Use W/S or the up/down arrows. With two humans, the left player uses W/S and the right uses the arrows.
- In Burst preview, hold **Shift** with a direction to move four units instead of two, spending eight energy per boosted tick. Ordinary actions recharge one energy; each serve starts with 96. Two humans use their respective left/right Shift keys.
- On touch screens, drag the court to steer your paddle or hold the Up/Down buttons. Hold **Burst** while dragging or moving to use the same legal boost. One human can drag anywhere on the court; two humans use their respective halves. Releasing, cancelling, pausing or restarting clears held controls.
- Use **P1** or **P2** in Action scores to inspect that player's last decision. Pause, step, or scrub the timeline to see the exact scores and actions recorded during play. Selecting an event in the match log jumps to its frame.
- The simulation makes 30 decisions per second; court animation interpolates between those states at the display's refresh rate. Paused and stepped frames show exact recorded states. Animation does not change physics or model decisions, and switching away from the page pauses play.
- Opening or reloading the arena draws a fresh seed. Its five rallies start in five different height bands, and both paddles receive rising and falling serves. A point is worth one; a draw scores zero. Rallies have a 1,600-tick limit. Model-only matches also end a rally when its exact state repeats after a return, including both energy values in Burst. Human matches use the full limit. The log distinguishes loops from timeouts. **Restart** repeats the set. **New serves** draws a fresh seed. Choosing different players keeps the current set for comparison.
- **Import model** assigns a local ONNX file to either paddle. Imported models are kept in this browser's IndexedDB when available. The model library contains downloadable baseline files. The field guide documents the submission contract.

## Rally balance

Two competitive paddles should be able to sustain a rally under pressure and still win the point through shot placement. To check a rules change, keep the attacking opponent and serve seeds fixed, swap sides, and compare a baseline defender with one that wins more often across several opponents. Track how often a rally ends before the second return, the median number of returns in rallies that end in a point, and both loop and timeout draws. Match score determines the winner.

Burst preview keeps contact placement as the aiming mechanism, increases horizontal speed by half a unit per return up to 13, and adds a limited movement boost. Saving energy can make a later shot reachable; repeated pressure can exhaust that reserve. The [burst rule study](training/results/burst-study/rules-README.md), also at `/training/burst-rules.html`, records matched-serve measurements and a reproducible pressure sequence with defensive counterfactuals. It is a preview for playtesting, not a claim that balance or learned tactics are solved.

Classic v4 remains frozen. Every return increases horizontal speed by one, capped at 13; contact position controls vertical deflection. Its [gradual-speed study](training/results/pace-study/README.md) and `/training/pace-study.html` retain that game's measurements and models. Human enjoyment still requires actual playtesting.

The earlier [offensive-balance experiment](training/results/offense-study/README.md) compared steeper returns and contact-motion deflection with adaptive opponents and four equally budgeted learned-weight runs. Its archived `/training/offense-study.html` report explains why those settings were not promoted: the milder angle still admits guaranteed defense, while stronger candidates favor short first-receiver points between strong scripts.

## Model interface

Submit one opset 18 ONNX file (up to 256 KiB) with embedded float32 weights. Each game has its own contract:

| Rules | `observation` input | `action_scores` output |
| --- | --- | --- |
| Burst preview (`pong-burst-v1`) | float32 `[1, 8]` | float32 `[1, 5]`: up, stay, down, burst up, burst down |
| Classic v4 (`pong-v4`) | float32 `[1, 6]` | float32 `[1, 3]`: up, stay, down |

The first six observations are centered ball X and Y, horizontal and vertical velocity divided by six, own paddle Y, and opponent paddle Y. The observing player's court is mirrored horizontally to place its paddle on the left. Burst appends own energy and opponent energy, each divided by 96. Output scores must be finite; the first maximum wins ties. Both players decide from the same pre-move state. Importing selects the compatible rules; six-input weights do not silently receive a burst adapter.

The supported stateless graphs use Gemm, MatMul, Add, Relu, and Tanh. The importer checks tensor shapes and embedded weights, with limits of 65,536 multiply–accumulates, 256 operators, and 262,144 total intermediate output elements per decision. Cards show the actual architecture, parameter count and compute budget.

## Compare weights

**Rankings** runs a local round robin with a selected roster, common serve seeds and both side assignments. Each pair of five-rally games awards one league point for a win, half for a tie, and zero for a loss. Equal totals share rank; unfinished leagues remain unranked. Cancel, export the result, or repeat the exact benchmark. Rules, contracts, roster hashes and seeds identify each result; results from different benchmarks are not interchangeable ratings.

Classes require both limits: **Nano** ≤512 parameters / 2,048 MACs, **Small** ≤2,048 / 8,192, **Standard** ≤20,480 / 65,536, and **Open** ≤65,536 / 65,536. Smaller models may compete up. Training method is irrelevant. These are provisional budgets, and existing graph/file limits still apply. The [competition protocol](training/COMPETITION.md) explains scoring and reproducibility. A public leaderboard will need an independent evaluator; local browser results are not verified public standings.

## Verify

```sh
npm test
npm run build
npm run preview
```

The tests cover the game rules, perspective symmetry, finite output and tie handling, and the upload contract. The production build is written to `dist/`.

## Train your own weights

Start with the standalone [Burst contract](training/burst/README.md), downloadable as `/training/pong-burst-v1.zip`. It includes the engine, a headless TypeScript example, a dependency-free Python reference, and deterministic match fixtures. Copy `training/burst/` to train independently of the app; run `python3 training/burst/check.py` to check the Python port. Enable loop detection only for fixed deterministic stateless policies; sampled training actions can escape a repeated position.

[PyTorch training and results](training/README.md) links to the size comparison, learning curves, evaluation and standalone exports at `/training/burst-models.html`. Five fully learned Tanh networks span 229–18,309 parameters. **Tiny**, at 501 parameters, is the smallest useful learned starter demonstrated by this recipe; that does not establish tactical mastery or the smallest possible model. The separate **Focus** follow-up improves against the attacking teacher (44 wins / 255 losses / 21 draws), but loses to **Large** (25 / 197 / 98) on the same fresh 320-rally block. Focus is a specialist, not a universal upgrade. No fixed trajectory calculations are embedded in these networks.

The standalone [Classic v4 contract](training/v4/README.md) remains available at `/training/pong-v4.zip`. Its `pace_study.py` and `train_pace.py` workflow produced **Pacer v4** and **Apprentice v4**, fully learned `6→128→128→3` Tanh networks trained through DAgger. The v1/v2/v3 packages, weights and results remain frozen; those older weights are interface-compatible with Classic, but previous results do not establish strength under v4. Burst uses a separate eight-input/five-score contract.

The earlier **Contender** is fully learned: a plain `6→64→64→3` Tanh network with 4,803 learned parameters and no fixed trajectory or feedback layers. Under v3, PPO followed by direct weight search produced an export that passed three reserved blocks against v2 and the attacker, including winning a majority of ordinary five-rally games in every block. The [fully learned investigation](training/results/fully-learned/README.md) retains those historical results; open `/training/fully-learned.html` for its curves.

**Strategist** remains a clearly labelled hybrid comparison, with fixed trajectory features. It does not meet the fully learned requirement. Its [earlier investigation](training/results/training-improvement/README.md) and `/training/improvement.html` retain the v3 diagnosis, separate results and learning curves.

The [balance study](training/results/balance-v2/README.md) explains the matched-serve comparison and why cap 13 was chosen. The report includes both scored-rally lengths and draw rates.

The [timeout and rule study](training/results/rule-study/README.md) provides the measurements behind the v3 deadline and loop decisions, plus comparisons with faster paddles, different ball speeds, and larger paddles. Its original experimental report remains at `/training/rule-study.html`.

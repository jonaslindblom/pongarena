# Contributing to PongArena

Use Node.js 22 and `npm ci`. Import an ONNX file under **Models** to validate it and try it in the arena. Handwritten and trained weights follow the same [model contract](docs/RULES.md).

For a small handwritten example, [Follow](docs/FOLLOW.md) explains all 13 parameters and can be rebuilt with [its constructor](handwritten/follow.py).

Keep the two approaches independent: [Quill's constructor](handwritten/quill.py) writes fixed weights, while [training](training/README.md) starts from random weights and uses scripted teachers. Training and checkpoint selection must not use Quill's weights, outputs or matches; compare frozen models with Quill afterward.

## Submit a model

Submit one opset 18 ONNX file, at most 256 KiB. It must map float32 `observation` `[1, 6]` directly to `action_scores` `[1, 3]` through one to five dense `Gemm` layers. Hidden layers use uniform `Relu` or `Tanh`; the output is linear. All weights and biases must be embedded and finite.

Small allows at most **256 parameters and 1,024 MACs** per decision. Large allows **65,536 of each**. Both limits apply. See [rules, observations and scoring](docs/RULES.md) before exporting.

1. Add `public/models/<id>.onnx` and its entry in `src/roster.ts`. The ID must match the filename without `.onnx`. Describe how the weights were constructed or trained.
2. Include a reproducible constructor or training recipe. Keep checkpoints, datasets and raw runs out of the public tree.
3. Regenerate the league before running checks, since tests verify the bundled roster against its results:

   ```sh
   npm run league
   npm test
   npm run league:check
   npm run build
   ```

4. Include both `public/league.json` and `public/league.md` in the PR. Describe the architecture, validated resource counts, method and evaluation. Identify the opponent field and fresh evaluation seeds for claims of improvement.

The league generator validates every model and records its SHA-256, parameter count and MAC count. Python is optional for the app and league; see [training](training/README.md), [Follow](docs/FOLLOW.md) and [Quill](docs/QUILL.md) to reproduce their respective models.

## PR checks

The [Check and rank workflow](.github/workflows/check.yml) runs tests, replays the league and builds the app. A separate CPU job checks the training physics, runs a tiny training example and verifies its ONNX export. It also regenerates Follow and Quill and requires byte-for-byte matches with their shipped files. The workflow fails if either committed league file is stale, displays regenerated standings in the summary, and uploads them as **league-results**. Review and commit any needed update; CI has read-only repository permissions and does not push changes.

The reference league uses single-threaded ONNX Runtime Web CPU/WASM and 16 public seeds from `src/league.ts`: 192 games among the four bundled models. Browser leagues use the same game and scoring code. Scores from different opponent fields or seed sets are different benchmarks, so compare results with matching inputs.

Keep model contributions separate from changes to game physics, the model contract or scoring. Propose changes to those rules explicitly so their effect on the comparison can be reviewed.

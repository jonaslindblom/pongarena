# Contributing to PongArena

Contributions, including submitted model weights and their constructors or
training recipes, are accepted under the [MIT License](LICENSE). By submitting
a contribution, you agree to license it under MIT and confirm that you have
the rights to do so. Contributors retain their copyright. Third-party
dependencies and assets retain their own licenses and notices.

Use Node.js 22 and `npm ci`. Import an ONNX file under **Models** to validate it and try it in the arena. Handwritten and trained weights follow the same [model contract](docs/RULES.md).

For a small handwritten example, [Follow](docs/FOLLOW.md) explains all 13 parameters and can be rebuilt with [its constructor](handwritten/follow.py).

Keep the two approaches independent: [Quill's constructor](handwritten/quill.py) writes fixed weights, while [training](training/README.md) starts from random weights, learns from scripts and improves Large through game outcomes. Training and checkpoint selection must not use handwritten models' weights, outputs or matches; compare frozen models with Quill afterward.

## Submit a model

Submit one opset 18 ONNX file, at most 256 KiB. It must map float32 `observation` `[1, 6]` directly to `action_scores` `[1, 3]` through one to five dense `Gemm` layers. Hidden layers use uniform `Relu` or `Tanh`; the output is linear. All weights and biases must be embedded and finite.

Small allows at most **256 parameters and 1,024 MACs** per decision. Large allows **65,536 of each**. Both limits apply. See [rules, observations and scoring](docs/RULES.md) before exporting.

1. Add `public/models/<id>.onnx` and its entry in `src/roster.ts`. The ID must match the filename without `.onnx`. Include `submittedBy: 'your-github-handle'` so the model card can link to its submitter; supply a GitHub username, not a URL. CI requires new or changed model bytes to credit the PR author and preserves existing credits. This credits the submission, rather than claiming authorship of every weight. Describe how the weights were constructed or trained.
2. Include a reproducible constructor or training recipe. Keep checkpoints, datasets and raw runs out of the public tree.
3. Regenerate the league before running checks, since tests verify the bundled roster against its results:

   ```sh
   npm run models:check
   npm run league
   npm test
   npm run league:check
   npm run build
   ```

   Browser checks run in CI. To run them locally, install Chromium with
   `npx playwright install chromium`, then run `npm run test:e2e`. They cover
   submitter links, model layouts, play/watch actions, downloads and imports on desktop and mobile.

4. Include both `public/league.json` and `public/league.md` in the PR. Describe the architecture, validated resource counts, method and evaluation. Identify the opponent field and fresh evaluation seeds for claims of improvement.

`models:check` rejects invalid files, ports, graphs, resource counts, missing roster entries and duplicate models. It also loads every model in WASM and checks exact finite output on 217 observations. League inference checks every decision. Python is optional for the app and league.

## PR checks

The [normal checks](.github/workflows/check.yml) validate models, run tests, replay the committed 16-seed development league and build the app. A separate job checks the training pipeline and reproduces Follow and Quill. Keep both committed league files current when changing the roster.

The [model submission gate](.github/workflows/model-evaluation.yml) runs the trusted base evaluator against PR model files and parses roster metadata as data. It freezes all model hashes, then draws **128 shared serve sets** and evaluates the complete proposed roster on both sides. Invalid submissions fail before ranking. Model bytes and evaluator identity determine the saved plan; retries and display-name changes reuse it, while changed model bytes get fresh serves. Keep model and evaluator changes in separate PRs.

CI retains `evaluation-plan-<PR>-<identity>` and `model-league-<PR>-<identity>` artifacts for 90 days. A missing or expired previously recorded plan fails instead of drawing replacement serves. On merge, Pages verifies the matching trusted results and uses them for the published standings. Compare scores only within the same opponent field and serve set.

Repository branch protection must require **Validate and evaluate submitted models**, **Browser end-to-end tests** and up-to-date branches to prevent merging failed or stale checks. See [publishing](docs/PUBLISHING.md).

Keep model contributions separate from changes to game physics, the model contract or scoring. Propose changes to those rules explicitly so their effect on the comparison can be reviewed.

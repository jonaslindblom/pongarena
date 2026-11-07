# Contributing to PongArena

Use Node.js 22 and `npm ci`. Start with a local model import under **Models** to check the contract and try your opponent. Changes to frozen Classic v4 rules or the submission contract need a separate proposal; model PRs should preserve the comparison protocol.

## Submit a weight model

The active contract is `observation-6/action-scores-3/dense-mlp-v3` under `pong-v4`. Handwritten and trained weights are both welcome. Scripts enter **All controllers** separately and do not receive a model size class.

- Submit one opset 18 ONNX file, no more than 256 KiB, with embedded finite float32 weights and biases.
- Input: `observation`, float32 `[1, 6]`. Output: `action_scores`, float32 `[1, 3]`; the first maximum selects `up`, `stay` or `down`.
- Use one to five sequential `Gemm` layers, with the same `Relu` or `Tanh` after each hidden layer and a linear output. Each layer owns its fixed matrix and bias. No branches, shared parameters, extra features or input/output processing.
- Satisfy both class limits: **Small** ≤256 parameters / 1,024 MACs; **Large** ≤65,536 / 65,536. Small models may compete up.

See the [full contract and scoring](training/COMPETITION.md) and [game specification](training/v4/README.md) before exporting. The historical package's general-graph export section is superseded by the current dense-network contract.

1. Add `public/models/<id>.onnx` and a matching preset in `src/roster.ts`. The ID must match the filename without `.onnx`; describe the construction or training method accurately.
2. Validate the file and obtain its hash and resource counts:

   ```sh
   node --import tsx training/scripts/check_models.ts public/models/<id>.onnx
   ```

3. Add the ID, file path and SHA-256 to `training/catalog.json`. Record a constructor or concise training recipe and evaluation evidence; include checkpoint provenance when available. Keep large checkpoints and raw training runs out of the public tree.
4. Run the checks and regenerate the reference league:

   ```sh
   node --import tsx training/scripts/check_models.ts
   npm test
   npm run league
   npm run league:check
   npm run build
   ```

5. Commit the model, metadata and both `public/league.json` and `public/league.md`. Include the model's origin, architecture, validated counts and evaluation method in the PR description. Any claim of improvement should identify the opponent field and independent evaluation seeds.

The catalog checker verifies local checkpoints when present and explicitly reports missing optional checkpoints. They are not needed to run the app or the public league. Do not remove retired fixtures merely to reduce the active roster; they protect compatibility and preserve earlier comparisons.

## How the reference league is checked

`npm run league` evaluates every bundled model and script with single-threaded ONNX Runtime Web CPU/WASM. It uses the 16 public seeds in `src/league.ts` and writes complete match records plus a Markdown table. The seeds are for reproducible development comparisons; reserve fresh evaluation seeds for training claims.

The **Check and rank** workflow runs on pull requests, pushes to `main` and manual dispatch. It runs tests and replays the league, fails if either committed artifact is stale, and uploads regenerated files as **league-results** with a workflow summary. Download those files if needed, review them and commit the update yourself. The workflow has read-only repository permissions and does not push commits. Fork workflows may require maintainer approval.

Review the passing run on the relevant commit before treating a proposed table as verified. Local browser leagues remain separate from the checked-in reference result. Rules, classes, roster hashes and seed sets identify a benchmark; scores from different benchmarks are not interchangeable ratings.

## Release status

Publication and the source/model-weight license are still pending. See [publishing notes](docs/PUBLISHING.md) before creating the public repository or enabling Pages.

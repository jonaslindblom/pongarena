# Working on PongArena

PongArena is a browser playground and league for tiny neural networks. Help people
play, understand the weights, and submit their own models. Inference runs locally
with ONNX Runtime Web; submissions and official judging use GitHub pull requests
and Actions. There is no submission backend.

## Start with the task

- **Build or submit a model:** [CONTRIBUTING.md](CONTRIBUTING.md) is the entry point;
  [docs/RULES.md](docs/RULES.md) defines observations, actions, graphs and scoring.
- **Program weights:** use [programmed/README.md](programmed/README.md) and the
  Follow constructor as the smallest example.
- **Train weights:** use [training/README.md](training/README.md). Try `--smoke`
  before a full training run; use fresh output directories.
- **Prompt weights:** use
  [the prompting experiment](experiments/one-prompt-weights/follow-up/README.md).
  Saved responses replay without an API key. Requesting new responses costs money.
- **Change evaluation or publication:** read [docs/PUBLISHING.md](docs/PUBLISHING.md)
  and the affected workflow before editing.

Read the relevant implementation, callers and tests. Keep changes focused and
match existing conventions. Surface conflicts between docs and code instead of
silently choosing a new rule.

## Local setup

Use the Node version in `.nvmrc` and npm with the committed lockfile:

```sh
nvm install
nvm use
npm ci
npm run dev
```

If nvm is unavailable, use an equivalent installation of that Node version.
Python 3.12 is needed only for constructors and training; follow their setup docs.
The browser app and local model checks need neither Python nor API credentials.

## Model contribution: one file is enough

1. Develop a new controller using any method allowed by the contribution rules.
   Import it under **Models** to try it locally. Imported weights stay in that browser.
2. Export `public/models/<id>.onnx`. Use a new ID containing only letters, digits,
   underscores or hyphens; core IDs, `keyboard` and the `local-` prefix are reserved.
3. Embed a truthful creation description in the ONNX model's `doc_string` and set
   `pongarena.method` to `programmed`, `trained`, `prompted` or `mixed`.
   `pongarena.name` and an HTTPS `pongarena.source` are optional.
   CONTRIBUTING contains the exact metadata example and limits.
4. Run the local model checks below, then open a model PR against current `main`.
   The normal submission diff contains only the new ONNX file. Recipes, training
   scripts and source links are optional; never require them just to register a model.
5. Leave evaluation and acceptance to CI and maintainer review. CI validates the
   model, generates its card, evaluates it and posts standings on the PR. Pages
   publishes the verified results after merge.

Do not edit `src/roster.ts` or `src/coreModelHashes.ts` to register a community
submission. GitHub supplies contributor credit and the project assigns Core status;
embedded metadata cannot claim either. Preserve required licenses and notices and
confirm the MIT contribution terms in the PR template.

### Export traps

- Opset 18; float32 `observation [1, 6]` → `action_scores [1, 3]`; at most 256 KiB.
- One to five sequential `Gemm` layers, uniform `Relu` or `Tanh` hidden activations,
  linear output, finite embedded weights and biases. `MatMul` + `Add` exports fail.
- Small: at most 256 parameters **and** 1,024 MACs. Large: at most 65,536 of each.
- Keep input ordering and normalization unchanged. X is along the court; Y is
  across it. Actions are left, stay, right on screen; see RULES before writing formulas.
- Changing metadata does not make a duplicate network a new entrant.

## Checks that match the change

For a model submission:

```sh
npm run models:check
npm test
npm run build
```

`npm test` generates the catalog and the development league before running tests.
The 16-set development league is not the official 512-set evaluation. Do not
commit generated catalogs, standings or evaluation plans; `.gitignore` lists them.

For app or evaluator changes, also run the affected tests. Use
`npm run test:e2e` for browser behavior (install Chromium with
`npx playwright install chromium` first), and `npm run league:check` when changing
physics, inference or scoring. Physics changes must keep `src/game.ts` and
`training/game.py` consistent and preserve the parity fixtures. Check UI changes
on narrow screens as well as desktop.
For Python changes, run `python -m unittest discover -s training -p 'test_*.py'`
in the documented environment and the relevant export/smoke checks.
Documentation-only edits need link and command review, not a full tournament.
Report checks actually run, failures, retries and skips; do not imply unrun checks passed.

## Preserve fair comparisons and trusted publication

- Keep model submissions separate from evaluator, physics, contract and workflow
  changes. Never weaken validation to accept a model or bypass the trusted PR
  evaluation with a direct model push to `main`.
- PR model files and descriptions are untrusted data. The submission gate executes
  trusted base code; do not execute contributor code with privileged credentials.
- Reuse saved evaluation plans on retries. Do not reroll serves to improve a score
  or present development results as published standings.
- Preserve frozen models, seeds, hashes and selection rules supporting Findings.
  Tuning informed by held-out results requires a new independent evaluation.
  Follow the training docs' independence rules for the project's trained baselines.
- Use **Programmed**, **Trained**, **Prompted** and **Mixed** for creation methods;
  **Core**, **Community** and **Local** describe origin. Keep claims tied to the
  actual models, opponent field and evaluation, not broad architecture limits.

For follow-up work after a merge, fetch the current default branch and start a
fresh branch. Do not base new work on an old PR branch. Preserve unrelated local
changes; ask if they block the requested work or the intended scope is unclear.
Propose competition-rule changes explicitly in a separate PR.

# Publishing PongArena

PongArena’s original source code, documentation and bundled model weights use
the [MIT License](../LICENSE). Keep that license and third-party license
notices with releases. Repository visibility and website deployment are
managed separately.

## Check the release

```sh
npm ci
npm run models:check
npm run league
npm test
npm run league:check
npm run build
git status --short
```

Review and commit the model files, metadata and both league results. The app and league need no Python dependencies. A separate CI job installs the training environment, runs a small pipeline check and reproduces Follow and Quill from their standalone constructors. Full training stays outside PR checks. After creating the GitHub repository and adding its URL as `origin`, push the reviewed `main` branch:

```sh
git push -u origin main
```

## Ranking in pull requests

The [normal checks](../.github/workflows/check.yml) validate models, run tests, replay the committed development league and build the app. Development results remain available as **development-league-results** for seven days.

The [model submission gate](../.github/workflows/model-evaluation.yml) uses `pull_request_target` and executes only the trusted base evaluator. PR ONNX files and roster metadata supply data; PR scripts and workflows are not executed by this gate. Model and evaluator changes must use separate PRs.

For model-file changes, the gate freezes model hashes, draws 512 shared serve sets and evaluates all proposed entrants with paired sides. It retains `evaluation-plan-<PR>-<identity>` and `model-league-<PR>-<identity>` for 90 days. Retries and display-name changes reuse the saved plan; changed model bytes get fresh serves. A missing or expired recorded plan fails instead of silently rerolling. Documentation-only PRs do not redraw serves.

The evaluator's shared default and publication check both require 512 sets. The `pong-v3` rule change creates a new identity and plan, preserving earlier 128-set artifacts as historical results. Reports include the same paired-seed bootstrap score intervals displayed in the app.

Configure branch protection to require **Validate and evaluate submitted models**, **Browser end-to-end tests** and up-to-date branches. A failing workflow check blocks merging only when the repository requires it. Rebase and reevaluate if another model merge changes the roster before yours merges.

Standard GitHub-hosted runners are free for public repositories; this workflow uses a standard Ubuntu CPU runner. Larger runners are billed even for public repositories, and artifact storage has a separate allowance. See [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions) for current terms.

The evaluation workflow has read-only contents and Actions access. Review evaluator and workflow changes separately from model submissions.

## Publish the website

The [Publish Pages workflow](../.github/workflows/pages.yml) runs on pushes to `main` and manual dispatch. It verifies the merged PR's trusted results against the exact model bytes and evaluator identity, then builds those standings into `dist/`. Model merges without matching results fail publication. It retains `published-league-<identity>` so retries and metadata-only updates preserve the accepted standings. Infrastructure changes may bootstrap a trusted-main evaluation with a persisted `main-evaluation-plan-<identity>`; retries reuse it. These artifacts have the same 90-day retention. Deployment permissions stay limited to the deployment job.

1. In the repository, select **Settings → Pages → Source → GitHub Actions**.
2. Merge a pull request into `main`, push to `main`, or run **Actions → Publish Pages → Run workflow** on `main`.
3. Open the deployment URL and check play, model loading and the league. The build uses relative asset paths to support a repository Pages URL.

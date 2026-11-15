# Publishing PongArena

PongArena’s original source code, documentation and bundled model weights use
the [MIT License](../LICENSE). Keep that license and third-party license
notices with releases. The static build includes [third-party notices](../public/THIRD_PARTY_NOTICES.txt); review them when changing runtime dependencies or fonts. Repository visibility and website deployment are
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

Review and commit the model files with their embedded metadata. The generated catalog and league results are ignored by Git. The app and league need no Python dependencies. A separate CI job installs the training environment, runs a small pipeline check and reproduces Follow, Predict, Attack, Counter and Quill from their standalone constructors. Full training stays outside PR checks. After the required PR checks pass, merge the reviewed changes into `main`. For a separate deployment of your own fork, push its reviewed `main` branch:

```sh
git push -u origin main
```

## Ranking in pull requests

The [normal checks](../.github/workflows/check.yml) validate models, run tests, generate the development league and build the app. Development results remain available as **development-league-results** for seven days.

The [model submission gate](../.github/workflows/model-evaluation.yml) uses `pull_request_target` and executes only the trusted base evaluator. PR ONNX files and embedded metadata supply data; PR scripts and workflows are not executed by this gate. Model and evaluator changes must use separate PRs.

For model-file changes, the gate freezes model hashes, draws 512 shared serve sets and evaluates all proposed entrants with paired sides. It retains `evaluation-plan-<PR>-<identity>` and `model-league-<PR>-<identity>` for 90 days. Retries and metadata-only changes reuse the saved plan; changed networks get fresh serves. Canonical graph-and-weight hashes identify networks, and separate file checksums protect the exact bytes. A missing or expired recorded plan fails instead of silently rerolling. Documentation-only PRs do not redraw serves.

After a successful model evaluation, the [results workflow](../.github/workflows/model-results-comment.yml) posts the standings on the PR. It updates the same comment on reruns and checks that the evaluated commit is still the PR head. To report an earlier successful run, dispatch the workflow with its run ID.

The evaluator's shared default and publication check both require 512 sets. Reports include the same paired-seed bootstrap score intervals displayed in the app.

### Required checks

In **Settings → Rules → Rulesets** (or branch protection), protect `main` with pull requests, up-to-date branches and all four status checks: **check**, **training**, **Browser end-to-end tests** and **Validate and evaluate submitted models**. A failing workflow check blocks merging only when the repository requires it. Rebase and reevaluate if another model merge changes the roster before yours merges.

Standard GitHub-hosted runners are free for public repositories; this workflow uses a standard Ubuntu CPU runner. Larger runners are billed even for public repositories, and artifact storage has a separate allowance. See [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions) for current terms.

The evaluation workflow has read-only contents and Actions access. Review evaluator and workflow changes separately from model submissions.

## Publish the website

The [Publish Pages workflow](../.github/workflows/pages.yml) runs on pushes to `main` and manual dispatch. It verifies the merged PR's trusted results against the validated network identities and evaluator identity, then builds those standings into `dist/`. Model merges without matching results fail publication. It retains `published-league-<identity>` so retries and metadata-only updates preserve the accepted standings. Infrastructure changes may bootstrap a trusted-main evaluation with a persisted `main-evaluation-plan-<identity>`; retries reuse it. These artifacts have the same 90-day retention. Deployment permissions stay limited to the deployment job.

1. In the repository, select **Settings → Pages → Source → GitHub Actions**.
2. Merge a pull request into `main`, push to `main`, or run **Actions → Publish Pages → Run workflow** on `main`.
3. Open the deployment URL and check play, model loading and the league. The build uses relative asset paths to support a repository Pages URL.

## Generated catalog and submitter credit

`npm run models:catalog` discovers all ONNX files and writes `public/models.json`. Core examples are maintained in `src/roster.ts` and their network identities are pinned in `src/coreModelHashes.ts`; a submitted file cannot claim Core status. The default local command preserves available credits from an earlier catalog and can leave new community models without a submitter link.

The trusted model gate and Pages run `models:catalog -- --github` with read-only GitHub access and a full-history checkout. The catalog traces each community network's first appearance through main's first-parent history, then reads its accepted PR association from GitHub. The gate credits a new network to the authenticated current PR author. Publication requires accepted PR history. Renaming a file or editing metadata preserves its network identity and original submission credit; source and author claims in ONNX metadata are not used for attribution.

Pages builds require a complete matching 512-set evaluation bundle and fail if the promoted standings or report are missing or inconsistent; they never fall back to development results. Local build preparation preserves matching promoted standings. Fresh clones and changed model collections get development standings automatically. The initial rollout changes evaluation identity to the canonical-network schema, so Pages performs the existing trusted-main bootstrap once; subsequent publication and retry rules remain unchanged.

# Publishing PongArena

Choose a license for the source and model weights before publication. Prepare the intended initial commit from the reviewed release files; creating the public repository and publishing the site are separate steps.

## Check the release

```sh
npm ci
npm run league
npm test
npm run league:check
npm run build
git status --short
```

Review and commit the model files, metadata and both league results. The app and league need no Python dependencies. A separate CI job installs the training environment and runs a small pipeline check; full training stays outside PR checks. After creating the GitHub repository and adding its URL as `origin`, push the reviewed `main` branch:

```sh
git push -u origin main
```

## Ranking in pull requests

The [Check and rank workflow](../.github/workflows/check.yml) runs on pull requests, pushes to `main` and manual dispatch. It verifies the committed league, displays regenerated standings in its summary and uploads **league-results** for seven days. It does not push commits. The website reads the checked-in results immediately.

Standard GitHub-hosted runners are free for public repositories; this workflow uses a standard Ubuntu CPU runner. Larger runners are billed even for public repositories, and artifact storage has a separate allowance. See [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions) for current terms.

The workflow uses `pull_request` with read-only repository permissions. Keep secrets out of model evaluation and review any workflow changes submitted in PRs.

## Publish the website

The [Publish Pages workflow](../.github/workflows/pages.yml) is manual and accepts only `main`. It runs tests, verifies the league, builds `dist/`, then deploys with permissions limited to its deployment job.

1. In the repository, select **Settings → Pages → Source → GitHub Actions**.
2. Run **Actions → Publish Pages → Run workflow** on `main`.
3. Open the deployment URL and check play, model loading and the league. The build uses relative asset paths to support a repository Pages URL.

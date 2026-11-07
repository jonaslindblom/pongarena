# Publishing PongArena

The repository is prepared locally. Creating a public repository, selecting its license and enabling a public website remain release decisions.

## GitHub Actions compute

Standard GitHub-hosted runners are free for public repositories. The league can run on the standard `ubuntu-24.04` CPU runner; it does not need a paid larger runner or GPU. Public standard-runner usage does not consume the private-repository compute allowance. Larger runners are billed even for public repositories. [GitHub billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions), [runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

Free compute still has operating limits: a hosted job can run for at most six hours, and GitHub Free allows 20 concurrent standard jobs. Artifact storage has a separate allowance (500 MB on GitHub Free); keep league artifacts small and use short retention. These limits were checked on 7 November 2026. [Actions limits](https://docs.github.com/en/actions/reference/limits).

The included [Check and rank workflow](../.github/workflows/check.yml) runs `npm test`, `npm run league:check` and `npm run build` on pull requests, pushes to `main` and manual dispatch. It uploads regenerated results as `league-results` for seven days and writes the standings to the workflow summary. Model PRs run `npm run league` locally and commit both `public/league.json` and `public/league.md`; CI verifies them and does not push changes. The website displays the checked-in snapshot without running a tournament in the visitor's browser. The workflow is prepared locally; no GitHub run has been dispatched.

Use `pull_request` with read-only repository permissions. Fork pull requests receive a read-only `GITHUB_TOKEN` and no repository secrets; first-time contributors may need a maintainer to approve the workflow. Do not use `pull_request_target` to build or evaluate submitted code. [Fork workflow behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflows-in-forked-repositories), [workflow approval settings](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository#controlling-changes-from-forks-to-workflows-in-public-repositories).

## Start the public repository from a clean snapshot

The local development history contains a 209 MB experiment result, `training/results/compact-lab/evaluation/selection.json`. Although that file is gone from the current tree, pushing the existing history would include it. GitHub blocks Git objects larger than 100 MiB. [GitHub file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

Keep this local repository and its experiment history. Create the public repository from a fresh snapshot of the reviewed, committed files instead of rewriting or deleting the original history. The current tree's largest file is under 10 MB.

Before exporting, select a license for the source and model weights and commit the intended release files. No license is selected by this preparation.

The public Refined Nano evidence bundle uses portable repository-relative paths for five validation file references in one protocol. Its `SHA256.json` entry for that protocol was updated, and all 194 manifest entries verify. Those five strings and the corresponding manifest hash are the only content changes; measured data, model bytes and recorded model hashes are unchanged. The byte-identical original is preserved locally at `training/archive/refined-nano-v4-evidence-before-publication.zip`, which is ignored and excluded from the export.

From the original checkout, after the release changes have been committed:

```sh
git status --short
publish_dir=$(mktemp -d "${TMPDIR:-/tmp}/pongarena-public.XXXXXX")
git archive HEAD | tar -x -C "$publish_dir"
git -C "$publish_dir" init -b main
git -C "$publish_dir" add --force .
git -C "$publish_dir" commit -m "Initial public release"
cd "$publish_dir"
npm ci
npm test
npm run league:check
npm run build
```

The forced add is intentional and must happen before installing dependencies or building: the export contains only files already tracked in the original repository, including required `training/` sources that the local `.gitignore` otherwise ignores. Do not force-add the original working directory.

Review this new repository, then connect it to the chosen GitHub repository. Its history starts at the public release; the original checkout, checkpoints and ignored experiment archive remain intact. Python training dependencies and local checkpoints are optional for the web app and Node-based league workflow.

## Publish the website when ready

The included [Publish Pages workflow](../.github/workflows/pages.yml) is manual and accepts only the `main` branch. Its build job installs dependencies, runs tests, verifies the league and builds `dist/`. A separate deployment job receives the Pages and identity-token permissions. Pull-request checks do not deploy the site.

After creating and reviewing the public repository, select **Settings → Pages → Source → GitHub Actions**. Run **Actions → Publish Pages → Run workflow** on `main` when ready to publish. Confirm the app's base path matches the chosen Pages URL, typically `https://<owner>.github.io/<repository>/`. This preparation has not created a repository, enabled Pages or deployed a website.

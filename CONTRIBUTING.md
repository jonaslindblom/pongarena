# Contributing to PongArena

Contributions, including submitted model weights and their constructors or
training recipes, are accepted under the [MIT License](LICENSE). By submitting
a contribution, you agree to license it under MIT and confirm that you have
the rights to do so. Contributors retain their copyright. Third-party
dependencies and assets retain their own licenses and notices.

Use Node.js 26 LTS (the tested version is in [`.nvmrc`](.nvmrc)) and `npm ci`. Import an ONNX file under **Models** to validate it and try it in the arena. Programmed, trained and prompted weights follow the same [model contract](docs/RULES.md).

For a small example of programmed weights, [Follow](docs/FOLLOW.md) explains all 13 parameters and can be rebuilt with the [shared constructor setup](programmed/README.md).

Keep the methods independent: [Quill's constructor](programmed/quill.py) rebuilds weights from formulas that Codex developed and revised using simulated match feedback. The [training recipes](training/README.md) start from random weights and use scripts, learned teachers and game outcomes. [Small](experiments/quill-training/README.md) shares Quill's architecture and learns through Large distillation, PPO and CMA-ES. Training and checkpoint selection must not use programmed models' weights, outputs or matches; compare frozen models with Quill afterward.

## Submit a model

Submit one opset 18 ONNX file, at most 256 KiB. It must map float32 `observation` `[1, 6]` directly to `action_scores` `[1, 3]` through one to five dense `Gemm` layers. Hidden layers use uniform `Relu` or `Tanh`; the output is linear. All weights and biases must be embedded and finite.

Small allows at most **256 parameters and 1,024 MACs** per decision. Large allows **65,536 of each**. Both limits apply. See [rules, observations and scoring](docs/RULES.md) before exporting. The [coordinate glossary](docs/RULES.md#coordinates-and-player-identity) distinguishes game coordinates, normalized model inputs and displayed player labels; keep the existing observation and action order.

1. Give the ONNX file a short description of how the weights were created and a creation method (see below).
2. Add **only `public/models/<id>.onnx`** and open a pull request. Use letters, digits, underscores or hyphens for the filename. Core model IDs, `keyboard` and the `local-` prefix are reserved.
3. CI discovers the model, checks its graph and runtime output, generates its model card, and evaluates it with the full league. After review and merge, the site publishes the accepted results.

Training scripts, constructors, prompts, papers and reproducible recipes are **optional**. Include a source link in the file or supporting material in the PR if you want to share them. Keep checkpoints, datasets and raw runs out of the public tree. The repository's MIT contribution terms above still apply to submitted weights and any included supporting material.

### Description inside the ONNX file

Set the model-level `doc_string` to a short, plain-text explanation (up to 4,000 characters). The website displays this as the contributor's description of how the weights were created; evaluation independently measures performance.

Use these model-level `metadata_props` keys:

| Key | Requirement |
| --- | --- |
| `pongarena.method` | Required: `programmed`, `trained`, `prompted` or `mixed` |
| `pongarena.name` | Optional display name, up to 80 characters; defaults to the filename |
| `pongarena.source` | Optional HTTPS link to a recipe, repository, paper or prompt |

For example, after exporting an ONNX model:

```python
import onnx

model = onnx.load("falcon.onnx")
model.doc_string = "Trained from random initialization with PPO against self-play opponents."
props = {entry.key: entry.value for entry in model.metadata_props}
props.update({
    "pongarena.name": "Falcon",
    "pongarena.method": "trained",
    # Optional: "pongarena.source": "https://github.com/you/training-recipe",
})
onnx.helper.set_model_props(model, props)
onnx.save(model, "falcon.onnx")
```

Metadata counts toward the 256 KiB file limit. No separate roster entry or generated league files are needed. Local browser imports can omit the description and method, so you can try a model before preparing its submission.

CI records **Submitted by @handle** from the authenticated GitHub PR author. Later metadata edits and file renames preserve the original network's submission credit. Embedded text cannot claim submitter credit or Core status. The site labels project teaching examples **Core**, accepted contributions **Community**, and browser imports **Local**; all accepted models compete under the same rules.

### Optional local checks

```sh
npm ci
npm run models:check
npm test
npm run build
```

The catalog and development standings are generated automatically and ignored by Git. Browser checks run in CI; to run them locally, install Chromium with `npx playwright install chromium`, then run `npm run test:e2e`.

`models:check` rejects invalid files, metadata, ports, graphs, resource counts and duplicate networks, including copies with different descriptions. It loads every model in WASM and checks exact finite output on 217 observations. League inference checks every decision. Python is optional for the app and league.

## PR checks

The [normal checks](.github/workflows/check.yml) validate models, run tests, generate the 16-seed development league and build the app. A separate job checks the training pipeline and reproduces all five programmed models: Follow, Predict, Attack, Counter and Quill.

The [model submission gate](.github/workflows/model-evaluation.yml) runs the trusted base evaluator against PR model files and reads their embedded metadata as data. It freezes all model hashes, then draws **512 shared serve sets** and evaluates the complete proposed roster on both sides. Invalid submissions fail before ranking. The validated network and evaluator determine the saved plan; retries, descriptions, display names and equivalent ONNX encodings reuse it, while changed networks get fresh serves. Exact file checksums separately protect the bytes used during each run and downloaded by the browser. PRs that only edit embedded descriptions or names still pass validation and update the catalog, but reuse the accepted evaluation without drawing new serves. Keep model and evaluator changes in separate PRs.

CI retains `evaluation-plan-<PR>-<identity>` and `model-league-<PR>-<identity>` artifacts for 90 days. A missing or expired previously recorded plan fails instead of drawing replacement serves. On merge, Pages verifies the matching trusted results and uses them for the published standings. Compare scores only within the same opponent field and serve set.

Repository branch protection must require all checks and up-to-date branches to prevent merging failed or stale checks. The authoritative [required-check list](docs/PUBLISHING.md#required-checks) is in the publishing guide.

Keep model contributions separate from changes to game physics, the model contract or scoring. Propose changes to those rules explicitly so their effect on the comparison can be reviewed.

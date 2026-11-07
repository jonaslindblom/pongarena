# Train a PongArena paddle

The active game is **Classic v4** (`pong-v4`). The current competition contract is `observation-6/action-scores-3/dense-mlp-v3`: six raw observations, three scores, one to five dense affine layers, uniform ReLU or Tanh hidden activations, and a linear output. See [COMPETITION.md](COMPETITION.md) for exact eligibility and resource budgets. The dense contract supersedes the older general-graph export section in the immutable [v4 package](v4/README.md).

## Current models

`public/models/` contains the six active models. [catalog.json](catalog.json) records their files, SHA-256 identities, selected training checkpoints and retained evidence paths, plus three archived models marked `retired`. Do not select a checkpoint by filename or historical performance alone: validate its actual graph.

| Model | Parameters | Purpose |
| --- | ---: | --- |
| Robust Strong v4 | 17,795 | Strongest retained full-size learned policy from the robustness confirmation study |
| Plain Nano v4 | 447 | Trained Nano with local CMA refinement |
| Quill Plus v4 | 163 | Four handwritten refinements; initial opponent in Watch models |
| Vector Nano v4 | 487 | Handcrafted dense Nano alternative |
| Prism Small v4 | 1,815 | Handcrafted dense model in the Large class; initial player in Watch models |
| Tracker | 163 | Simple baseline and human-play starter |

Robust Strong and Plain Nano still have substantial weaknesses against adaptive scripts. Smaller models do not preserve full-size strength. These are retained study conclusions, not a newly run championship; see [evidence.md](evidence.md).

Apprentice v4, Pacer v4 and Refined Nano v4 are retired from the app roster. Their unchanged ONNX bytes are in `tests/fixtures/models/` and the public [retired-model download](../public/training/retired-v4-models.zip). Extract and import those files to reproduce saved leagues. Their catalog entries, checkpoints, reports and evidence are retained. Pacer remains a fixed training/evaluation reference; moving it to the archive does not change the training opponent field.

Archived Refined Nano is the frozen `cma-fine-g40` candidate: `6→16→3` Tanh, 163 parameters and 144 MACs. Its selected checkpoint, training/selection protocols, confirmation results and later Quill Plus comparison are retained in `public/training/refined-nano-v4-evidence.zip`; `/training/refined-nano-v4.html` summarizes both studies. The bundle includes the exact checkpoint at `training/runs/nano163-refine-20261106/cma/cma-fine-g40.pt`. Copy it to the catalog's local checkpoint path to enable checkpoint checks in a fresh checkout. The full-size DAgger trainer below cannot load this smaller topology.

Old v1/v2/v3 policies may still be eligible dense networks, but are retired from the app roster. Burst models use the wrong input/action interface. Strategist, Handcrafted Nano/Small, Counter Attacker and shortcut-based Robust Small violate the active architecture contract. Retired model bytes and minimal historical manifests live in `tests/fixtures/` for compatibility and rejection tests.

## Setup and checks

Run from the repository root with Python 3.12. [requirements.txt](requirements.txt) pins the dependencies used by the maintained training tools. [requirements-lock.txt](requirements-lock.txt) preserves the full recorded research environment, including plotting packages used only by archived report generators.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r training/requirements.txt
python3 training/v4/check.py
node --import tsx training/scripts/check_models.ts
node --import tsx training/scripts/check_replays.ts
.venv/bin/python training/scripts/check_training.py
npm test
npm run build
```

The model check verifies the six-model active roster and all nine cataloged models' hashes, available local checkpoint hashes and current dense eligibility. Checkpoints and the archive are local ignored artifacts; source-only checkouts report their unavailable checkpoint checks explicitly. The Python training check compares Torch/native ONNX decisions, retained checkpoints and vector/scalar v4 physics. The replay checker uses the archived Pacer/Apprentice v4 weights and replays; it does not claim to replay newer studies.

Maintained training source and documentation are tracked. Generated runs, results, checkpoints and local archives are ignored. New runs belong in `training/runs/`; use a fresh output directory, and never write directly over deployed weights or retained checkpoints.

## Train a full-size dense policy

[train_pace.py](scripts/train_pace.py) is the maintained DAgger entry point. It trains a plain `6→128→128→3` Tanh policy against the current Robust Strong, Pacer and Tracker references plus the three adaptive scripts. Geometry is used by the teacher during training, not included in the exported network. This is an updated training recipe, not an exact replay of the historical Pacer study.

Supply a JSON array of distinct unsigned 32-bit development seeds outside `[0, 1000000000)`, which is reserved for training. Choose new development seeds and reserve a separate final evaluation set; do not tune against published study confirmation seeds. No seed set is generated or silently reused by the command.

```sh
.venv/bin/python training/scripts/train_pace.py \
  --out training/runs/my-dagger-run \
  --development-seeds /absolute/path/to/development-seeds.json \
  --seed 8001
```

To warm-start the same full-size topology, add `--initial training/checkpoints/robust-strong-v4.pt`. The retained checkpoints are trusted local Torch state dictionaries. Checkpoint files preserve weights, not complete optimizer/RNG/environment state; this is not exact resumption. Plain Nano has a different topology and cannot be loaded by this full-size trainer.

A short workflow check can use `--rounds 1 --envs 8 --horizon 8 --epochs 1 --eval-every 1`. It verifies the pipeline and does not establish playing strength. Training disables exact-loop exits because the policy changes and actions are sampled from the teacher/learner mixture.

## Evaluate a candidate

Use explicit fresh seeds, both side assignments and five rallies per side. This command validates the candidate and model opponents with the same TypeScript validator as the app before running native ONNX inference. Fixed deterministic policies enable exact-loop draws. Output includes model/opponent hashes, individual seeds, side results and paired results.

```sh
.venv/bin/python training/scripts/evaluate_current.py \
  training/runs/my-dagger-run/best.onnx \
  --seeds-file /absolute/path/to/final-seeds.json \
  --output training/runs/my-final-evaluation.json
```

The default field is Robust Strong, archived Pacer, Attacker, Defender and Interceptor. `--opponents` accepts active or archived catalog IDs or `attack`, `defend`, `interceptor`. A finite field and a small seed set do not establish universal strength. Use development results to select a frozen candidate, then open the independent final set once. The app's [reference league](../public/league.md) and optional local leagues use the separate [round-robin protocol](COMPETITION.md).

## Reconstruct handcrafted weights

These constructors require fresh output paths. Vector and Prism verify their original byte hashes; compare Quill Plus's regenerated hash with [catalog.json](catalog.json). Validate the exported files before considering them for the app.

```sh
.venv/bin/python training/scripts/vector_dense_v4.py --out training/runs/vector-nano.onnx
.venv/bin/python training/scripts/robust_dense_v4.py --out training/runs/prism-small.onnx
.venv/bin/python training/scripts/quill_plus_dense_v4.py --out training/runs/quill-plus.onnx
node --import tsx training/scripts/check_models.ts training/runs/vector-nano.onnx training/runs/prism-small.onnx training/runs/quill-plus.onnx
```

The Quill constructors use NumPy and ONNX only. Original Quill is retired from the active roster; its bytes remain at `tests/fixtures/models/quill-nano-v4.onnx` for regression checks and its base constructor remains a dependency of Quill Plus. Their 163 values are generated from fixed handwritten formulas, with no optimizer, training examples or loaded teacher weights. Original Quill's manual development used 32 seeds; the candidate was frozen before the separate 128-seed native ONNX evaluation. `/training/quill-nano-v4.html` and its evidence download retain all ten matchups, the constructor, development proposals, seed lists and native/WASM parity checks. Those final seeds are consumed; use fresh seeds for future model selection.

Quill Plus is the only active Quill preset, selected for the stronger ten-opponent average. It imports original Quill's unchanged constructor and modifies four generated scalars. Development used 64 new seeds and ten manually specified revisions; the selected `speed-recovery` candidate was then frozen. Both original Quill and Plus were evaluated on the same separate 128 fresh seeds against ten references, with a further direct matchup. The aggregate rally-score gain is +2.60 percentage points (seed-bootstrap 95% CI +1.98 to +3.22); several Nano matchups regress and the direct matchup shows no clear advantage. `/training/quill-plus-v4.html` and its evidence download preserve all proposals, raw results, frozen identities and verification. These final seeds are also consumed.

The remaining Python modules in `scripts/` support this trainer, its v4 simulator, script teachers and evaluation. `analytic_dense_v4.py` contains the serializer shared by Vector and Prism. Historical experiment launchers and report generators are archived rather than offered as current workflows.

## Historical material

Completed experiments, failed runs, superseded checkpoints, original source snapshots, report generators and the previous training README are in the verified [experiment archive](archive/README.md). Full reports remain historical evidence; earlier browser import checks do not establish eligibility under today's contract. The Plain Nano search and Robust Strong PPO recipes are preserved there; they are not maintained active launchers.

`v1/`, `v2/`, `v3/` and `burst/` are immutable compatibility packages, retained for frozen-engine tests and historical interpretation. They are not active training targets. The current vector simulator shares the v3 state/serve implementation and overrides return physics for Classic v4; the training check verifies it against `v4/reference.py`.

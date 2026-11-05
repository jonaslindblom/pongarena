# Deterministic fine-tuning of a fully learned policy

The actor remains a plain 6→64→64→3 Tanh MLP with 4,803 learned parameters. It starts from the retained fully learned PPO checkpoint; no fixed trajectory calculation, paddle feedback, feature transform, lookup table or script enters its forward pass. The final ONNX graph is Gemm/Tanh/Gemm/Tanh/Gemm. Output-only search freezes previously learned hidden weights during this fine-tuning run; those weights are not hand-designed geometry.

Commands, from the repository root:

```sh
.venv/bin/python training/scripts/train_search.py --out training/runs/search-all-91 --seed 91 --generations 30 --check
.venv/bin/python training/scripts/train_search.py --out training/runs/search-output-93 --seed 93 --generations 30 --scope output --sigma .01
```

Use fresh output paths when reproducing. Each generation includes the incumbent and eight antithetic perturbations (four Gaussian directions, ±, scales 0.5/1/2/4 times sigma). All parameters mutate in the first run; only the learned final affine layer mutates in the second. Every population member plays the same eight fresh training seeds against both v2 and the unchanged attacker, with all five openings and both sides. The top two mutations and the incumbent then play a fixed 32-seed training confirmation panel. A mutation replaces the incumbent only if its weaker-target point margin plus 0.3 times its ordinary five-rally match win rate improves; mean point margin breaks ties. This training objective avoids stochastic-policy/deployment mismatch but can overfit its confirmation panel.

Every five generations the current actor is evaluated on a separate, fixed 64-seed development panel in [1.3b,1.4b), sampler 502. Checkpoint selection prioritizes minimum ordinary five-rally win rate, then minimum point margin, then mean ordinary win rate. These development measurements are selection data, not final evidence. All final testing belongs to the separately frozen root protocol.

All-weight search executed 30 generations and 52,475,112 active rally decisions, selecting generation 25 at 43,584,643 decisions. Output-only search executed 30 generations and 49,854,626 decisions, selecting generation 30.

| Run | Opponent | Development rally W / L / D | Ordinary game W / L / D |
| --- | --- | --- | --- |
| All weights | Attacker | 153 / 18 / 469 | 86 / 6 / 36 |
| All weights | v2 | 260 / 17 / 363 | 114 / 1 / 13 |
| Output layer | Attacker | 152 / 26 / 462 | 87 / 9 / 32 |
| Output layer | v2 | 336 / 105 / 199 | 101 / 9 / 18 |

Both use the unchanged v3 vector environment and exact loop exits for fixed deterministic policies. `--check` confirms duplicate population members and the standard evaluator agree on three complete seed sets. `export-check.json` records independent 8,192-observation native ONNX/PyTorch checks with exact action agreement. `metrics.jsonl` contains full development evaluations; `search.jsonl` contains every training confirmation and accepted-mutation decision. `env_steps` counts active training trajectory decisions, excluding validation and pretraining; it is not a gradient-update count. Retain configs, metrics, search records, selected checkpoints and export checks. Runs have no exact resume facility or claim of cross-platform bit-for-bit training.

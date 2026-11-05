# Dense weights-only restart

Status: **no improved compact policy found in these experiments**. No candidate in this directory is promoted to the active catalog. The earlier arithmetic/physics ONNX programs and Strategist are ineligible under the corrected dense-only contract; their old results do not establish dense-network strength.

The active contract is `observation-6/action-scores-3/dense-mlp-v2`. It accepts only a sequential chain of one to three dense affine layers, with uniform ordinary ReLU or Tanh between hidden layers and a linear three-score output. Every layer owns finite, embedded weights and biases. Dynamic input products, branches, feature/lookup stages, computed parameters, nonunit Gemm scaling, unused constants, and extra processing are rejected. The change is recorded in commit `16805c9`; game physics is unchanged.

## Interpreting actual learned weights

The main source is the frozen Pacer checkpoint `training/results/pace-study/training/candidate/pacer/best.pt`, SHA-256 `57cd566fd8a0ddecb19dd2d20cce54a6bd1d18fb9784c37e5f5e2d8fd6790371`. It is a conventional `6→128→128→3` Tanh network. No geometry or scripted features were constructed for the replacement networks.

For a first-layer unit, the row weights define an affine detector and its bias sets the activation threshold. Subsequent layers combine those activations; the final row differences determine up/stay/down votes. A large weight shared by all output rows can change their common offset without changing the selected action, so the audit ranks action-contrast contributions rather than raw magnitude alone.

A causal sensitivity audit of **19,012 reachable source-policy states** measures mean absolute centered-score derivatives per normalized input unit:

| Input | Sensitivity |
| --- | ---: |
| Own paddle Y | 16.62 |
| Vertical velocity | 10.60 |
| Ball Y | 6.99 |
| Ball X | 5.98 |
| Horizontal velocity | 5.42 |
| Opponent Y | 1.25 |

These are sensitivities to normalized observations, not raw court units or unique neuron labels. Some learned first-layer rows combine vertical position and velocity with ratios resembling roughly 31–35 ticks of straight-flight anticipation. The same rows also mix paddle position, horizontal phase and biases; they are not standalone trajectory calculators. Controlled activations and downstream vote weights are needed to interpret their effect.

The second Tanh layer is saturated above absolute 0.99 on about 78.9% of this state set. Nearest affine-detector cosine similarity has median 0.887. The representation therefore has substantial context-dependent nonlinear structure and lacks enough nearly identical units for exact channel merging. These measurements motivated the transfer attempts; they do not prove a compact policy impossible.

## Fixed dense architectures and transfers

Primary replacement architectures were declared before editing: Nano `6→16→16→3` (435 parameters, 400 MACs) and Small `6→32→32→3` (1,379 parameters, 1,312 MACs). Both are exactly three Gemm layers and two Tanh activations. There are no input/output adapters or additional nodes.

Four source-weight transformations were tested at both widths:

1. **Contribution-ranked pruning:** retain important hidden channels and their original weights; fold removed channels' neutral-input constant contributions into biases.
2. **Sign-aware feature merging:** select representative affine detectors using weight/bias cosine similarity, then combine their existing downstream votes with the matching sign. No regression coefficients are fitted.
3. **Copied-bias pruning control:** use the same channel masks as pruning while copying retained source biases unchanged, to test whether neutral compensation caused the failures.
4. **Source-only local linearization:** derive the second-layer Tanh derivative at the source neutral operating point, transfer its effective output weights, and use a small-gain identity middle layer in the fixed architecture. This uses source-weight algebra only, without labels, backpropagation updates or fitting.

Source decision agreement falls to roughly 33–49%, and gameplay fails:

| Candidate | Tracker W/L/D | Anticipator W/L/D | Challenger W/L/D | Pacer W/L/D | Adaptive attacker W/L/D |
| --- | ---: | ---: | ---: | ---: | ---: |
| prune-16 | 50/188/2 | 22/218/0 | 26/214/0 | 5/235/0 | 0/240/0 |
| merge-16 | 112/126/2 | 72/168/0 | 100/140/0 | 12/228/0 | 0/240/0 |
| prune-32 | 50/188/2 | 22/218/0 | 26/214/0 | 4/236/0 | 0/240/0 |
| merge-32 | 82/158/0 | 53/187/0 | 57/183/0 | 4/236/0 | 0/240/0 |
| copy-bias-16 | 60/180/0 | 29/211/0 | 32/208/0 | 9/230/1 | 0/240/0 |
| taylor-16 | 76/164/0 | 43/197/0 | 52/188/0 | 7/233/0 | 0/240/0 |
| copy-bias-32 | 64/176/0 | 25/215/0 | 35/205/0 | 18/222/0 | 0/240/0 |
| taylor-32 | 54/184/2 | 34/205/1 | 33/207/0 | 3/237/0 | 0/240/0 |

Each pairing uses 24 fresh development seeds, five serves and both physical assignments: 240 rallies. `seed-audit.json` checks the exact seeds against 334 retained JSON/JSONL files and 7,259,773 integer leaves with zero overlaps. These are development outcomes in the verified v4 vector environment using the explicit dense PyTorch networks. No final holdout was consumed because no candidate qualified for promotion.

The copied-bias controls show that neutral compensation was not the sole failure. The local linearization also loses nonlinear behavior that matters away from its operating point. Neither result justifies another claim of improvement.

## Additional explained weight controls

`edit-screen.json` retains two edits to the existing ordinary `6→16→3` ReLU models. Anticipator's vertical velocity coefficient 0.525 corresponds to 7 ticks because velocity is divided by 6 and Y by 80: horizon = coefficient×80/6. Increasing it to 1.35 (18 ticks) loses 67/166/7 against Anticipator and 0/240 against the attacker. Straight-line anticipation fails when future wall contacts matter. Increasing Challenger's primary own-position feedback by 10% also fails to improve its overall field.

A separate fixed `6→32→3` ReLU control copies Challenger's learned feature rows with vertically reflected weight signs and swapped action votes. This is a normal single-hidden dense network (323 parameters, 288 MACs, three nodes), with no graph branches or runtime feature operations. It scores 128/112/0 against Challenger but its margin confidence interval spans zero; it loses 112/123/5 to Anticipator and 0/240 to the attacker. It is retained as a rejected transfer, not advertised as a stronger model.

## Evidence and limits

`analysis.json` records actual source rows, contribution ranks, derivative sensitivities, channel masks, activation statistics and source agreement. `screen.json`, `controls.json`, `edit-screen.json` and `symmetry-screen.json` retain every outcome, including failures. Source snapshots are in `source/`; they retain the original temporary paths.

All nine retained ONNX experiments in `rejected/` pass the strict active dense validator; `validation.json` records their exact widths, resource counts and hashes. They contain only canonical dense/activation nodes. Their metadata explicitly says rejected research candidate. `native-verification.json` checks 8,192 reachable observations per file against its dense PyTorch counterpart: zero action mismatches across 73,728 probes. Only descriptive metadata changed when archiving; graph tensors match the tested candidates.

No formal training, optimizer, teacher-label fitting, least-squares fitting, mutation search or hidden physics implementation was used. These experiments rule out the tested direct pruning, merging, local-linearization and simple-edit hypotheses; they are not an impossibility proof. The current catalog continues to use the existing eligible learned/baseline weights. There is no replacement winner to publish yet.

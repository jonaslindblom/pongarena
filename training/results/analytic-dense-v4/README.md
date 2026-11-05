# Analytic dense weights: Prism Nano

**Prism Nano v4 is a verified improvement over the three existing Nano references. It does not beat Pacer or the adaptive attacker.** It is now a catalog comparison, while the stronger compact-model goal remains open. Every result here uses the corrected dense-only contract; earlier graph-program results remain ineligible archives.

The frozen model is a genuine `6→29→9→3` ReLU network with **503 parameters, 462 MACs and five nodes**: `Gemm → Relu → Gemm → Relu → Gemm`. Six raw observations enter its first dense layer directly and three linear action scores leave its final layer. Every matrix and bias is embedded and counted. No formal training, optimizer, fitting, runtime helper, additional preprocessing or graph operator is used.

These weights are **handcrafted physical priors**, not a discovery or transfer of learned Pacer features. The separate [causal learned-weight study](../causal-weight-audit-v4/README.md) records the seven unsuccessful Pacer interventions and what they reveal about its learned features.

## Meaning of the fixed weights

The first layer has 18 ReLU ridge responses that approximate a square-difference position/velocity interaction, six signed coordinate carries, one outgoing horizontal-velocity response, two incoming-phase responses and two opponent-position responses. The second layer has five reflected-landing responses, an incoming-phase carry, two own-position carries and an incoming opponent cue. All are ordinary units in the two dense layers.

The interaction approximates straight-flight anticipation at horizontal speed 13 with a half-tick correction. A triangular response approximates wall reflection. Incoming paddle feedback adds a conservative six-court-unit aiming cue toward the opponent's side; outgoing feedback returns toward center. The final scores are opposite signed target errors, with a fixed stay deadband of **1.101 court units**.

This is an approximation expressed through neural weights. It does not evaluate exact integer contact times or solve shot planning. In particular, using the cap-speed prior on a speed-5 serve can forecast the wrong landing by about 65–70 court units. No hidden exact solver compensates for that error.

## Frozen native confirmation

SHA-256: `1eedf32a2fff0721aa101d4ba8a925a6fac7734f259daa9180ecb6523f65991a`.

The file was frozen before generating 192 new seeds, split into three blocks of 64. Every block includes five serves and both physical assignments: **640 rallies per opponent per block**, 1,920 per opponent overall. Candidate and neural references use their actual fixed-batch-one ONNX exports in native ONNX Runtime. Attack and defense are the unchanged adaptive v4 scripts. Physics, score selection, the 1,600-tick deadline and exact loop detection are unchanged.

| Opponent | Block 1 W/L/D | Block 2 W/L/D | Block 3 W/L/D | Total W/L/D |
| --- | ---: | ---: | ---: | ---: |
| Tracker | 587/53/0 | 585/55/0 | 577/63/0 | 1749/171/0 |
| Anticipator | 568/70/2 | 567/73/0 | 555/84/1 | 1690/227/3 |
| Challenger | 568/66/6 | 569/68/3 | 567/70/3 | 1704/204/12 |
| Pacer | 245/369/26 | 213/399/28 | 216/402/22 | 674/1170/76 |
| Adaptive attack | 0/606/34 | 0/600/40 | 0/605/35 | 0/1811/109 |
| Adaptive defense | 12/547/81 | 10/554/76 | 9/556/75 | 31/1657/232 |

Against each Nano reference, each block passes all declared criteria: a majority of paired ten-rally contests won, positive point-margin 95% interval resampling whole seeds, and more wins than losses on **both** sides. This is a measured improvement over those references, not an independent public championship. Pacer/attacker qualification fails in every block. No overall superiority is claimed.

`native-final/protocol.json` records the freeze, seed blocks and audit. RNG 2026110527 draws without replacement from `[1180000000,1190000000)`. A conservative scan of 370 retained JSON/JSONL files and 7,595,947 integer leaves finds no overlap. All these seeds are now consumed and must not be used to select later models. `native-final/results.json` retains all 18 pairings, per-seed outcomes and intervals, including every stronger-opponent failure.

## Development controls and numerical correction

The original 1.1-unit-deadband candidate has three PyTorch/native action differences on 8,192 probes, at mathematical threshold ties; maximum score error is about 2.9e−6. That failed check remains in `development/verification.json`. A fixed 0.001-court-unit guard creates the frozen 1.101 version. It is re-evaluated before final testing, rather than silently assigning old results to changed weights. `development/edit-verification.json` records zero action mismatches for the guarded file.

Every development pairing below has 24 seeds × five serves × both assignments = 240 rallies. The same development block is used for interpretation and hypotheses; its comparisons are not final confirmation.

| Explained candidate | Parameters | Tracker W/L/D | Anticipator W/L/D | Challenger W/L/D | Pacer W/L/D | Attack W/L/D | Defense W/L/D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial cap-speed prior, 1.1 deadband | 503 | 211/29/0 | 205/35/0 | 205/31/4 | 80/153/7 | 0/230/10 | 5/214/21 |
| Guarded cap-speed prior: frozen Prism | 503 | 213/27/0 | 205/35/0 | 205/31/4 | 80/153/7 | 0/230/10 | 5/214/21 |
| Direct far-corner aiming prior | 503 | 216/24/0 | 204/35/1 | 211/29/0 | 41/170/29 | 19/221/0 | 0/189/51 |
| Coarse edge-sensitive recovery prior | 899 | 214/26/0 | 199/41/0 | 201/35/4 | 90/145/5 | 0/233/7 | 5/214/21 |

The far-corner edit analytically changes the landing gain and opponent cue using approximate outgoing angle travel; it trades improved attacker results for worse Pacer/defense results. The recovery edit adds 12 ordinary second-layer hinges for a coarse edge-sensitive recovery curve, without a third hidden layer. Neither is promoted as a stronger overall model.

The failure audit finds 38 catchable misses across the two scripted pairings, caused mainly by missing speed dependence on early flights. Of the 444 concessions, 406 are already uncatchable at the final flight's start. Most later losses concern contact placement and recovery, despite accurate cap-speed forecasts. This motivates further speed, discrete-timing and shot-placement studies; it does not imply that fixing one diagnostic proxy will improve complete matches.

## Verification and reproduction

Run `python training/scripts/analytic_dense_v4.py --out <fresh-model-path.onnx>` with the project's numerical dependencies. The constructor requires a fresh output path and asserts the exact frozen SHA. It reproduces both matrices and descriptive metadata byte for byte. The public file and `models/prism-nano-v4.onnx` are identical.

The strict contract/hash/class regression passes. Native probes show zero action mismatches on 8,192 reachable observations for the frozen model. Browser WASM checks load all ten active presets and show zero action mismatches on another 128 observations; `browser-probes.json` retains those inputs and reference scores. All **66 application tests pass, zero skipped**, and the production build passes. Production preview loads Prism without console errors. The development server's external dependency symlink prevents one font request; the bundled production preview avoids that filesystem restriction. `checks.json` records these checks.

Raw controls, numerical checks and failure summaries are in `development/`. Source snapshots preserve their original temporary paths in `source/`; the maintained portable constructor is `training/scripts/analytic_dense_v4.py`. Later experiments must preserve the freeze and use new final seeds for any new candidate.

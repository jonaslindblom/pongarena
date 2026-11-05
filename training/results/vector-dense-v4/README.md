# Vector Nano: a dense Nano Pacer opponent

**Vector Nano v4 beats Pacer in all three fresh native test blocks using 487 parameters, versus Pacer's 17,795.** It also beats Prism Nano and the three older Nano references. It loses to Prism Small and both adaptive scripts in every block; this is not overall superiority or an attacker counter.

The frozen network is an ordinary **`6→28→9→3` ReLU MLP**, **487 parameters / 447 MACs / five nodes**. It contains exactly `Gemm → Relu → Gemm → Relu → Gemm`, with three unique fixed float32 matrices and their biases. Six raw observations enter directly; three linear scores leave. There is no runtime helper, feature adapter, branching graph, dynamic parameter, input-product operator or other inference operation. All matrix entries, including zeros, count toward the Nano budget.

No formal training, optimizer, fitted coefficients, random mutation search or learned-source transfer is used. These are transparent hand-constructed physical priors encoded in ordinary dense weights. The source Pacer weight audit is documented separately in the [causal study](../causal-weight-audit-v4/README.md); its failed transfers are not presented as the origin of this model's coefficients.

## Meaning of the weights

The first layer has 20 square-ridge responses for a position/velocity interaction, three positive coordinate carries, three direction responses and two opponent-position responses. The second layer has five approximate reflected-landing responses, an incoming-direction carry, an own-position carry and two incoming opponent cues. The output combines their votes into signed paddle-target error, with a 1.101-court-unit stay deadband. Outgoing movement targets center 80.

The opponent cue changes at integer paddle Y 79/80. It aims the continuous outgoing ray toward Y 4 when the opponent is at or below the court's middle, and toward Y 156 otherwise. At cap speed, a flight has 26 ticks and the continuous angle slope is 3/5, giving the construction-time contact goal:

```
p = (83/78) * landing - (5/78) * desired_end
desired_end = 4 if opponent_Y >= 80 else 156
```

Equivalently, in normalized coordinates the target is `(83/78)*landing_norm + (19/312)*sign`, where the dense opponent responses represent sign ±1. These equations describe the coefficients; no conditional or trajectory function executes alongside the network.

The cap timing correction is 8/13. A coarser square grid with spacing 4/9 preserves exact position/velocity cancellation at the cap-speed contact X position 20, up to float32 error. Intermediate cap frames and all slower flights remain approximate. Contact/angle quantization, paddle parity, reachable aiming and subsequent recovery can invalidate the intended ray. The opponent cue does not solve those planning problems.

## Frozen native confirmation

SHA-256: `ad41b0b8b9be1264c7b3daea95aec2341ab14f4802038ef175aea5d4898ac551`.

The candidate is frozen before generating 192 new seeds in three blocks of 64. Every cell has **640 rallies**: five serves per seed and both physical assignments. Candidate and neural references use their actual fixed-batch-one native ONNX exports; adaptive scripts and v4 rules are unchanged.

| Opponent | Block 1 W/L/D | Block 2 W/L/D | Block 3 W/L/D | Total W/L/D |
| --- | ---: | ---: | ---: | ---: |
| Tracker | 576/64/0 | 585/55/0 | 587/53/0 | 1748/172/0 |
| Anticipator | 565/75/0 | 575/65/0 | 580/60/0 | 1720/200/0 |
| Challenger | 571/69/0 | 580/60/0 | 586/54/0 | 1737/183/0 |
| Pacer | 389/182/69 | 351/221/68 | 397/178/65 | **1137/581/202** |
| Prism Nano | 447/131/62 | 457/126/57 | 462/115/63 | 1366/372/182 |
| Prism Small | 237/295/108 | 243/282/115 | 242/293/105 | **722/870/328** |
| Adaptive attack | 48/569/23 | 60/560/20 | 55/561/24 | **163/1690/67** |
| Adaptive defense | 99/362/179 | 104/359/177 | 108/354/178 | **311/1075/534** |

Against Pacer, Prism Nano and the older Nano field, every block passes majority paired-contest wins, positive point-margin 95% interval resampling whole seeds, and more rally wins than losses on both sides. Pacer intervals are `[0.2531,0.3938]`, `[0.1406,0.2656]` and `[0.2766,0.4047]`. Prism Small and script qualification fail in every block. All 24 raw pairings, including those failures, are retained in `native-final/results.json`.

`native-final/protocol.json` records RNG 2026110603, seed range `[1200000000,1210000000)`, the exact freeze and all seed blocks. A conservative scan of 411 retained JSON/JSONL files and 8,117,060 integer leaves finds zero overlap. These final seeds are now consumed and cannot select subsequent candidates. These are local native confirmations, not independently audited public standings.

## Development and verification

A single larger opposite-end control uses the finer global cap grid: **743 parameters (`6→44→9→3`)**. It scores 146/62/32 against Pacer, 19/214/7 against attack, 40/149/51 against defense and 88/107/45 against Prism Small on 24 development seeds. One explicit size/precision compression changes only the grid to the contact-aligned 4/9 form; endpoint coefficients and opponent thresholds remain fixed. The 487-parameter result scores **143/73/24** against Pacer, **171/46/23** against Prism Nano, **87/108/45** against Prism Small, **19/214/7** against attack and **40/147/53** against defense. Exported candidate versus exported Pacer reproduces 143/73/24 on those same development seeds before final confirmation.

The offline angle audit of the endpoint rule illustrates its limits: only 42/153 contact heights robustly force the intended corner under angle/parity quantization, and 72/153 have universal one-reply safety. This is a different tradeoff from Prism Small's coherent safe branch, not a general solution to reachability. No weights are selected using final outcomes.

Run `python training/scripts/vector_dense_v4.py --out <fresh-model-path.onnx>` to reproduce the exact frozen bytes and asserted SHA. Construction-time physical constants are embedded as matrices, not inference helpers. The public file and `models/vector-nano-v4.onnx` are identical. `source/` preserves the original research scripts; `development/` retains both models' complete development results, geometry and native checks.

The strict topology/hash/class regression passes. **17,442 contact, wall and opponent-threshold probes** show zero native/PyTorch or intended-formula action mismatches, with maximum native/PyTorch score error about 2.15e−6. Browser WASM checks load all twelve presets and show zero action mismatches on another 256 observations. All **68 application tests pass, none skipped**, and the production build passes. Production preview displays all three new catalog models without console errors. `checks.json` and `browser-probes.json` retain these checks.

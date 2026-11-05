# Prism Small: a compact dense Pacer opponent

**Prism Small v4 beats Pacer head-to-head in three fresh native test blocks, with about one tenth of its parameters. It still loses heavily to both adaptive scripts.** This establishes an opponent-specific compact improvement; the attacker-beating goal remains open. The frozen file is included in the catalog alongside [Prism Nano](../analytic-dense-v4/README.md).

The network is an ordinary **`6→42→33→3` ReLU MLP**, with **1,815 parameters, 1,737 MACs and five nodes**. Its graph is exactly `Gemm → Relu → Gemm → Relu → Gemm`, with three unique float32 weight matrices and their biases. It maps the raw six-input port directly to three linear scores. This version leaves the opponent-Y column at zero; it does not perform opponent-dependent aiming. All six input dimensions still count toward its parameter budget.

No formal training, optimizer, fitted weights, random mutation search, external inference helper or additional graph operation is used. The coefficients are constructed transparently from physical constraints. They are not transferred trained Pacer weights; the [separate causal study](../causal-weight-audit-v4/README.md) records those unsuccessful learned-weight interventions. Physical calculations and interval enumeration below run only during construction and diagnosis. The exported policy contains just the dense network.

## What changed in the weights

The previous Nano prior assumes cap speed and uses a generic half-tick correction. Two structural discoveries improve the dense representation:

1. **Discrete timing.** After a cap-speed return, the incoming X positions lie on `345−13n`. Thus the continuous time expression needs correction `ceil(330/13)−330/13 = 8/13`, rather than 1/2, to match contact ticks on that trajectory lattice.
2. **Cancellation inside ordinary hidden units.** Uniform piecewise-square interpolation has periodic error. With spacing `2/9`, the errors at `x_norm+vy/9` and `x_norm−vy/9` cancel for integer vertical velocities. The difference of the two sets of first-layer ReLU responses therefore represents the position/velocity interaction exactly on the physical cap-speed lattice, up to float32 error.

The first layer has 36 square-ridge responses, three positive coordinate carries and three direction responses. A numerical check covers 75,582 cap-position/height/vertical-velocity combinations: maximum unfolded forecast error is about **0.000122 court units**. This is a statement about that physical lattice, not arbitrary six-dimensional inputs or slower speeds. No runtime division, ceiling, multiplication of dynamic tensors, modulo or trajectory function is exported.

Accurate forecasts alone do not yield good contact choices. Pointwise integer minimum-risk goals are brittle because the paddle moves two units at a time. A 1.101-unit stay deadband around an integer admits three nearby centers, potentially selecting a different quantized outgoing velocity or even missing the catch. Under the same deadband, the previous integer goals are robust-safe at only **6/153** contact heights and admit an immediate miss at twelve heights.

The new contact curve uses half-integer goals and follows one connected family of safe shots:

| Contact height knot | 4 | 50 | 52 | 78 | 82 | 108 | 110 | 156 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paddle goal | 15.5 | 61.5 | 65.5 | 91.5 | 68.5 | 94.5 | 98.5 | 144.5 |

Between knots the curve is linear and vertically symmetric. Its half-integer goals admit just two adjacent stationary paddle centers. The fixed branch is robust-safe at **124/153** integer contact heights in the one-reply risk audit, with no admitted immediate catch misses. That covers every height with an adjacent safe pair for an interior goal; a separate forced-boundary exception could rescue heights 5 and 155, but is not implemented. “Safe” here is an offline one-reply physical diagnostic, not an indefinite winning guarantee.

The second layer uses 31 ordinary ridge responses for this contact curve composed with wall reflection, one incoming-direction carry and one own-position carry. The output compares the resulting target with own position using the same 1.101-unit deadband. Outgoing movement still targets center 80.

## Frozen native results

SHA-256: `64b6f14a604fd6d7ff72b91a2861b7a061f6568b6edde80d4697c2e2160e31ca`.

The source is frozen before generating 192 fresh seeds in three blocks of 64. Each cell is **640 rallies**: five serves per seed and both physical assignments. Actual frozen batch-one ONNX inference is used for the candidate and all neural references. Script opponents, rules, first-argmax action selection, timeout and exact loop detection are unchanged.

| Opponent | Block 1 W/L/D | Block 2 W/L/D | Block 3 W/L/D | Total W/L/D |
| --- | ---: | ---: | ---: | ---: |
| Tracker | 573/67/0 | 576/63/1 | 590/50/0 | 1739/180/1 |
| Anticipator | 578/62/0 | 589/50/1 | 598/42/0 | 1765/154/1 |
| Challenger | 581/59/0 | 588/52/0 | 599/41/0 | 1768/152/0 |
| Pacer | 345/175/120 | 372/146/122 | 357/150/133 | **1074/471/375** |
| Prism Nano | 379/161/100 | 377/146/117 | 384/130/126 | 1140/437/343 |
| Adaptive attack | 22/612/6 | 23/611/6 | 21/614/5 | **66/1837/17** |
| Adaptive defense | 0/565/75 | 0/559/81 | 0/557/83 | **0/1681/239** |

Against Pacer, Prism Nano and the three older Nano references, every block passes all three criteria: majority of paired ten-rally contests won, positive point-margin 95% interval resampling whole seeds, and more wins than losses on both physical sides. Pacer point-margin intervals are `[0.2109,0.3203]`, `[0.2953,0.4079]` and `[0.2672,0.3797]`. This is not transitive superiority: the scripts remain much stronger, and Prism Small performs worse against them than Pacer does in the development panel.

`native-final/protocol.json` records the freeze and seed audit. RNG 2026110530 draws without replacement from `[1190000000,1200000000)`. A conservative scan of 383 retained JSON/JSONL files and 7,806,579 integer leaves finds zero overlap. These final seeds are now consumed; later candidates require new final blocks. `native-final/results.json` retains all 21 pairings, per-seed outcomes, intervals and failed strong-target qualification.

## All intermediate controls

Each development pairing uses the same 24 interpretation seeds, five serves and both assignments: 240 rallies. These adaptively explored results are not independent final confirmations. All exports below pass the strict active dense validator; only the frozen Prism Small is promoted from this study.

| Explained control | Parameters | Pacer W/L/D | Attack W/L/D | Defense W/L/D |
| --- | ---: | ---: | ---: | ---: |
| Variable-speed polynomial approximation | 2039 | 87/130/23 | 2/238/0 | 6/194/40 |
| Variable-speed approximation with lattice tick correction | 2007 | 90/124/26 | 0/227/13 | 1/202/37 |
| Coarse contact-risk target | 1990 | 87/117/36 | 5/235/0 | 1/239/0 |
| Full-resolution integer risk target | 9740 | 104/131/5 | 10/227/3 | 0/229/11 |
| Exact cap-contact forecast, six-unit aim | 455 | 61/152/27 | 0/238/2 | 5/189/46 |
| Exact cap-contact forecast, integer risk target | 9395 | 140/97/3 | 15/225/0 | 16/224/0 |
| Global cap-lattice forecast, six-unit aim | 695 | 65/166/9 | 2/237/1 | 0/206/34 |
| Global cap-lattice forecast, integer risk target | 14419 | 144/88/8 | 5/235/0 | 4/225/11 |
| Global cap-lattice forecast, integer target and edge recovery | 14971 | 159/75/6 | 22/218/0 | 4/224/12 |
| Connected half-integer branch: Prism Small | 1815 | **144/50/46** | **9/231/0** | **0/203/37** |

The variable-speed models use six fixed polynomial ridge groups, with coefficients derived from a quadratic reciprocal approximation and fixed curvature-based knots. A later version encodes speed-specific timing through ordinary first-layer responses. Both improve some lower-speed forecasts and simple opponents, but do not establish stronger overall play. Their construction and numerical errors are retained in `development/speed/`.

## Why the attacker still wins

The full-resolution controls expose a larger problem than numerical accuracy. Many incoming flights begin where the chosen safe contact goal is already unreachable. The policy can sometimes catch the ball anyway, but its actual contact differs from the ideal and produces an unsafe reply. Recovery that merely preserves a future catch does not necessarily preserve the ability to make the next desired safe shot.

The contact audit of the integer full-risk control has 216 cap-speed attempts against attack: 137 are already uncatchable at flight start, and the ideal goal is initially reachable zero times. The edge-recovery variant improves reachable goals, yet still loses heavily. The connected branch fixes the stationary parity confound; it does not solve this longer planning problem or opponent-dependent offense. Further exploration must address those mechanisms, rather than infer match strength from one-step risk or forecast accuracy alone.

## Verification and reproduction

`python training/scripts/robust_dense_v4.py --out <fresh-model-path.onnx>` reconstructs the exact frozen bytes and asserts their SHA. It uses the existing plain dense serializer from `analytic_dense_v4.py`. Construction-time geometry is compiled into ordinary matrices; only those matrices and ReLU layers are present at inference.

The strict topology/hash/class regression passes. The branch audit checks **380,817 contact probes with zero action mismatches** and maximum target error about 0.000683 court units. Native ONNX checks show **zero action mismatches on 8,192 reachable observations**; browser WASM checks show **zero mismatches on another 256**, loading all eleven presets. All **67 application tests pass, none skipped**, and the production build passes. Production preview displays both Prism models without console errors. The development dependency symlink's font restriction remains an environment limitation; bundled production fonts load correctly.

`checks.json`, `validation.json` and `browser-probes.json` retain verification. `development/branch/` retains construction, stationary safety, native parity and the comparison to brittle integer targets. `models/controls/` and `development/analytic/` preserve every intermediate outcome. `source/` keeps original research scripts with their temporary paths; the maintained portable reproduction entry point is `robust_dense_v4.py`.

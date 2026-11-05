# Matched-serve diagnostics

These experiments use the isolated candidate engine. The arena, its current v3 physics and submitted model files are unchanged. `screen.json` uses 32 diagnostic seeds, all five serves and both controller roles (320 rallies per pair). These are selection data, not the later frozen confirmation set.

## Controllers

The two adaptive scripts observe the same six inputs as a model. They enumerate legal contact centers and, for motion deflection, final movement directions. An exact dynamic program supplies the first action of a feasible contact plan; it includes staying, clamping, partial movement at a boundary, and clamp-then-reverse paths. No geometric prediction is inserted into a neural network.

While the ball travels away, both scripts recover toward the midpoint of the interval that covers every geometrically legal opponent reply. When the interval is empty, its midpoint minimizes the largest one-flight miss margin. The defender chooses its own return to maximize subsequent recovery slack. The attacker chooses placement against projected opposing recovery, with a small penalty for its own recovery risk. These are fixed finite-horizon heuristics, not a solution of the full game. In particular, an empty interval is not a proof of forceable victory.

`interceptor`, used in the latency diagnostic, simply aims at the incoming landing and uses the same maximin recovery while the ball travels away. It does not deliberately choose a safe outgoing return.

## Diagnostic results

| Rules | Attacker mirror: scored / 320 | Median returns in those scores | First receiver won scored rallies | Attacker vs defender W / L / D |
| --- | ---: | ---: | ---: | --- |
| Baseline | 0 | — | — | 0 / 0 / 320 |
| Angle 8 | 0 | — | — | 0 / 0 / 320 |
| Angle 9 | 88 | 2 | 86.4% | 34 / 36 / 250 |
| Motion 2 | 24 | 1 | 100% | 12 / 2 / 306 |

The stronger variants create scored points, but these particular strong-controller matchups mostly score early and favor the first receiver. They do not establish the desired pattern of sustained expert rallies followed by a constructed attacking point. Baseline and Angle 8 also have independently verified never-concede strategies; see the separate invariant artifacts.

`latency.json` applies uncorrected ball/opponent observation lags of 0, 3, 5 and 8 ticks, while keeping the player's own paddle observation current. At 30 ticks/second, five ticks is 167 ms. This is **not a human playtest or a human performance estimate**: the controller does not extrapolate the known observation age, and delay can change shot selection non-monotonically. Delayed policies have memory, so their physical-state loop shortcut is disabled and the 1,600-tick deadline still applies.

The distinction between catching and return selection matters. With no observation delay, Motion 2's attacker beats the center-chasing interceptor in 314 of 320 rallies, with median 15 returns in decisive rallies; against the defender that selects safer returns it wins 12, loses 2 and draws 306. Thus this mechanic can expose a tactical difference between these controllers, while the stronger mirror still shows substantial early serve bias. With five ticks of uncompensated observation lag, the Motion 2 interceptor loses 314 of 320 with median 2 returns, illustrating sensitivity to timing rather than proving the game unplayable for humans.

A locally unreachable scoring shot means that, at its release, the receiver could not move its paddle to any legal catching center in time. This says nothing about whether its earlier recovery choices were avoidably bad. The records separate these outcomes from missed serves and physically catchable missed returns.

## Verification

- `verify_policy.py` independently reconstructs contact feasibility and verifies all 4,530,504 entries, including 1,898,388 selected first actions for multi-tick plans. It then checks 47,916 catchable incoming trajectories across both policies and all four rules; all are caught.
- `verify_invariant.py` reads the saved invariant sets without calling the solver. Using the evaluator's separate shot and contact-path implementations, it checks every retained state against every legal incoming shot: 204,483 baseline cases and 182,299 Angle 8 cases. It also checks all 1,050 standard serve height/slope cases in both roles.
- The diagnostic screen's source hash points to the exact policy source archived as `../training/search/opponent-source.py`. The later source adds terminal frames to optional replay recording only; it does not change decisions or match outcomes.

```sh
.venv/bin/python training/scripts/offense_eval.py --count 32
.venv/bin/python training/results/offense-study/evaluation/verify_policy.py
.venv/bin/python training/results/offense-study/evaluation/verify_invariant.py
.venv/bin/python training/results/offense-study/evaluation/latency_study.py
```

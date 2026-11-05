# Pong rules study: timeouts, loops, physics and match format

This is an **evaluation**, not a new rules release. The live arena, v1/v2 contracts and model weights are unchanged. The visual report is [`public/training/rule-study.html`](../../../public/training/rule-study.html), served at `/training/rule-study.html`.

## Recommendation

Keep the current paddle speed 2, horizontal return range 5–13, and contact radius 15. Of the tested changes, none clearly improves both strong-player competition and the delayed-controller benchmark. A larger paddle deserves a retraining experiment if human playtests call for more forgiveness, but the present transfer results do not justify shipping it.

The most promising next change is **a 1,600-tick rally budget plus early termination of proven loops in deterministic model-versus-model matches**. Keep draws: neither a repeated exchange nor an unresolved rally provides evidence for awarding a point. A longer deadline alone noticeably slows matches. Early loop termination needs implementation and specification before release; this study only measures its potential. Human matches cannot use the same cycle proof.

For competition, aggregate both side assignments on identical five serves, then repeat over many seeds. Keep five-rally games as a quick viewing format. Do not use return counts as a tiebreaker.

## What longer deadlines change

Each confirmation row contains 2,000 rally trajectories: 200 seeds, five openings, both side assignments. All deadline snapshots come from one continuous trajectory, not separate restarted games.

| Matchup | Draws at 800 | Draws at 1,600 | Draws at 3,200 | 800-tick draws proven cyclic by 3,200 | 800-tick draws that later score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selfplay vs itself | 832 (41.6%) | 790 (39.5%) | 790 (39.5%) | 790 | 42 |
| Selfplay vs attacker | 1,004 (50.2%) | 961 (48.05%) | 961 (48.05%) | 961 | 43 |
| Selfplay vs Apprentice | 478 (23.9%) | 190 (9.5%) | 183 (9.15%) | 183 | 295 |

For Selfplay vs Apprentice, all 295 later points are Selfplay wins. The 1,600 limit captures 288 of them (97.6%); seven more occur by tick 1,716. The 800 cutoff therefore obscures a real strength difference. For Selfplay vs attacker, all 43 later points are attacker wins. Longer survival is not evidence that Selfplay has become the stronger policy.

At 3,200, every remaining draw in every baseline pairing has a proven repeated state. This is a result for these controllers and serves, not proof that any future pair of submissions will repeat by that deadline. At 1,600, 15 of the attacker matchup's eventual cycles have not yet been detected, and seven Apprentice rallies are still genuinely unfinished.

| Matchup | Current 800 budget: mean seconds | 1,600 budget alone | 1,600 + proven model-loop exit |
| --- | ---: | ---: | ---: |
| Selfplay vs itself | 16.53 | 27.16 | 13.30 |
| Selfplay vs attacker | 17.26 | 30.17 | 12.87 |
| Selfplay vs Apprentice | 13.02 | 16.93 | 13.34 |

Times are simulation ticks / 30, excluding inference, rendering and pauses. The shortcut ends a cycle when its repeated state is detected; it does not pretend the cycle was known at its first occurrence. It preserves point outcomes for the chosen deadline, but changes episode length and thus may change training returns. A training implementation must make that choice explicit.

## Physics comparison

These confirmation results use the 800-tick deadline so that only physics changes. Median returns include **scored rallies only**. Early finishes are points before the second return, as a share of all rallies. First-receiver percentages are meaningful as serve-advantage diagnostics in the identical-attacker pairing.

| Candidate | Attacker mirror: median returns | Attacker mirror: draws | First receiver wins, scored only | Attacker wins vs interceptor | Delayed defender: early finishes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current v2 | 6 | 0% | 53.0% | 63.5% | 22.65% |
| Larger paddle | 6 | 2.2% | 48.9% | 65.5% | 20.25% |
| Paddle speed 2.25, symmetric rounding | — | 100% | — | 0% | 11.9% |
| Paddle speed 2.5, symmetric rounding | — | 100% | — | 0% | 2.25% |
| Horizontal cap 14 | 3 | 7.3% | 70.7% | 80.4% | 28.75% |
| Minimum return speed 6 | 4 | 15.5% | 60.5% | 86.35% | 26.95% |

- **Faster paddles:** the delayed defender lasts longer, but the tested attackers cannot finish against each other or the interceptor. This is not a proof that no stronger policy could score. At speed 2.5, a centered paddle can traverse 65 units in the fastest 26-tick flight, exceeding the 61 needed to cover either wall landing after the contact allowance. The transition toward easy defense can be sharp.
- **Faster ball / faster center returns:** more early points, shorter scored rallies, and more first-receiver advantage. They do not solve the requested balance problem.
- **Larger paddle:** radius 15→17, center bounds [15,145]→[17,143], and shot response normalized to the larger radius. This corresponds to increasing the drawn paddle from 24 to 28 units with the same ball extent. Attacker-vs-interceptor median returns improve 7→9, but attacker-mirror median stays 6. Frozen Selfplay mirror draws increase 41.6%→88.5%. These actors were trained for v2; this is a transfer warning, not a claim about retrained policies' ceiling.
- The initial scan also tested cap 12 (92.7% attacker-mirror draws), paddle 2.25 + cap 14 (80.3%), and the ordinary-rounding versions of both faster paddles (100%). Those candidates were not advanced to confirmation.

Fractional paddle speeds create offsets such as ±1.5. Existing `floor(x + 0.5)` rounding gives vertical velocities 0 and +1 for those two contacts; current integer-offset v2 has no such half ties. The symmetric variants explicitly use `sign(x) * floor(abs(x) + 0.5)` for vertical shots. This is an additional stated rule change, not a silent correction. Both rounding variants failed the strong-script scoring criterion in the scan.

The delayed defender sees a five-tick-old observation, including its own paddle position. It is a deliberately limited sensitivity proxy, **not a human playtest or human win-rate prediction**. Real keyboard play remains unmeasured in this study.

## Match format and inputs

Five serves give one player three initial receptions and the other two. With two identical attackers, all 400 confirmation matches finish decisively at five rallies, split 200/200. Pairing the two side assignments yields 200 tied ten-rally contests, correctly reflecting identical policies. More ties under that paired format can indicate less side/serve noise rather than worse gameplay.

At 800 ticks, Selfplay vs Apprentice produces 313/35/52 five-rally match wins/losses/draws, versus 185/6/9 paired contests. At 1,600 the corresponding counts are 352/22/26 and 193/1/6. These formats have different numbers of contests; compare proportions and remember they reuse the same data. A leaderboard still needs multiple seeds and varied opponents.

Zero points each for a drawn rally and half a point each give identical point margins with a fixed number of rallies. There is no fairness benefit from changing that arithmetic. A draw does not assert equal model quality; it says the observed contest did not establish a winner.

The six-input contract omits time, rally index and match score. It tests stationary rally policies, not clock management or protecting a match lead. Adding these inputs would require a versioned interface and retraining. A hidden speed ramp would also change future dynamics without exposing its governing state. Neither was introduced here.

Training choices are separate: the current PPO recipe uses discount .99 and +.02 per return. Reward/discount ablations may improve offense without changing physics. No model was retrained in this study.

## Method, scope and reproducibility

`rule_study.py` is an isolated experimental engine. It preserves v2 tick order; the timeout is applied when summarizing a continuous run. A point on the exact boundary wins over the draw. Both policies choose from the same pre-move state. Scripts adapt movement, reachability and shots to each candidate. Paddle coordinates are decoded on the exact quarter-unit grid through the same float32 observations. The attacker remains a heuristic: its direct movement candidates omit some clamp-then-reverse paths, and it does not plan multi-exchange strategies.

At successful contacts, store exact float64 ball position/velocity plus both paddle positions. A repeated key proves an endless future cycle for deterministic stateless controllers under these stationary rules. The delayed controller additionally contributes its full six-observation history to the key. Counters are excluded because they affect neither decisions nor candidate physics. Human decisions, recurrent policies, random sampling or future clock-dependent rules would invalidate this reduced key. No-cycle-detected is not treated as proof of eventual scoring.

Cycles can stop simulation early. Draw duration statistics still use the full requested deadline; scored-return statistics exclude draws. Separate shortcut timing explicitly uses the first repeated-state detection time. Raw records retain point time/returns and cycle start/detection/period for every lane, ordered side, seed, rally.

- Scan: RNG seed 117; 60 unique dispersed serve seeds; 10 candidates × 8 pairings × 600 trajectories = 48,000 trajectories.
- Confirmation: RNG seed 118; 200 different dispersed seeds; 6 candidates × 8 pairings × 2,000 trajectories = 96,000 trajectories.
- Both use seeds sampled without replacement from [3,200,000,000,4,200,000,000), disjoint from prior training, validation, model-test and balance intervals. The two actual seed lists were checked disjoint.
- Each trajectory supplies 800/1,600/3,200 snapshots. Paired sides, especially same-model mirrors, are dependent observations; percentages are descriptive, not independent-rally confidence estimates.
- Policies: A attacker, D interceptor, H delayed interceptor, N released v2 Selfplay, P released v2 Apprentice, T Tracker. Learned weights are frozen. Batched PyTorch inference uses one CPU thread and deterministic first argmax. Result JSON contains rule parameters and source/checkpoint SHA256 hashes.

```sh
.venv/bin/python training/scripts/check_rule_study.py
.venv/bin/python training/scripts/rule_study.py
.venv/bin/python training/scripts/rule_study.py --count 200 --seed 118 \
  --configs v2 larger-paddle paddle2.25-symmetric paddle2.5-symmetric cap14 center6 \
  --out training/results/rule-study/holdout.json
.venv/bin/python training/scripts/rule_study_report.py
npm test
```

Verification passed: 975 fixture and 32,000 randomized physics transitions; 128,000 baseline scripted decisions; exact boundary precedence and vertical-symmetry checks; reproduction of released baseline metrics across 2,000 rallies; and 46 cycle shortcuts compared with full 3,200-tick simulation, including delayed history. Both actual ONNX exports agree on every action across 8,192 reachable observations per actor from baseline/fractional/larger-paddle games (maximum score errors 7.87e−6 / 7.63e−6). Native ONNX sampling does not establish exhaustive browser-WASM parity for every experimental trajectory. All 14 existing arena tests pass, none skipped.

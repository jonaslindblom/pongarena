# Gradual-speed Pong tuning

Selected candidate: **pace13-v9-scaled**, for the proposed Pong v4 release. This study changes the return law only; paddle movement, simultaneous decisions, serve generator, observation/action meanings, rally deadline and exact loop detection retain their existing definitions.

On a legal contact, let `d` be ball Y minus the updated paddle center, and let `s = min(13, abs(incoming_vx) + 1)`. Then:

```
vx = (+1 for left paddle, -1 for right paddle) * s
vy = floor(9 * s * d / 195 + 0.5)
```

The first return has horizontal speed 6, regardless of contact offset. Its maximum absolute vertical speed is 4. Horizontal speed reaches 13 only after eight returns, and a center hit never resets it. The velocity already in the six observations determines every future speed increase; no return counter or extra input is required. All reachable physical state remains integer-valued. The maximum vertical speed at cap is 9; velocity observations remain divided by 6 and must not be clipped.

This keeps approximately the same aiming angle as the ball speeds up, subject to explicit integer rounding. The first cross-court flight lasts 55 ticks (1.83 seconds); the fastest lasts 26 ticks (0.87 seconds). The angle scaling avoids the unscaled candidate's early `vx=6, vy=9` edge return and its multiple wall crossings.

## Selection and confirmation

The first diagnostic sweep compares the unchanged baseline with caps 11–14 and vertical factors 5–9, all using one speed increment per return. Nearby caps and angle factors often retain a provable defensive strategy or simply move serve advantage to the speed threshold. A second diagnostic compares scaled factors 8 and 9 at cap 13. Both screens use the same 32 seeds, five serves and both side assignments, 320 rallies per matchup.

The scaled factor-9 candidate was selected before a separate 64-seed confirmation run. `selection.json` records the source hash and rationale. `seed-check.json` verifies separation from the screen and existing archived JSON seed lists. The final comparison includes the baseline, unscaled factor-9 candidate and selected scaled candidate, on 640 rallies per matchup.

| Selected scaled candidate | W / L / D | Median returns in scored rallies | First receiver share among scores |
| --- | --- | ---: | ---: |
| Adaptive attacker vs itself | 320 / 320 / 0 | 11 | 60.3% |
| Adaptive attacker vs defender | 384 / 119 / 137 | 11 | 60.0% |
| Adaptive defender vs itself | 320 / 320 / 0 | 12 | 71.6% |
| Adaptive attacker vs interceptor | 593 / 47 / 0 | 8 | 42.7% |
| Frozen Contender vs adaptive attacker | 28 / 612 / 0 | 7 | 50.3% |
| Frozen Contender vs adaptive defender | 0 / 555 / 85 | 6 | 45.9% |

All three strong-script pairings have zero points before the second return. Their scored rallies last at least seven returns. Attacker-mirror rallies range from seven to nineteen returns, with a median duration of 13.67 seconds. Every scored return in those three pairings is locally unreachable by its receiver from the position at release. This establishes placement/recovery consequences in these controller matchups; it does not prove optimal multi-exchange strategy or rule out better defense.

The old unscaled factor-9 candidate scores in 72.8% of attacker mirrors with a median of nine returns and 63.5% first-receiver share. The selected scaled candidate gives a gentler first return and longer strong exchanges. First-receiver advantage remains measurable, especially in defender mirrors. Paired side assignments remain appropriate for model ranking; the selected values are a practical improvement, not a claim of perfect balance.

## Stronger defensive check

The stage-aware controlled-invariant calculation solves a conservative safety game at the cap, then works backward through earlier speeds. For each opponent contact height and own paddle center, every geometrically legal opposing return must allow an exactly reachable own return and recovery into the retained state set. All legal opposing placements are considered even if an opponent's prior position would make some inaccessible.

This proves never-concede play for several attractive-looking candidates, including cap 13 with unscaled vertical factors 5 and 6. These were not chosen despite some finite scripts scoring under them. The selected factor-9 scaled candidate retains only two safe contact-height subsets at the cap, and neither standard-serve role can enter a closed safe strategy under this abstraction. All 1,050 serve height/slope cases fail this sufficient test for each role. Failure of the sufficient test is **not** a proof that optimal attack wins.

## Verification

`verify.py` supplies an independently written scalar transition. It compares 45,493 random scalar/vector transitions across the baseline and selected candidate, plus 1,082,322 exhaustive contact transitions for the selected candidate. Checks explicitly enforce offset-independent speed, the gentle first return, speed cap, unchanged baseline behavior, speed as part of the loop key, and point-before-loop-before-timeout precedence.

`fixtures.json` retains 15 complete selected-rule rallies, all five serves for seeds 11, 2026 and 4,294,967,295, with recorded pre-step states/actions and terminal states. `verification.json` records counts and the candidate source hash. A separate implementation should replay the actions, not reproduce the diagnostic controllers.

The delayed interceptor is a five-tick uncompensated ball/opponent-observation lag with current own-paddle position. It is a sensitivity check, **not a human playtest**. Neither this numerical study nor the learned-model training demonstrates subjective human enjoyment; actual keyboard and touch sessions remain necessary.

## Reproduce

```sh
.venv/bin/python training/scripts/pace_study.py --out training/results/pace-study/new-screen.json
.venv/bin/python training/scripts/pace_study.py --configs pace13-v9-scaled pace13-v8-scaled --out training/results/pace-study/new-scaled-screen.json
.venv/bin/python training/results/pace-study/verify.py
```

The retained confirmation command was:

```sh
.venv/bin/python training/scripts/pace_study.py \
  --configs baseline pace13-v9 pace13-v9-scaled --confirmation --count 64 \
  --pairs attack:attack attack:defend defend:defend attack:interceptor attack:delayed contender:attack contender:defend \
  --out training/results/pace-study/confirmation.json
```

Do not use these confirmation seeds to tune a new candidate. The script's `PaceArena`, `make_policy` and `run_pair` APIs also support training and separate model evaluation. Diagnostic geometric controllers are never embedded in the exported learned networks.

New fully learned Pacer/Apprentice weights, their DAgger method, retained source snapshots, seed separation and final evaluation are documented in [training/README.md](training/README.md). The generated public report is `/training/pace-study.html`.

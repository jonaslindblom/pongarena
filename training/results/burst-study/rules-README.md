# Burst preview: rules, pressure and counterplay

Rules ID **pong-burst-v1**, candidate **burst96-cost8-half**. This study keeps Classic v4 frozen and supplies a separate playable preview. It is not a proof of optimal balance and does not show that a learned network has mastered the demonstrated tactics.

The standalone contract is `training/burst/README.md`; downloadable package `/training/pong-burst-v1.zip`. Open `/training/burst-rules.html` for synchronized pressure/counter replays.

## Why this candidate

The user found the ordinary paddle sluggish and the automatic escalation too quick. The preview therefore gives a short movement burst without granting unlimited faster movement, and halves the horizontal speed increment per return. Permanently increasing paddle speed to 3 with the old cap 13 would permit a simple recenter/intercept defense: 26 ticks at speed 3 covers 78 units, enough both to return from any paddle center to midcourt and to catch any incoming ball from there.

The retained candidate has normal movement 2, burst movement 4, capacity 96, burst cost 8, and normal-action recharge 1. A full meter supplies 12 consecutive burst ticks. An unsuccessful burst request moves normally and does not recharge. Horizontal speed increases by 0.5 per return to cap 13; placement sets vertical deflection using `floor(9 * nextSpeed * offset / 195 + 0.5)`. Energy is observable and included in the exact loop key.

`rules-budget-screen.json` compares four budgets on the same eight seeds, all five serves and both sides. These are diagnostic screens, not fresh final tests:

| Capacity / cost | Attacker mirror median scored returns | Attacker vs stronger defender W/L/D | Median scored returns |
| --- | ---: | --- | ---: |
| **96 / 8** | **19** | **52 / 0 / 28** | **26** |
| 64 / 8 | 19 | 52 / 0 / 28 | 22 |
| 64 / 12 | 17 | 52 / 11 / 17 | 20 |
| 96 / 12 | 18 | 52 / 0 / 28 | 20 |

96/8 retains the largest immediate recovery window and longest pressured exchanges in this narrow screen. This practical preview choice preserves the physics used by the separate initial model-size experiment. `rules-selection.json` records the rationale and source hash. No physical human or phone playtest was performed.

## Fresh rule confirmation

Thirty-two fresh seeds from [2.83,2.84) billion, generated with NumPy seed 20261214, produce 320 rallies per pairing. Every seed uses all five serves and both side assignments. These seeds were not used to pick parameters or train weights. `rules-seed-check.json` checks exact separation from diagnostic rule and model-development seeds.

| Pairing | W / L / D | Median returns in scored rallies |
| --- | --- | ---: |
| Attacker vs itself | 160 / 160 / 0 | 19 |
| Attacker vs reply-envelope defender | 204 / 0 / 116 | 26 |
| Reply-envelope defender vs itself | 0 / 0 / 320 | — |

There are no scored rallies ending before the second return in these three pairings. Two stronger defenders can still maintain exact loops; they are not forced to attack. More capable or multi-exchange defenders may avoid additional points. The original training teacher and its simple energy/travel defender are separately retained in `rules-screen.json`.

The stronger diagnostic controller in `burst_tactics.py` enumerates every legal opposing reply angle after a candidate return and estimates energy-aware recovery. It is still a finite-horizon heuristic. It does not solve the complete resource game or exhaust all future opposing choices. The frozen `burst_rules.py` training teacher is unchanged, so training results and teacher hashes remain reproducible.

## An actual pressure sequence, with a legal counter

`rules-tactics.json` contains a diagnostic match at seed **2,828,560,875**, rally 1. The left attacker repeatedly sends the ball to the right defender's top edge. Selected right contacts:

| Tick | Ball Y | Defender center | Defender energy after return |
| --- | ---: | ---: | ---: |
| 701 | 4 | 18 | 64 |
| 753 | 4 | 18 | 44 |
| 805 | 4 | 18 | 24 |
| 857 | 4 | 18 | 4 |

At tick 883 the attacker changes from the top to a shot landing at Y=152. The defender is at Y=76 with energy 3 and 26 ticks available. Exhaustive action reachability, retaining maximum energy for every reachable paddle center, proves that this exact flight cannot be returned. Restoring energy to 96 makes the same flight returnable. The actual score arrives at tick 909 after 26 returns.

This is not merely a prerecorded losing defender. At tick 727 we branch over **every exactly reachable contact center from 15 through 19**. Each branch uses a maximum-remaining-energy legal action path to that contact, then resumes the same two policies. The usual center-18 return leaves energy 44 and eventually loses. The center-19 branch deliberately spends more energy, leaving only 8, but its altered angle breaks the pressure pattern and reaches an exact loop draw after 30 returns. By tick 831, only center 18 remains reachable for that incoming ball.

These branches establish a **local, reproducible resource consequence and placement counter**. They do not establish that the earlier attacker can force that entire sequence against every possible response. Nor do they establish that a small or large learned network chooses the counter. Learned policies must be tested separately on the same snapshots and in complete held-out games.

## Verification

- `burst_check.py` checks 12,000 randomized transitions against an independently written scalar reference, both float32 observations, resource updates, loop key inclusion and depleted-request semantics.
- Three complete five-rally matches reproduce across the scalar and vector engines: 11,893 action transitions.
- The standalone fixtures retain 102 isolated examples plus 11,905 complete match frames including next serves. `training/burst/check.ts` and `check.py` reproduce every frame independently.
- The single-flight resource counterfactual and legal counter paths are checked by exhaustive discrete action reachability, including clamping and energy recharge. More remaining energy weakly dominates less at a fixed center, so it suffices to retain its maximum.
- `rules-tactics.json` retains complete branches, not only the favorable return.
- `burst_tactic_check.py` independently reproduces all nine alternative branches, 1,092 transitions, after restoring the complete prefix and its loop history. The reported placement counter still draws with that history restored.

`rules-sources/` preserves the exact three source hashes recorded by the confirmation. The later branch helper restores the full earlier loop history; it leaves the confirmed controllers and game physics unchanged.

## Reproduce

```sh
.venv/bin/python training/scripts/burst_check.py
python3 training/burst/check.py
node --import tsx training/burst/check.ts
.venv/bin/python training/scripts/burst_tactics.py
python3 training/scripts/burst_tactic_check.py
.venv/bin/python training/scripts/burst_study.py --confirmation --count 32 --out /tmp/burst-rule-confirmation.json
python3 training/scripts/package_burst.py
python3 training/scripts/burst_report.py
```

The rule study refuses to overwrite an existing output file. Its fresh output can reproduce the retained confirmation; that does not make those seeds fresh for a future tuning decision. Future changes require a new rules version, updated fixtures and new reserved evaluation seeds. Keep this competition separate from Classic v4, and keep scripted diagnostic findings separate from learned-policy performance.

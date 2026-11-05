# Compact predictive counter to the v4 attacker

This study constructs one fixed Small-class policy that approximates the adaptive attacker's shot selection and anticipates its recovery. It uses only the unchanged six observations and produces the ordinary up/stay/down scores. The weights are hand-constructed: no gradient updates, optimizer, distillation, or mutation search are used. Simulated matches are development and evaluation evidence.

Commit `a4772c3` added the earlier **Handcrafted Nano v4** and **Handcrafted Small v4** to the app catalog. The counter described here is catalogued separately as **Counter Small v4**, after passing its own three fresh blocks against the unchanged attacker. Its strength is assessed from that frozen evaluation; the earlier models' Pacer results do not establish its performance against the attacker. Pacer versus Apprentice remains the default matchup.

## Frozen model and resource budget

| Property | Counter |
| --- | ---: |
| Embedded float32 parameters | 1,330 |
| MACs per decision | 8,137 |
| ONNX operators | 253 |
| ONNX bytes | 15,341 |
| Smallest class | Small |

Small permits 2,048 parameters and 8,192 MACs; the common graph limit is 256 operators. This graph is close to the compute and operator limits. It does not fit Nano's 512-parameter / 2,048-MAC budget. Resource counts come from the actual arena validator, including all embedded constants and repeated operations; they are not runtime latency measurements.

Frozen ONNX SHA-256: `aeb3531d64f5ec06f4be4acf6d62474a5c8e6d420e0fec621945aaff7af78e86`.

## What the counter approximates

The earlier catalog models predicted landing heights accurately, but their incoming targets did not adapt to the opponent's position. The actual attacker in `pace_study.py` enumerates reachable contact placements, forecasts the opponent's recovery, and chooses a shot with a large one-flight reachability margin, with smaller recovery-risk and travel penalties.

The counter retains the compact exact-flight construction: recover integer speed, interpolate its reciprocal at legal speeds, compute the integer contact time with binary ReLU ceiling ramps, multiply by vertical velocity, and reflect at the walls. It then considers all 19 possible outgoing integer vertical velocities, −9 through +9. Each velocity corresponds to a band of legal contact centers. The graph intersects that band with paddle bounds and contact offsets and quantizes it onto the actor's ordinary two-unit movement lattice.

For each possible first move, the graph checks whether the remaining ticks can reach a candidate band. It scores the resulting shot by distance from the projected receiver position, penalizes infeasibility, and includes smaller travel and subsequent-recovery-risk terms. A pairwise maximum reduction finds the best score for each action; exact equal scores prefer staying.

While outgoing, the graph forecasts the opponent's reachable extreme landing heights. It predicts which endpoint is farther from the recovery point that the attacker expects us to approach, and moves toward that endpoint. This is an explicit model of the attacker's finite-horizon heuristic, not a solved minimax policy or a learned opponent model.

## Deliberate approximations

- **Wall-clamp parity changes are omitted.** The ordinary two-unit lattice is retained; a clamp followed by reversal can make additional centers reachable. This is a conservative omission relative to the exact `contact_paths()` enumeration used by the script.
- **Recovery and safety use the cap-13 tables.** Flight and angle calculations use the actual speed stage, but the compact risk/recovery prior remains the cap-speed approximation during early exchanges.
- **Outgoing prediction uses endpoints.** It does not reproduce the attacker's full choice among every reachable return or plan a multi-return sequence. Its prediction can be wrong when the attacker chooses another shot.
- **Local placement is not a universal guarantee.** Winning a fixed matchup does not establish optimal play, dominance over other policies, or a minimum model size.

## Development and numerical corrections

The exploratory screen uses 24 development seeds from RNG `2026110511` in `[4,100,000,000, 4,150,000,000)`, five serves on both sides: 240 rallies per pairing. The final integer-band candidate scored **144 / 86 / 10** against attack, **207 / 31 / 2** against Pacer, and **228 / 12 / 0** against the catalog Small model on that development block. These are development results, not final evidence.

Two numerical issues were corrected before the final freeze:

1. The prototype's `1e-6` up/down tie penalty was below float32 maximum-reduction noise; independent probes found intended stay decisions could flip. An intermediate correction changed that one initializer to the binary penalty `1/1024`. Source cleanup before the correction was verified byte-identical to the exploratory export.
2. Real-valued upper contact bounds left fractional risk/travel terms even on integer game states. The final graph quantizes both ends of the contact band onto the same movement lattice, while preserving the original empty-band gap so an empty interval cannot become feasible through quantization. It scales the incoming quality formula by 2,000: geometry and risk terms become exact integers within float32's exact-integer range. A final half-point up/down penalty therefore favors stay only when the integer best-action scores tie. This replaces the intermediate small penalty; it is an analytical numerical correction, not a fitted weight.

The corrected graph independently matched the discrete incoming reference actions on 30,224 probes with zero action mismatches. Development variants and numeric failures must remain separate from final-seed evidence.

## Frozen final evaluation

The fixed candidate passes the declared primary criteria in all three fresh blocks against the unchanged adaptive attacker. Across 1,920 rallies it scores **1,203 wins, 649 losses, and 68 draws**.

| Attacker block | Rally W / L / D | Paired contest W / L / D | Point-margin 95% CI |
| --- | ---: | ---: | ---: |
| 1 | 391 / 229 / 20 | 44 / 8 / 12 | +0.186 to +0.323 |
| 2 | 413 / 206 / 21 | 52 / 8 / 4 | +0.250 to +0.388 |
| 3 | 399 / 214 / 27 | 49 / 8 / 7 | +0.222 to +0.358 |

The first final block supplies 640 rallies per secondary pairing:

| Opponent | Counter W / L / D | Point-margin 95% CI |
| --- | ---: | ---: |
| defend | 326 / 243 / 71 | +0.075 to +0.184 |
| pacer-v4 | 570 / 66 / 4 | +0.733 to +0.839 |
| handcrafted-nano-v4 | 561 / 79 / 0 | +0.706 to +0.797 |
| handcrafted-small-v4 | 601 / 39 / 0 | +0.837 to +0.916 |

These comparisons establish a positive tested matchup against the actual attacker, including both sides, rather than a guarantee against any future policy or a proof of universal superiority.

The declared primary target is the **actual unchanged adaptive attack policy** from `pace_study.py` under Classic v4. `freeze.json` locks the ONNX, source/reference hashes, and three separate 64-seed blocks before final gameplay. Its conservative archive scan found no final-seed overlap in retained integer or numeric-string leaves.

The primary criterion requires all three blocks to have a positive whole-seed point-margin 95% confidence interval, more rally wins than losses on each physical side, and a majority of paired ten-rally contests won. Every seed supplies all five rallies in both assignments. Secondary tests use the first final block against adaptive defense, Pacer, and both earlier compact handcrafted models. The evaluator runs the actual static batch-one native CPU ONNX export with deterministic first argmax and exact loop detection. These are matchup tests, not a complete league or a public leaderboard.

## Independent numerical, engine, and browser verification

`independent-verification.json` records 8,192 broad numeric probes plus 22,032 action-boundary probes: **30,224 total**, all with zero intended-action mismatches. A further **52,326 lattice-band cases** preserve the legal centers and empty-band gap. Incoming scores lie on the intended half-integer grid.

Six complete native matches contain 10,296 frames, with 12,296 Counter decisions, 1,608 Pacer decisions, and 6,640 adaptive-script decisions. Both the frozen v4 and live TypeScript engines reproduce every transition: 20,592 checked transition states, zero state/action mismatches. The model regenerates byte-identically and all frozen sources/reference hashes remain unchanged. Complete match evidence is retained in `independent-replays.json.gz`; exact numerical and verification source snapshots are in `verification-source/`.

`browser-verification.json` replays those six matches in ONNX Runtime Web WASM using live Counter/Pacer inference, retained native adaptive-attacker controls, and the live TypeScript v4 engine. All 13,904 ONNX decisions retain their actions and all states match. Counter scores match native exactly across its 12,296 decisions; Pacer's maximum score difference is 7.63e−6 with no action differences.

## Construction and retained artifacts

`training/scripts/counter_attacker_v4.py` exposes `build(path)` and a non-mutating `validate(path)`. The CLI constructs into a fresh directory, checks the actual arena budget and finite zero-input scores, and writes `verification.json`:

```sh
.venv/bin/python training/scripts/counter_attacker_v4.py --out /tmp/counter-v4-build
```

The command performs no training. It needs the existing project Python and npm dependencies and refuses to overwrite a result directory. `evaluate.py` is the recorded freeze/test entry point and deliberately refuses to overwrite `freeze.json`; do not delete frozen evidence to rerun it. The final blocks are consumed as evaluation runs and must not be reused for further model selection.

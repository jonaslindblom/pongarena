# Offensive balance: current v3 has a simple defensive ceiling

The arena and its rules remain unchanged. This investigation distinguishes success against the selected opponents from offensive depth in the game itself. The earlier training results remain valid measurements, but winning those matchups does not establish that skilled attack can force a point against sound defense.

## Contender's improvement is mostly defensive

On the same 1,000 diagnostic rallies against v2, Contender scores 401 points and concedes 17; its preceding fully learned PPO checkpoint scores 396 and concedes 108. The 96-point net improvement decomposes into five more scores and 91 fewer concessions. This is descriptive, not a causal ablation. Of Contender's 401 scores, 244 follow return flights that were physically catchable at release and 44 are missed serves. The remaining 113 return flights were already unreachable from the receiver's actual position when struck.

Against the attacker, Contender's 214 scoring returns were all locally unreachable at release. These are real placement wins against that controller. They are nevertheless avoidable through better earlier recovery: the attacker always recovers toward center80, which lies outside the safe region for some replies. [The point audit](README.md) records exact definitions, matched-seed counts, limitations and verification.

## A constructive guarantee, not just a draw-rate observation

Enumerate every legal integer contact height and offset. There are 4,041 legal returns under current physics. For a prospective shot landing at height `z` after `t` ticks, a receiver at center `p` can catch it when its reachable paddle interval comes within 15 units of `z`. Intersect those starting-position sets over every legal return at a given contact height: this is the universal safe band.

For all 153 contact heights, this band is nonempty. The narrowest spans 30 units (31 integer positions). After **every** legal return, the hitter can reach the safe band for the ball's next contact height before the opponent hits. All standard serve cases also satisfy the required first interception/recovery condition. A stateless controller can therefore:

1. Catch the incoming ball by moving toward its exact landing.
2. While the ball travels away, move into the safe band for its known next contact height.
3. Catch whatever legal reply the opponent chooses, then repeat.

This proves by induction that either side can avoid conceding from a standard serve against any legal sequence of opponent actions. A deadline or valid loop exit draws the rally; it cannot force this defender to miss. The argument relies on the current discrete physics, instantaneous paddle direction changes, exact observations and legal actions. It is not a claim about human execution or the representational accuracy of an arbitrary neural network.

The script verifies 529,371 exact discrete reachability cases, including staying and clamp-then-reverse movement; 8,082 independent scalar-engine flights; 204,483 greedy interception cases; all 4,041 recoveries; and 1,050 serve height/slope cases in both roles. An independent set-based enumeration also reproduced the result. [geometry.json](geometry.json) retains the induction and its coverage. The constructed defender conceded zero points across 1,000 rallies each against Contender, v2 and attacker, corroborating the mechanism; that finite benchmark alone is not the proof.

Optimal play drawing is not by itself a reason to reject a game. The concern here is that a short-horizon geometric recovery policy guarantees survival with generous movement margins, removing the need to accept meaningful defensive risk. The previous balance studies tested imperfect attacking scripts and missed this invariant.

## Candidate direction: contact-motion deflection

The first separate prototype to test should keep horizontal ball speed and paddle speed, but let actual paddle movement on the contact tick affect outgoing vertical velocity. For example, moving upward adds upward deflection and moving downward adds downward deflection. This changes the return once; it does not introduce persistent spin or hidden state. The current six observations remain sufficient because the outgoing velocity is already observable. A rules revision and retraining would still be required.

The isolated geometry scan below is a hypothesis filter, not a simulated tournament or playtest. For deflection amount `k`, it uses `vy = round(offset/3) + k × movement_direction`, with actual movement accounting for paddle clamps. Maximum absolute `vy` becomes `5+k`; horizontal speed remains unchanged. A shot with an empty safe band does not by itself prove that an attacker can force the game into that state.

| Geometry hypothesis | Shortest cross-court flight | Contact heights lacking a universal safe band | Recovery failures for this band strategy |
| --- | ---: | ---: | ---: |
| Current cap13 | 26 ticks | 0 / 153 | 0 / 4,041 |
| Horizontal cap14 | 24 ticks | 0 / 153 | 0 / 4,041 |
| Horizontal cap15 | 22 ticks | 0 / 153 | 20 / 4,041 |
| Horizontal cap16 | 21 ticks | 33 / 153 | 859 / 4,041 |
| Cap13 + contact deflection1 | 26 ticks | 0 / 153 | 0 / 12,013 |
| Cap13 + contact deflection2 | 26 ticks | 27 / 153 | 2,287 / 12,013 |

A small horizontal increase leaves the defensive construction intact. Larger increases shorten human reaction time and the earlier balance study already found more early points and first-receiver advantage. Deflection2 is a more promising first hypothesis because it breaks this particular universal-coverage construction without shortening cross-court flight times. It could still be too forgiving, too punishing, or defensible by a different strategy. These calculations do not establish balance.

Other candidates include a paddle reversal cost or acceleration, but paddle velocities would then need to become observations and part of the loop-state key. Removing draws, adding rally-length tiebreakers, or simply extending the clock does not create an offensive choice. Any changing-speed pressure rule must expose the state that governs it.

Before adopting a revision, test whether reachable multi-exchange attacks defeat adaptive defense, not only the original attacker/interceptor. Measure short serve-dominated points, scored-rally lengths, side/serve advantage, draw rates and recovery/shot-choice tradeoffs on matched serves. Retrain plain networks for each serious candidate; existing v3 weights are only transfer diagnostics. Keep the current reaction-time window as an explicit constraint and include real keyboard playtests. Select rules for interesting decisions at several skill levels, not just a higher win rate for one policy.

## Reproduce

```sh
.venv/bin/python training/scripts/analyze_points.py
.venv/bin/python training/scripts/attack_geometry.py
.venv/bin/python training/scripts/analyze_points.py --safe-defender
.venv/bin/python training/scripts/attack_variants.py
```

These scripts write only diagnostic artifacts under `training/results/attack-audit/`; they do not modify the arena, trained weights, reference opponents or frozen training contracts.

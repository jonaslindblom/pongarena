# Evidence for retained weights

The current roster was curated for eligibility and useful size/strength comparisons. No new performance ranking was run during cleanup. Exact model identities and archived source paths are in [catalog.json](catalog.json).

- **Robust Strong:** the preselected `follow-strong-league-6211` model gains 3.63 rally-score percentage points over the prior full-size winner on the six-reference confirmation benchmark; direct games are 337 wins, 169 losses and 6 draws. It remains below 50% rally score against Attacker and Defender. Original evidence: `training/results/robustness-lab/README.md`, `evaluation/confirmation.json`, `evaluation/verification.json`, `releases/manifest.json` inside the archive.
- **Plain Nano:** the preselected `precise-cma-7151` model has 447 parameters and 408 MACs. Its original confirmation gives 332 wins, 145 losses and 35 draws against unchanged Pacer, but Defender remains a weakness. Original evidence: `training/results/plain-nano/README.md` and its `evaluation/` and `releases/` directories inside the archive.
- **Pacer / Apprentice:** learned `6→128→128→3` Tanh DAgger exports from the gradual-speed study. Their original selection manifests and candidates are under `training/results/pace-study/training/` inside the archive. They remain the default app matchup.
- **Vector Nano:** fixed dense `6→28→9→3` ReLU weights, 487 parameters and 447 MACs. Original evidence: `training/results/vector-dense-v4/` inside the archive. Construction is maintained in `scripts/vector_dense_v4.py`.
- **Prism Small:** fixed dense `6→42→33→3` ReLU weights, 1,815 parameters and 1,737 MACs. Original evidence: `training/results/robust-dense-v4/` inside the archive. Construction is maintained in `scripts/robust_dense_v4.py`.
- **Tracker:** a 163-parameter dense baseline retained for simple matches and device testing, not a strength claim.

Robust Small's reported gains used a shortcut architecture rejected by the current validator. It is archived, not promoted. Old dense models are retired for roster relevance, not declared ineligible merely because of their age. Historical comparisons used different reference fields; their aggregate scores must not be compared as if they were one leaderboard.

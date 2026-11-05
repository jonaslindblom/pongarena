# Focused PPO experiments under fixed v3 rules

All five runs retain the 6→64→64→3 Tanh actor. They start from original v2 weights, or the first stage's selected terminal-reward checkpoint. Each rally keeps one opponent: 40% v2, 40% unchanged attacker, 20% interceptor. The training-only critic sees opponent identity and remaining time. These extra critic inputs never reach the submitted actor.

The scripts disable loop exits for sampled training actions, while deterministic validation uses the existing v3 loop rule. Validation selects the maximum weaker-target score `(wins + 0.5 × draws) / rallies`, with mean target score breaking ties. This is development selection, not the final acceptance test.

Run from the repository root using the pinned training environment. Use fresh output directories:

```sh
.venv/bin/python training/scripts/train_focus.py --out training/runs/focus-control-s71 --seed 71 --updates 300 --gamma .99 --return-reward .02
.venv/bin/python training/scripts/train_focus.py --out training/runs/focus-terminal-s71 --seed 71 --updates 300 --gamma .99 --return-reward 0
.venv/bin/python training/scripts/train_focus.py --out training/runs/focus-outcome-s71 --seed 71 --updates 300 --gamma 1 --return-reward 0
.venv/bin/python training/scripts/train_focus.py --out training/runs/focus-terminal-stage2-s73 --seed 73 --updates 600 --gamma .99 --return-reward 0 --initial training/runs/focus-terminal-s71/best.pt --logit-scale 1 --learning-rate .00005
.venv/bin/python training/scripts/train_focus.py --out training/runs/focus-outcome-stage2-s73 --seed 73 --updates 600 --gamma 1 --return-reward 0 --initial training/runs/focus-terminal-s71/best.pt --logit-scale 1 --learning-rate .00005
```

Each first stage executed 19,660,800 decisions; each second stage executed 39,321,600. The second stages initialize a fresh critic and optimizer, so these are fine-tuning runs, not exact resumes. Their plotted decision counts start at zero and exclude earlier training. Original v2 pretraining is also excluded.

The long-horizon second stage selected update 275. On 128 fresh shared development seeds it scored 522 wins / 130 losses / 628 draws against v2 and 231 / 81 / 968 against the attacker. Its paired ten-rally win rates were 92.2% and 66.4%, respectively. These results support improvement for the small fully learned actor; it was not the candidate selected for reserved final testing.

The feature-assisted distillation candidate was stronger on that same development comparison, with zero losses against both targets and an 86.7% paired win rate against the attacker. Its export was frozen in commit `683c40f` before the reserved tests. See `candidate/freeze.json`, `development/`, and [DISTILLATION.md](DISTILLATION.md).

These experiments diagnose objective mismatch, but do not isolate every source of the original v3 regression. The focused PPO variants also change opponent distribution, critic inputs and optimization settings compared with the original recipe. Raw per-checkpoint measurements, configurations, selected weights and selections are retained in `runs/`.

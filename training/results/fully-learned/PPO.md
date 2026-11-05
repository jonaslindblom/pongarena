# Plain neural PPO continuation

These runs use only a fully learned `6→64→64→3` Tanh actor. Every affine weight and bias is trainable. No trajectory calculation, paddle-position feedback, handcrafted inference feature, scripted action override or hybrid model checkpoint enters the actor. Its initial weights come from the previously trained plain PPO checkpoint at `training/results/training-improvement/runs/focus-outcome-stage2-s73/best.pt`; each configuration records that file's SHA256.

The unchanged v2 model and scripted attacker are opponents, not action-label teachers. Opponents stay fixed throughout each rally: 40% v2, 40% attacker and 20% interceptor. The separate learned critic sees the six observations, three opponent identity indicators and remaining time. That extra training context is never exported.

All three runs use terminal rewards only (+1 score, −1 concede, 0 draw), gamma 1, 256 environments × 256 steps, four PPO epochs, minibatches of 2,048, clipping .2 and gradient norm .5. Actor and critic have separate Adam optimizers, epsilon 1e−5; critic learning rate is .0003. The first 25 updates train only the critic, preserving the pretrained actor. The actor learning rate subsequently decays to 20% of its initial value. Sampled rollouts disable loop exits; deterministic validation enables the unchanged v3 detector. Both retain the 1,600-tick deadline.

Checkpoint selection prioritizes the lower **ordinary five-rally game win rate** against v2 and attacker, counting draws in the denominator. Ties use the lower rally score `(wins + half draws) / rallies`, then its mean. Development measurements use 64 dispersed seeds sampled with NumPy generator 502 from `[1,300,000,000, 1,400,000,000)`, all five serves and both sides. They are not fresh final evidence.

## Exact commands

Run from the repository root with the pinned Python environment and fresh output directories:

```sh
.venv/bin/python training/scripts/train_plain.py --out training/runs/plain-ppo-conservative-101 --seed 101 --updates 500 --critic-warmup 25 --learning-rate .00001 --gae-lambda .97 --anchor-kl .05 --entropy .001
.venv/bin/python training/scripts/train_plain.py --out training/runs/plain-ppo-credit-103 --seed 103 --updates 500 --critic-warmup 25 --learning-rate .00002 --gae-lambda .995 --anchor-kl .02 --entropy 0
.venv/bin/python training/scripts/train_plain.py --out training/runs/plain-ppo-retain-107 --seed 107 --updates 300 --critic-warmup 25 --learning-rate .000002 --gae-lambda .97 --anchor-kl 1 --entropy 0 --logit-scale 4
```

The two 500-update configurations were deliberately interrupted after sustained development regression. Their last completed logged updates were 250 and 225 respectively, including critic warmup. Both selected update 0: the unchanged initial policy. Retain their original planned 500-update schedule when reproducing the logged prefix, because it affects learning-rate decay. The termination files record the interruption; any unlogged work is excluded from reported counts.

The third run reduces the sampling/deployment gap by initially multiplying the learned output weights and biases by four. This preserves the initial greedy decisions while making sampling more decisive. Those tensors remain trainable; no fixed inference operator is added. A KL penalty against a frozen copy of this learned initial actor limits destructive drift during optimization. That reference is training-only and is absent from the export. These changes are a combined experiment, not isolated causal evidence for each component.

On 32,768 reachable development-diagnostic observations, initial mean action entropy was .1822 and mean probability of the greedy action .9260. Scaling logits by four changed these to .0486 and .9796. These measurements motivated the third run; they are not its final competitive evaluation.

## Records and verification

The run folders retain original configurations, full learning measurements, selected PyTorch/ONNX weights, selection rows, console logs and explicit termination metadata. Optimizer/RNG/environment state and replay buffers are not archived, so this is reproducible training from the initial checkpoint, not exact resume. Decision counts include critic warmup and exclude ancestral PPO training. Logged losses are the final minibatch, not a full-rollout average.

A small smoke run verified that critic-only warmup preserves initial validation outcomes, and that the first actor update changes them. Its first process printed successful completion but returned exit 139 during native shutdown. A fresh rerun reproduced every measurement and completed with exit 0; no checkpoint was selected from a smoke run. The training script also passes Python compilation. Final-model verification and shared development comparisons are recorded separately by the release workflow.

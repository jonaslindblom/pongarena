"""Build a static, shareable training report from recorded measurements (no retraining)."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/rally-lab-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, default=Path('training/results/seed7'))
    p.add_argument('--out', type=Path, default=Path('public/training'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (args.results / 'metrics.jsonl').read_text().splitlines()]
    tests = {n: json.loads((args.results / f'{n}-test.json').read_text()) for n in ['apprentice', 'selfplay']}
    config = json.loads((args.results / 'config.json').read_text())
    picks = {s: json.loads((args.results / f'{s}-selection.json').read_text()) for s in ['imitation', 'ppo']}
    x = np.array([r['env_steps'] for r in rows]) / 1e6
    split = max(r['env_steps'] for r in rows if r['stage'] == 'imitation') / 1e6
    colors = ['#3c7b4d', '#78a64b', '#244f78', '#c88a39', '#b95748']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelcolor': '#536153', 'text.color': '#203c30',
                         'axes.edgecolor': '#ccd4c6', 'figure.facecolor': '#f5f6f0', 'axes.facecolor': '#fdfef9'})
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8), constrained_layout=True)
    fig.suptitle('Rally Lab  /  Pong v1 learning record', fontsize=21, fontweight='bold', ha='left', x=.045)
    ax = axes[0, 0]
    ax.plot(x, [r['validation_score'] for r in rows], '-o', color='#275a3c', ms=4)
    for stage, name in [('imitation', 'Apprentice'), ('ppo', 'Selfplay')]:
        row = picks[stage]
        ax.scatter([row['env_steps'] / 1e6], [row['validation_score']], marker='*', s=170, color='#c67d2a', zorder=4)
        ax.annotate(name, (row['env_steps'] / 1e6, row['validation_score']), xytext=(8, -20), textcoords='offset points', fontsize=9)
    ax.set(title='Validation score · selected checkpoints starred', xlabel='Cumulative environment decisions (millions)',
           ylabel='Mean across five opponents: wins + ½ draws', ylim=(0, 1))
    ax = axes[0, 1]
    for color, opponent in zip(colors, tests['selfplay']['opponents']):
        ax.plot(x, [r['opponents'][opponent]['point_rate'] for r in rows], color=color, label=opponent, linewidth=1.8)
    ax.set(title='Validation rally wins by opponent', xlabel='Cumulative environment decisions (millions)', ylabel='Rallies won / all rallies', ylim=(0, 1))
    ax.legend(fontsize=8, ncol=3, loc='lower right')
    for ax in axes[0]:
        ax.axvline(split, color='#9b9e91', linestyle='--', linewidth=1)
        ax.text(split + .07, .04, 'PPO begins', fontsize=8, color='#7a7f70')
    bc = [r for r in rows if r['stage'] == 'imitation' and r['iteration'] > 0]
    axes[1, 0].plot([r['iteration'] for r in bc], [r['cross_entropy'] for r in bc], '-o', color='#a57334', ms=4)
    axes[1, 0].set(title='Imitation warm start · DAgger', xlabel='Data collection / fitting round', ylabel='Cross-entropy, final minibatch (lower is better)')
    ppo = [r for r in rows if r['stage'] == 'ppo']
    axes[1, 1].plot([r['iteration'] for r in ppo], [r['entropy'] for r in ppo], '-o', color='#396780', ms=4)
    axes[1, 1].set(title='PPO action entropy', xlabel='PPO update', ylabel='Entropy in nats, final minibatch')
    for ax in axes.flat:
        ax.grid(axis='y', color='#e1e5db', linewidth=.6)
        ax.set_axisbelow(True)
    fig.savefig(args.out / 'learning-curves.png', dpi=150)
    fig.savefig(args.out / 'learning-curves.svg')
    plt.close(fig)

    body = ''
    for name, label in [('apprentice', 'Apprentice · imitation'), ('selfplay', 'Selfplay · PPO')]:
        body += f'<h3>{label}</h3><div class="table-wrap"><table><thead><tr><th>Opponent</th><th>Rally wins</th><th>Losses</th><th>Draws</th><th>Matches W / L / D</th><th>Median returns¹</th></tr></thead><tbody>'
        for opponent, r in tests[name]['opponents'].items():
            body += f'<tr><td>{opponent.title()}</td><td><strong>{r["wins"] / 10:.1f}%</strong> <small>({r["wins"]})</small></td><td>{r["losses"]}</td><td>{r["draws"]}</td><td>{r["match_wins"]} / {r["match_losses"]} / {r["match_draws"]}</td><td>{r["median_decisive_returns"]:g}</td></tr>'
        body += '</tbody></table></div>'
    elapsed = rows[-1]['seconds']
    total_steps = rows[-1]['env_steps']
    imitation_steps = max(r['env_steps'] for r in rows if r['stage'] == 'imitation')
    test_start = tests['selfplay']['seed_start']
    test_count = tests['selfplay']['seed_count']
    challenger_win_rate = tests['selfplay']['opponents']['challenger']['point_rate']
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rally Lab · Training record</title><link rel="icon" href="/mark.svg">
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f6f0;color:#203c30;font:15px/1.65 system-ui,sans-serif}}main{{max-width:1140px;padding:42px 24px 70px;margin:auto}}a{{color:#376e43}}nav{{display:flex;justify-content:space-between;margin-bottom:52px}}h1{{font-size:clamp(34px,6vw,64px);letter-spacing:-.06em;line-height:1.08;margin:10px 0 22px}}h2{{font-size:26px;letter-spacing:-.03em;margin-top:0}}h3{{margin-top:30px}}p{{max-width:850px;color:#5a6b5b}}.eyebrow{{font:11px monospace;letter-spacing:.15em;color:#75935c}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:32px 0}}.stat,section{{background:#fdfef9;border:1px solid #dce3d5;border-radius:12px;padding:22px}}.stat b{{font-size:30px;display:block;letter-spacing:-.04em}}.stat:last-child b{{font-size:clamp(13px,2vw,22px);line-height:1.5;white-space:nowrap}}.stat small{{font-size:11px;color:#70816c}}section{{margin-top:20px}}img{{width:100%;height:auto;border-radius:8px}}.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;text-align:left}}th{{font-weight:500;color:#70816c}}td,th{{padding:12px 10px;border-bottom:1px solid #e2e7dc;white-space:nowrap}}td small{{color:#75806f}}.downloads{{display:flex;gap:12px;flex-wrap:wrap}}.downloads a{{border:1px solid #bfd0b7;padding:9px 15px;border-radius:7px;text-decoration:none;font-size:13px}}code{{background:#edf1e7;padding:3px 6px;border-radius:4px}}footer{{color:#7b8772;font-size:12px;margin-top:30px}}@media(max-width:600px){{.stats{{grid-template-columns:1fr 1fr}}section{{padding:16px}}nav{{margin-bottom:30px}}}}
</style><main><nav><a href="/">← Back to the arena</a><span>Rally Lab / Experiment 001</span></nav>
<div class="eyebrow">FROZEN RULES · PONG V1 · PYTORCH</div><h1>From imitation<br>to self-play.</h1>
<p>Two compact policies, one unchanged game. Apprentice learns a predictive attacker's decisions. Selfplay refines that starting policy through clipped PPO against built-in paddles, scripted references and historical versions of itself.</p>
<div class="stats"><div class="stat"><b>{total_steps / 1e6:.2f}M</b><small>Training environment decisions</small></div><div class="stat"><b>4,803</b><small>Parameters in each actor</small></div><div class="stat"><b>{challenger_win_rate:.1%}</b><small>Selfplay rally wins vs Challenger</small></div><div class="stat"><b>6 → 64 → 64 → 3</b><small>Tanh · 4,672 MACs per decision</small></div></div>
<div class="downloads"><a href="/models/selfplay.onnx" download>Download Selfplay · ONNX</a><a href="/models/apprentice.onnx" download>Download Apprentice · ONNX</a><a href="pong-v1.zip" download>Standalone v1 training contract</a></div>
<section><h2>The learning curves</h2><p>Raw measurements from one training seed (7), with no smoothing. Validation uses 24 fixed seeds, all five serves and both sides: 240 rallies per opponent at every checkpoint. The vertical line marks the switch to PPO; PPO restarts from the best imitation checkpoint. Stars mark the exported policies.</p><a href="learning-curves.svg"><img src="learning-curves.png" alt="Four plots showing validation score, per-opponent win rates, imitation loss and PPO entropy over training"></a>
<p>The strongest PPO checkpoint was update {picks['ppo']['iteration']} at {picks['ppo']['env_steps']:,} cumulative decisions. Later training did not reliably improve it. Checkpoint selection uses the mean of (rally wins + half rally draws) across five opponents. This is an evaluation statistic; the arena still awards one point per miss and zero for a timed-out rally.</p></section>
<section><h2>Unseen test serves</h2><p>These results use the exported ONNX files, deterministic argmax, seeds {test_start:,}–{test_start + test_count - 1:,}, every serve and both sides. Each row is {test_count * 10:,} rallies / {test_count * 2:,} five-rally matches. Test seeds were excluded from training and checkpoint selection.</p>{body}<p>¹ Median returns among rallies that ended in a point. Both models still lose heavily to the predictive attacker. This is a useful trained baseline, not evidence of optimal play or state-of-the-art benchmark performance.</p></section>
<section><h2>How the weights were learned</h2><p>{config["dagger_rounds"]} DAgger rounds collect {imitation_steps:,} decisions and fit a small actor to a scripted attacker's labels, including states visited by the learner. The best validation checkpoint becomes Apprentice. PPO then runs {config["updates"]} updates × {config["envs"]} environments × {config["horizon"]} steps, adding {total_steps - imitation_steps:,} decisions. Its opponent pool includes Tracker, Anticipator, Challenger, two scripts and up to five historical actors.</p>
<p>PPO uses generalized advantage estimation (γ=.99, λ=.95), a .2 clipped objective, clipped value loss, four optimization epochs, normalized advantages, gradient clipping and a decaying learning rate. Rewards are +1 for scoring, −1 for conceding and +.02 per return. A small decaying imitation loss helps retain the warm start. Training samples actions; the arena takes the first maximum. The separate critic and all training machinery are excluded from the ONNX files.</p>
<p>The training run took about {elapsed:.0f} seconds locally, excluding environment setup and final evaluations. This is one seed on one machine, with no multi-seed confidence interval. Scripted opponents are stress tests, not optimal-play proofs. CPU float32 inference differences across runtimes can matter near action ties.</p>
<p>Methods: <a href="https://arxiv.org/abs/1707.06347">Proximal Policy Optimization</a>, <a href="https://arxiv.org/abs/1011.0686">DAgger</a>, and <a href="https://docs.cleanrl.dev/rl-algorithms/ppo/">CleanRL's PPO implementation notes</a>. The implementation here is purpose-built for the frozen v1 reference.</p></section>
<section><h2>Reproduce and inspect</h2><p>Start with <code>training/README.md</code> in the repository. The scripts cover simulator parity, data collection, training, ONNX export, held-out evaluation, plotting and packaging. Raw logs, configuration, selected weights and test results are retained in <code>training/results/seed7/</code>.</p><p>Verified: 883 exact scalar transitions and 11 boundary cases; 32,000 randomized batched transitions; 8,192 Torch/ONNX observations per exported actor with identical actions; eight complete ONNX matches reproduce in the live TypeScript engine (7,210 frames). The current v1 engine's source hash is unchanged.</p></section>
<footer>Rally Lab · Local training record · v1 baseline commit 59ca903</footer></main></html>'''
    (args.out / 'report.html').write_text(html)
    print(args.out / 'report.html')


if __name__ == '__main__':
    main()

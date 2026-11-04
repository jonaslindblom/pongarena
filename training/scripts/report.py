"""Render the v2 balance study, three training runs, and final exported-model evaluation."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/rally-lab-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def curve(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, default=Path('training/results/v2'))
    p.add_argument('--out', type=Path, default=Path('public/training'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    rows = curve(args.results / 'metrics.jsonl')
    manifest = read(args.results / 'selection.json')
    tests = {n: read(args.results / f'{n}-test.json') for n in ['apprentice', 'selfplay']}
    config = read(args.results / 'config.json')
    picks = {s: read(args.results / f'{s}-selection.json') for s in ['imitation', 'ppo']}
    matchups = read(args.results / 'matchups.json')['opponents']
    balance = read(root / 'training/results/balance-v2/holdout.json')
    export = read(args.results / 'export-check.json')
    replays = read(args.results / 'onnx-replays.json')
    x = np.array([r['env_steps'] for r in rows]) / 1e6
    split = max(r['env_steps'] for r in rows if r['stage'] == 'imitation') / 1e6
    colors = ['#3c7b4d', '#78a64b', '#244f78', '#c88a39', '#b95748']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelcolor': '#536153', 'text.color': '#203c30',
                         'axes.edgecolor': '#ccd4c6', 'figure.facecolor': '#f5f6f0', 'axes.facecolor': '#fdfef9'})
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8), constrained_layout=True)
    fig.suptitle('Rally Lab  /  Pong v2 learning record', fontsize=21, fontweight='bold', ha='left', x=.045)
    ax = axes[0, 0]
    for index, run in enumerate(manifest['runs']):
        data = curve(args.results / f'seed{run["seed"]}' / 'metrics.jsonl')
        selected = run['seed'] == manifest['selected_seed']
        ax.plot([r['env_steps'] / 1e6 for r in data], [r['validation_score'] for r in data],
                color='#275a3c' if selected else ['#91a7bc', '#aab8a2', '#c2aa81'][index % 3], linewidth=2 if selected else 1,
                linestyle='-' if selected else '--', label=f'Seed {run["seed"]}' + (' · selected' if selected else ''))
    for stage in ['imitation', 'ppo']:
        row = picks[stage]
        ax.scatter([row['env_steps'] / 1e6], [row['validation_score']], marker='*', s=140, color='#c67d2a', zorder=4)
    ax.set(title='Validation · three independent training seeds', xlabel='Environment decisions per run (millions)',
           ylabel='Mean across five opponents: wins + ½ draws', ylim=(0, 1))
    ax.legend(fontsize=8, loc='lower right')
    ax = axes[0, 1]
    for color, opponent in zip(colors, tests['selfplay']['opponents']):
        ax.plot(x, [r['opponents'][opponent]['point_rate'] for r in rows], color=color, label=opponent, linewidth=1.8)
    ax.set(title=f'Selected seed {manifest["selected_seed"]} · rally wins', xlabel='Environment decisions (millions)', ylabel='Rallies won / all rallies', ylim=(0, 1))
    ax.legend(fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(.5, -.18))
    for ax in axes[0]:
        ax.axvline(split, color='#9b9e91', linestyle='--', linewidth=1)
        ax.text(split + .07, .04, 'PPO begins', fontsize=8, color='#7a7f70')
    bc = [r for r in rows if r['stage'] == 'imitation' and r['iteration'] > 0]
    axes[1, 0].plot([r['iteration'] for r in bc], [r['cross_entropy'] for r in bc], '-o', color='#a57334', ms=4)
    axes[1, 0].set(title='Imitation warm start · DAgger', xlabel='Data collection / fitting round', ylabel='Cross-entropy, final minibatch')
    ppo = [r for r in rows if r['stage'] == 'ppo']
    axes[1, 1].plot([r['iteration'] for r in ppo], [r['entropy'] for r in ppo], '-o', color='#396780', ms=4)
    axes[1, 1].set(title='PPO action entropy', xlabel='PPO update', ylabel='Entropy in nats, final minibatch')
    for ax in axes.flat:
        ax.grid(axis='y', color='#e1e5db', linewidth=.6)
        ax.set_axisbelow(True)
    fig.savefig(args.out / 'learning-curves.png', dpi=150)
    fig.savefig(args.out / 'learning-curves.svg')
    plt.close(fig)
    svg = args.out / 'learning-curves.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')

    def result_table(entries):
        text = '<div class="table-wrap"><table><thead><tr><th>Opponent</th><th>Rally wins</th><th>Losses</th><th>Draws</th><th>Matches W / L / D</th><th>Median returns¹</th></tr></thead><tbody>'
        for opponent, r in entries.items():
            median = '—' if r['median_decisive_returns'] is None else f'{r["median_decisive_returns"]:g}'
            text += f'<tr><td>{opponent.title()}</td><td><strong>{r["point_rate"]:.1%}</strong> <small>({r["wins"]})</small></td><td>{r["losses"]}</td><td>{r["draws"]}</td><td>{r["match_wins"]} / {r["match_losses"]} / {r["match_draws"]}</td><td>{median}</td></tr>'
        return text + '</tbody></table></div>'
    body = ''.join(f'<h3>{n.title()} · {"imitation" if n == "apprentice" else "PPO"}</h3>' + result_table(tests[n]['opponents']) for n in tests)
    balance_body = '<div class="table-wrap"><table><thead><tr><th>Edge speed cap</th><th>Median returns¹</th><th>First receiver wins²</th><th>Draws</th><th>Early finishes³</th></tr></thead><tbody>'
    for r in balance['rows']:
        if r['pair'] == 'AA':
            balance_body += f'<tr><td>{r["cap"]}' + (' · chosen v2' if r['cap'] == 13 else ' · v1' if r['cap'] == 15 else '') + f'</td><td>{r["median_decisive_returns"]:g}</td><td>{r["first_receiver_win_rate"]:.1%}</td><td>{r["draws"]/r["n"]:.1%}</td><td>{r["early_rate"]:.1%}</td></tr>'
    balance_body += '</tbody></table></div>'
    run_body = ''.join(f'<tr><td>{r["seed"]}</td><td>{r["selected_update"]}</td><td>{r["validation_score"]:.3f}</td><td>{"Selected" if r["seed"] == manifest["selected_seed"] else "Retained for comparison"}</td></tr>' for r in manifest['runs'])
    total_steps = sum(r['total_steps'] for r in manifest['runs'])
    per_run = rows[-1]['env_steps']
    imitation_steps = int(split * 1e6)
    test_start, test_count = tests['selfplay']['seed_start'], tests['selfplay']['seed_count']
    actor_test = tests['selfplay']['opponents']['attacker']
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rally Lab · Pong v2 balance & training</title><link rel="icon" href="/mark.svg">
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f6f0;color:#203c30;font:15px/1.65 system-ui,sans-serif}}main{{max-width:1140px;padding:42px 24px 70px;margin:auto}}a{{color:#376e43}}nav{{display:flex;justify-content:space-between;gap:20px;margin-bottom:52px}}h1{{font-size:clamp(34px,6vw,64px);letter-spacing:-.06em;line-height:1.08;margin:10px 0 22px}}h2{{font-size:26px;letter-spacing:-.03em;margin-top:0}}h3{{margin-top:30px}}p{{max-width:850px;color:#5a6b5b}}.eyebrow{{font:11px monospace;letter-spacing:.15em;color:#75935c}}.stats{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:32px 0}}.stat,section{{background:#fdfef9;border:1px solid #dce3d5;border-radius:12px;padding:22px}}.stat b{{font-size:30px;display:block;letter-spacing:-.04em}}.stat small{{font-size:11px;color:#70816c}}section{{margin-top:20px}}img{{width:100%;height:auto;border-radius:8px}}.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;text-align:left}}th{{font-weight:500;color:#70816c}}td,th{{padding:12px 10px;border-bottom:1px solid #e2e7dc;white-space:nowrap}}td small{{color:#75806f}}.downloads{{display:flex;gap:12px;flex-wrap:wrap}}.downloads a{{border:1px solid #bfd0b7;padding:9px 15px;border-radius:7px;text-decoration:none;font-size:13px}}code{{background:#edf1e7;padding:3px 6px;border-radius:4px;overflow-wrap:anywhere}}footer{{color:#7b8772;font-size:12px;margin-top:30px}}@media(max-width:600px){{.stats{{grid-template-columns:1fr 1fr}}.stat{{padding:16px}}.stat b{{font-size:clamp(22px,6vw,30px)}}section{{padding:16px}}nav{{margin-bottom:30px}}}}
</style><main><nav><a href="/">← Back to the arena</a><a href="v1/report.html">Archived v1 experiment ↗</a></nav>
<div class="eyebrow">PONG V2 · BALANCE STUDY + PYTORCH</div><h1>Room to rally.</h1>
<p>A smaller edge-speed bonus gives players more time to recover. We compared six caps on identical serves, confirmed the choice on separate seeds, then trained three independent runs against the new rules.</p>
<div class="stats"><div class="stat"><b>15 → 13</b><small>Maximum horizontal return speed</small></div><div class="stat"><b>{matchups['selfplay']['median_decisive_returns']:g} returns</b><small>Median scored Selfplay mirror rally</small></div><div class="stat"><b>{total_steps / 1e6:.2f}M</b><small>Training decisions across three runs</small></div><div class="stat"><b>4,803</b><small>Parameters in each exported actor</small></div></div>
<div class="downloads"><a href="/models/selfplay-v2.onnx" download>Selfplay v2 · ONNX</a><a href="/models/apprentice-v2.onnx" download>Apprentice v2 · ONNX</a><a href="pong-v2.zip" download>Standalone v2 contract</a><a href="pong-v1.zip" download>Frozen v1 contract</a></div>
<section><h2>Finding the balance</h2><p>Only the horizontal return formula changes: <code>5 + round(8 × |offset| / 15)</code>, replacing the v1 multiplier 10. Paddle movement, vertical aiming, serve seeds, tick rate, timeout and all model inputs/outputs stay the same. The fastest crossing becomes 26 ticks (~0.87s), previously 22 (~0.73s).</p><p>Two identical predictive attackers played the same 2,000 rallies per candidate in the confirmation set (200 seeds, five serves, both sides). Their shot planner uses each candidate's actual speed formula.</p>{balance_body}
<p>¹ Scored rallies only. ² Among scored rallies. ³ A point before the second return, as a share of all rallies. Caps 10 and 11 produced 100% attacker-mirror draws in the initial scan. Cap 13 keeps scoring while removing most of the first-receiver advantage. Against a centering interceptor, the attacker still wins 62.4%, with a median 7 returns in scored rallies.</p><p>This is the best compromise among these tested caps. A delayed-controller sensitivity check also improves, but human comfort still needs actual playtesting. No human-only physics or hidden difficulty phase is added. The reproducible comparison and raw results are in <code>training/results/balance-v2/</code>.</p></section>
<section><h2>The new learning curves</h2><p>Seeds 7, 17, 29 each ran {per_run:,} environment decisions. Raw curves are unsmoothed. Each checkpoint is evaluated on the same 24 validation seeds, all five serves and both sides: 240 rallies per opponent. Selection uses the mean of (rally wins + half draws) over five reference opponents. This is a training diagnostic; the arena's match scoring is unchanged.</p><a href="learning-curves.svg"><img src="learning-curves.png" alt="Validation curves for three training seeds, selected-model win rates by opponent, imitation loss and PPO entropy"></a>
<div class="table-wrap"><table><thead><tr><th>Training seed</th><th>Best PPO update</th><th>Validation score</th><th>Release</th></tr></thead><tbody>{run_body}</tbody></table></div>
<p>Selfplay uses seed {manifest['selected_seed']}, update {picks['ppo']['iteration']}. Apprentice is that run's own best imitation checkpoint. Test results were not used to select either the rule cap or the released training run. Three runs expose training variation; the chosen run's test score is not a multi-seed average or a confidence interval.</p></section>
<section><h2>Unseen test serves</h2><p>The exported ONNX files use deterministic argmax on seeds {test_start:,}–{test_start + test_count - 1:,}, both sides, all five serves. Each row contains {test_count * 10:,} rallies / {test_count * 2:,} five-rally matches.</p>{body}
<p>¹ Median returns in rallies that ended in a point. The attacker remains stronger: Selfplay wins {actor_test['point_rate']:.1%}, loses {actor_test['losses']/actor_test['rallies']:.1%} and draws {actor_test['draws']/actor_test['rallies']:.1%}. Surviving until the timeout is not a win. These learned policies are useful baselines with room to improve.</p>
<h3>Two learned players</h3><p>Selfplay is the first player in both rows. These are new v2 weights, not the old policy transferred to different physics.</p>{result_table(matchups)}<p>Selfplay mirror matches now have median {matchups['selfplay']['median_decisive_returns']:g} decisive returns and {matchups['selfplay']['draws']/matchups['selfplay']['rallies']:.1%} draws. The opening matchup, Selfplay vs Apprentice, has median {matchups['apprentice']['median_decisive_returns']:g} decisive returns and {matchups['apprentice']['draws']/matchups['apprentice']['rallies']:.1%} draws. Long scored rallies and timeout rates are both reported.</p></section>
<section><h2>How the weights were learned</h2><p>Each run collects {imitation_steps:,} DAgger decisions in {config['dagger_rounds']} rounds. A predictive attacker labels states visited by the learner. PPO then adds {per_run-imitation_steps:,} decisions ({config['updates']} updates × {config['envs']} environments × {config['horizon']} steps), facing the original three baselines, two scripts and up to five historical copies of itself.</p>
<p>The actor is <code>6 → 64 → 64 → 3</code> with Tanh hidden layers, 4,803 parameters and 4,672 MACs per decision. PPO uses GAE (γ=.99, λ=.95), clipping .2, four epochs, gradient clipping, decaying learning rate, entropy regularization and a small decaying imitation anchor. Rewards are +1 for scoring, −1 for conceding and +.02 per return. The separate critic, optimizer and opponent pool are excluded from ONNX.</p>
<p>The three local runs completed in about 120 seconds while running concurrently. Timing is machine-specific and excludes installation/final evaluation. Methods: <a href="https://arxiv.org/abs/1707.06347">PPO</a>, <a href="https://arxiv.org/abs/1011.0686">DAgger</a>, <a href="https://docs.cleanrl.dev/rl-algorithms/ppo/">CleanRL implementation notes</a>.</p></section>
<section><h2>Reproduce and verify</h2><p>Start with <code>training/README.md</code>. Current rules and fixtures live in <code>training/v2/</code>; original v1 stays frozen. Logs, all three selected checkpoint pairs, configuration, selection records and test results are retained in <code>training/results/v2/</code>.</p><p>Verified: 980 exact scalar match transitions and 11 boundary cases; 32,000 randomized batched transitions; {next(iter(export.values()))['observations']:,} Torch/ONNX observations per actor with identical actions; {len(replays)} complete native-ONNX matches reproduce in the live TypeScript engine ({sum(len(m['steps']) for m in replays):,} frames). V2 files have distinct filenames and rule metadata, preserving the v1 weights and report.</p></section>
<footer>Rally Lab · Pong v2 · Balance checkpoint a3e23de</footer></main></html>'''
    (args.out / 'report.html').write_text(html)
    print(args.out / 'report.html')


if __name__ == '__main__':
    main()

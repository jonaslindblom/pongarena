"""Render Pong v3 training, loop-aware evaluation and a same-rules v2 comparison."""
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
    p.add_argument('--results', type=Path, default=Path('training/results/v3'))
    p.add_argument('--out', type=Path, default=Path('public/training'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = curve(args.results / 'metrics.jsonl')
    manifest = read(args.results / 'selection.json')
    tests = {n: read(args.results / f'{n}-test.json') for n in ['apprentice', 'selfplay']}
    config = read(args.results / 'config.json')
    picks = {s: read(args.results / f'{s}-selection.json') for s in ['imitation', 'ppo']}
    matchups = read(args.results / 'matchups.json')['opponents']
    previous = read(args.results / 'v2-comparison.json')['models']
    export = read(args.results / 'export-check.json')
    replays = read(args.results / 'onnx-replays.json')
    x = np.array([r['env_steps'] for r in rows]) / 1e6
    split = max(r['env_steps'] for r in rows if r['stage'] == 'imitation') / 1e6
    colors = ['#3c7b4d', '#78a64b', '#244f78', '#c88a39', '#b95748']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelcolor': '#536153', 'text.color': '#203c30',
                         'axes.edgecolor': '#ccd4c6', 'figure.facecolor': '#f5f6f0', 'axes.facecolor': '#fdfef9'})
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8), constrained_layout=True)
    fig.suptitle('Rally Lab  /  Pong v3 learning record', fontsize=21, fontweight='bold', ha='left', x=.045)
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
        text = '<div class="table-wrap" tabindex="0"><table><thead><tr><th>Opponent</th><th>Rally wins</th><th>Losses</th><th>Loop draws</th><th>Timeout draws</th><th>Matches W / L / D</th><th>Median returns¹</th><th>Mean seconds²</th></tr></thead><tbody>'
        for opponent, r in entries.items():
            median = '—' if r['median_decisive_returns'] is None else f'{r["median_decisive_returns"]:g}'
            assert r['loop_draws'] + r['timeout_draws'] == r['draws']
            text += f'<tr><td>{opponent.title()}</td><td><strong>{r["point_rate"]:.1%}</strong> <small>({r["wins"]})</small></td><td>{r["losses"]}</td><td>{r["loop_draws"]}</td><td>{r["timeout_draws"]}</td><td>{r["match_wins"]} / {r["match_losses"]} / {r["match_draws"]}</td><td>{median}</td><td>{r["mean_ticks"]/30:.2f}</td></tr>'
        return text + '</tbody></table></div>'

    body = ''.join(f'<h3>{n.title()} · {"imitation" if n == "apprentice" else "PPO"}</h3>' + result_table(tests[n]['opponents']) for n in tests)
    comparison_body = ''
    for name in ['selfplay', 'apprentice']:
        comparison_body += f'<h3>{name.title()} · old and new weights on Pong v3</h3><div class="table-wrap" tabindex="0"><table><thead><tr><th>Opponent</th><th>v2 weights W / L / D</th><th>v3 weights W / L / D</th><th>Score-rate change³</th></tr></thead><tbody>'
        for opponent, current in tests[name]['opponents'].items():
            old = previous[f'{name}-v2']['opponents'][opponent]
            delta = 100 * (current['score_rate'] - old['score_rate'])
            comparison_body += f'<tr><td>{opponent.title()}</td><td>{old["wins"]} / {old["losses"]} / {old["draws"]}</td><td>{current["wins"]} / {current["losses"]} / {current["draws"]}</td><td>{delta:+.2f} pp</td></tr>'
        comparison_body += '</tbody></table></div>'
    run_body = ''.join(f'<tr><td>{r["seed"]}</td><td>{r["selected_update"]}</td><td>{r["validation_score"]:.3f}</td><td>{"Selected" if r["seed"] == manifest["selected_seed"] else "Retained for comparison"}</td></tr>' for r in manifest['runs'])
    total_steps = sum(r['total_steps'] for r in manifest['runs'])
    per_run = rows[-1]['env_steps']
    imitation_steps = int(split * 1e6)
    test_start, test_count = tests['selfplay']['seed_start'], tests['selfplay']['seed_count']
    actor_test = tests['selfplay']['opponents']['attacker']
    old_actor = previous['selfplay-v2']['opponents']
    delta_score = np.mean([r['score_rate'] - old_actor[name]['score_rate'] for name, r in tests['selfplay']['opponents'].items()])
    comparison_summary = f'The new Selfplay actor changes mean score rate across the five reference opponents by {100*delta_score:+.2f} percentage points on these test serves.'
    direct = matchups['selfplay-v2']
    defense = tests['selfplay']['opponents']['interceptor']
    comparison_details = (f'Against the attacker, rally wins change from {old_actor["attacker"]["point_rate"]:.1%} to {actor_test["point_rate"]:.1%}; '
                          f'against the interceptor, they change from {old_actor["interceptor"]["point_rate"]:.1%} to {defense["point_rate"]:.1%}. '
                          f'In their direct matchup, new Selfplay wins {direct["wins"]}, loses {direct["losses"]} and draws {direct["draws"]} against old Selfplay. '
                          'Retraining produces a different policy, not a guaranteed improvement across opponents.')
    attacker_summary = ('The scripted attacker still wins more rallies than Selfplay.' if actor_test['losses'] > actor_test['wins']
                        else 'Selfplay wins more rallies than the scripted attacker in this evaluation.' if actor_test['wins'] > actor_test['losses']
                        else 'Selfplay and the scripted attacker win the same number of rallies in this evaluation.')
    replay_frames = sum(len(m['steps']) for m in replays)
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rally Lab · Pong v3 training</title><link rel="icon" href="/mark.svg">
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f6f0;color:#203c30;font:15px/1.65 system-ui,sans-serif}}main{{max-width:1140px;padding:42px 24px 70px;margin:auto}}a{{color:#376e43}}nav{{display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap;margin-bottom:52px}}h1{{font-size:clamp(34px,6vw,64px);letter-spacing:-.06em;line-height:1.08;margin:10px 0 22px}}h2{{font-size:26px;letter-spacing:-.03em;margin-top:0}}h3{{margin-top:30px}}p{{max-width:900px;color:#5a6b5b}}.eyebrow{{font:11px monospace;letter-spacing:.15em;color:#75935c}}.stats{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:32px 0}}.stat,section{{background:#fdfef9;border:1px solid #dce3d5;border-radius:12px;padding:22px;min-width:0}}.stat b{{font-size:30px;display:block;letter-spacing:-.04em}}.stat small{{font-size:11px;color:#70816c}}section{{margin-top:20px}}img{{width:100%;height:auto;border-radius:8px}}.table-wrap{{max-width:100%;overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;text-align:left}}th{{font-weight:500;color:#70816c}}td,th{{padding:12px 10px;border-bottom:1px solid #e2e7dc;white-space:nowrap}}td small{{color:#75806f}}.downloads{{display:flex;gap:12px;flex-wrap:wrap}}.downloads a{{border:1px solid #bfd0b7;padding:9px 15px;border-radius:7px;text-decoration:none;font-size:13px}}code{{background:#edf1e7;padding:3px 6px;border-radius:4px;overflow-wrap:anywhere}}.note{{font-size:13px}}footer{{color:#7b8772;font-size:12px;margin-top:30px}}@media(max-width:600px){{.stats{{grid-template-columns:1fr 1fr}}.stat{{padding:16px}}.stat b{{font-size:clamp(22px,6vw,30px)}}section{{padding:16px}}nav{{margin-bottom:30px}}}}
</style><main><nav><a href="/">← Back to the arena</a><a href="rule-study.html">Why these rules? ↗</a><a href="v2/report.html">Archived v2 training ↗</a></nav>
<div class="eyebrow">PONG V3 · LONGER RALLIES + EXACT LOOP DETECTION</div><h1>Time to finish.</h1>
<p>Rallies have up to 1,600 ticks to find a winner. Deterministic model matches can finish earlier when an exact repeated state proves a draw. The ball, paddles, serves and six-input model interface keep the same physics as Pong v2.</p>
<div class="stats"><div class="stat"><b>1,600</b><small>Maximum action ticks per rally</small></div><div class="stat"><b>{matchups['selfplay']['loop_draws'] / matchups['selfplay']['rallies']:.1%}</b><small>Selfplay mirror rallies ending in proven loops</small></div><div class="stat"><b>{total_steps / 1e6:.2f}M</b><small>Training decisions across three runs</small></div><div class="stat"><b>4,803</b><small>Parameters in each exported actor</small></div></div>
<div class="downloads"><a href="/models/selfplay-v3.onnx" download>Selfplay v3 · ONNX</a><a href="/models/apprentice-v3.onnx" download>Apprentice v3 · ONNX</a><a href="pong-v3.zip" download>Standalone v3 contract</a><a href="pong-v2.zip" download>Frozen v2 contract</a></div>
<section><h2>What changed in the rules</h2><p>The 800-tick limit becomes 1,600: 53.3 seconds at the nominal 30 decisions per second. A miss still awards one point, and five rallies determine the match. A point on the final tick takes precedence. A draw awards neither player a point.</p>
<p>After each successful return, model matches compare the exact ball position and velocity plus both paddle positions with earlier return states in that rally. If the state repeats, frozen deterministic policies must repeat their future decisions, so the rally ends as a <strong>loop draw</strong>. Otherwise, an unresolved rally reaching the limit is a <strong>timeout draw</strong>. Each new rally and restarted match has a fresh loop history.</p>
<p>Loop exits are disabled when a human plays. They are also disabled during stochastic training: the same physical state need not produce the same next action. The standalone reference makes this distinction explicit. No new policy inputs, acceleration, speed ramp or hidden difficulty state were added.</p>
<p class="note">The preceding <a href="rule-study.html">matched-serve study</a> found that longer deadlines recover real points in some matchups, while many other draws are exact loops. Pong v3 preserves the v2 contact formula, paddle speed 2 and horizontal return range 5–13.</p></section>
<section><h2>Three independent learning runs</h2><p>Seeds 7, 17, 29 each ran {per_run:,} environment decisions. Curves are raw and unsmoothed. Every checkpoint is evaluated on the same fresh 24 validation seeds, all five serves and both sides: 240 rallies per opponent. Selection uses mean (rally wins + half draws) over five reference opponents; it does not use the final test set.</p><a href="learning-curves.svg"><img src="learning-curves.png" alt="Pong v3 validation curves for three training seeds, selected-run opponent win rates, imitation loss and PPO entropy"></a>
<div class="table-wrap"><table><thead><tr><th>Training seed</th><th>Best PPO update</th><th>Validation score</th><th>Release</th></tr></thead><tbody>{run_body}</tbody></table></div>
<p>Selfplay uses seed {manifest['selected_seed']}, update {picks['ppo']['iteration']}. Apprentice is that run’s own best imitation warm start. Three runs expose training variation; the selected actor’s test score is not a multi-seed average or an optimal-play claim.</p></section>
<section><h2>Unseen test serves</h2><p>The actual exported ONNX actors use deterministic argmax on seeds {test_start:,}–{test_start + test_count - 1:,}, both sides and all five serves. Every row contains {test_count * 10:,} rallies / {test_count * 2:,} five-rally matches under Pong v3. Loop and timeout counts add up to total draws.</p>{body}
<p class="note">¹ Median successful returns in scored rallies only. ² Simulation action ticks / 30, including early loop exits but excluding inference, display pauses and serves. Long rallies and draws are reported separately.</p>
<p>{attacker_summary} Selfplay wins {actor_test['point_rate']:.1%}, loses {actor_test['losses']/actor_test['rallies']:.1%} and draws {actor_test['draws']/actor_test['rallies']:.1%} against it. A draw is not a win.</p></section>
<section><h2>Did retraining improve the weights?</h2><p>The original v2 exports were reevaluated under the same v3 rules, on the same fresh test serves as the new actors. This isolates the weights comparison from the longer deadline and early loop exits. {comparison_summary}</p><p>{comparison_details}</p>{comparison_body}
<p class="note">³ Score rate = (wins + ½ draws) / rallies; pp = percentage points. Both models in each comparison face the same reference policies. These selected-checkpoint results are descriptive, not a statistical guarantee of improvement against unseen opponents.</p>
<h3>New Selfplay against learned opponents</h3><p>Selfplay v3 is the first player in every row. The v2 entries use their original frozen weights.</p>{result_table(matchups)}</section>
<section><h2>How the weights were learned</h2><p>Each run collects {imitation_steps:,} DAgger decisions over {config['dagger_rounds']} rounds. A predictive attacker labels the learner’s visited states. PPO then adds {per_run-imitation_steps:,} decisions ({config['updates']} updates × {config['envs']} environments × {config['horizon']} steps), facing three original baselines, two scripts and up to five historical actors.</p>
<p>The actor is <code>6 → 64 → 64 → 3</code> with Tanh hidden layers, 4,803 parameters and 4,672 MACs per decision. Architecture, reward recipe and training budget remain the same as the previous experiment. PPO uses GAE (γ=.99, λ=.95), clipping .2, four epochs, gradient clipping, decaying learning rate, entropy regularization and a decaying imitation anchor. Rewards are +1 for scoring, −1 for conceding and +.02 per return.</p>
<p>Training uses the 1,600-tick limit without loop termination, because sampled actions, changing weights and opponent changes invalidate a deterministic loop proof. Frozen validation and final evaluation enable loop termination. This changes episode lengths and reward accumulation relative to deterministic evaluation; the config and standalone contract document the distinction. Critics and optimizers are training-only and are absent from ONNX.</p>
<p>Methods: <a href="https://arxiv.org/abs/1707.06347">PPO</a>, <a href="https://arxiv.org/abs/1011.0686">DAgger</a>, <a href="https://docs.cleanrl.dev/rl-algorithms/ppo/">CleanRL implementation notes</a>.</p></section>
<section><h2>Reproduce and verify</h2><p>Start with <code>training/README.md</code>. The versioned contract and scalar reference are in <code>training/v3/</code>; v1 and v2 remain archived. Logs, selected checkpoint pairs from all three seeds, configuration, selection records and final results are retained in <code>training/results/v3/</code>.</p><p>The export check compares {next(iter(export.values()))['observations']:,} observations per actor between PyTorch and native ONNX with identical actions. It also generates {len(replays)} complete native-ONNX replays ({replay_frames:,} frames) for exact TypeScript comparison, including early loop draws. Reproduce with <code>verify_export.py</code> followed by <code>check_replays.ts</code>. Models carry <code>pong-v3</code> metadata and distinct filenames.</p>
<p><a href="v2/report.html">Archived v2 training</a> · <a href="v1/report.html">Archived v1 training</a> · <a href="rule-study.html">Timeout and physics study</a></p></section>
<footer>Rally Lab · Pong v3 · Fixed physics, 1,600-tick limit, deterministic model-loop exits</footer></main></html>'''
    (args.out / 'report.html').write_text(html)
    print(args.out / 'report.html')


if __name__ == '__main__':
    main()

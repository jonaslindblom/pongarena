"""Render the completed experimental rule study without changing the live arena."""
import argparse
from html import escape
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/rally-lab-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    'v2': 'Current v2', 'cap12': 'Ball cap 12', 'cap14': 'Ball cap 14',
    'paddle2.25': 'Paddle 2.25 · JS rounding', 'paddle2.5': 'Paddle 2.5 · JS rounding',
    'paddle2.25-symmetric': 'Paddle 2.25 · symmetric',
    'paddle2.5-symmetric': 'Paddle 2.5 · symmetric',
    'paddle2.25-cap14': 'Paddle 2.25 + cap 14',
    'center6': 'Center shot speed 6', 'larger-paddle': 'Larger paddle',
}
PAIRS = {'NN': 'Selfplay vs Selfplay', 'NA': 'Selfplay vs attacker', 'NP': 'Selfplay vs Apprentice',
         'AA': 'Attacker vs attacker', 'AD': 'Attacker vs interceptor',
         'DD': 'Interceptor vs interceptor', 'NT': 'Selfplay vs Tracker',
         'AH': 'Attacker vs delayed interceptor'}


def read_complete(path, expected):
    data = json.loads(path.read_text())
    if len(data['records']) != expected:
        raise ValueError(f'{path}: expected {expected} completed pairings, found {len(data["records"])}')
    return data


def index(data):
    return {(r['config'], r['pair'], r['limit']): r for r in data['rows']}


def pct(n, total=1):
    return f'{n / total:.1%}'


def median(row):
    value = row['median_decisive_returns']
    return '—' if value is None else f'{value:g}'


def table(headers, rows):
    head = ''.join(f'<th scope="col">{escape(h)}</th>' for h in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in row) + '</tr>' for row in rows)
    return f'<div class="table-wrap" tabindex="0"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def early_cycle_ticks(record, limit):
    point = np.array(record['point_tick'])
    cycle = np.array(record['cycle_tick'])
    return np.minimum(np.where(point > 0, point, np.where(cycle > 0, cycle, limit)), limit)


def chart(data, rows, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#536153', 'text.color': '#203c30',
                         'axes.edgecolor': '#ccd4c6', 'figure.facecolor': '#f5f6f0',
                         'axes.facecolor': '#fdfef9'})
    fig = plt.figure(figsize=(13.5, 10.5), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.4])
    top = fig.add_subplot(grid[0, 0])
    line = fig.add_subplot(grid[0, 1])
    physics = fig.add_subplot(grid[1, :])
    fig.suptitle('Rally Lab  /  What makes a good rally?', fontsize=22, fontweight='bold', ha='left', x=.03)
    pairs = ['NN', 'NA', 'NP']
    colors = ['#397451', '#cf9e4b', '#899eab', '#b35f4e']
    categories = ['Scored by 800', 'Scores later, by 3200', 'Proven repeating by 3200', 'Unresolved at 3200']
    parts = []
    for pair in pairs:
        r = rows['v2', pair, 800]
        parts.append([r['wins'] + r['losses'], r['draws_scoring_later'],
                      r['draws_proven_cyclic_by_end'], r['draws_unresolved_at_end']])
    parts = np.array(parts) / (10 * data['seed_count']) * 100
    left = np.zeros(3)
    for i, (label, color) in enumerate(zip(categories, colors)):
        top.barh(np.arange(3), parts[:, i], left=left, color=color, label=label, height=.55)
        for y, width, start in zip(np.arange(3), parts[:, i], left):
            if width >= 9:
                top.text(start + width / 2, y, f'{width:.1f}%', ha='center', va='center',
                         color='white' if i == 0 else '#203c30', fontsize=9)
        left += parts[:, i]
    top.set(yticks=np.arange(3), yticklabels=['Selfplay / self', 'Selfplay / attacker', 'Selfplay / Apprentice'],
            xlim=(0, 100), xlabel='Share of all rallies (%)', title='Where the 800-tick draws go')
    top.invert_yaxis()
    top.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0, -.19), frameon=False, ncol=1)
    for pair, color in zip(pairs, ['#397451', '#a06b40', '#587a9a']):
        limits = [800, 1600, 3200]
        line.plot(limits, [rows['v2', pair, t]['draws'] / rows['v2', pair, t]['n'] * 100 for t in limits],
                  '-o', color=color, label=PAIRS[pair].replace(' vs ', ' / '), lw=2, ms=5)
    line.set(title='Longer deadlines · unchanged v2 physics', xlabel='Rally limit (ticks)',
             ylabel='Draws / all rallies (%)', xticks=[800, 1600, 3200], ylim=(0, 100))
    line.legend(loc='upper right', fontsize=8, frameon=False)
    configs = list(data['configs'])
    values = []
    for name in configs:
        r = rows[name, 'AA', 800]
        draw = r['draws'] / r['n']
        values.append([r['early_rate'], 1 - draw - r['early_rate'] - r['long_scored_rate'],
                       r['long_scored_rate'], draw])
    values = np.array(values) * 100
    left = np.zeros(len(configs))
    for i, (label, color) in enumerate(zip(['Scores before 2 returns', 'Scores after 2–5 returns',
                                           'Scores after 6+ returns', 'Draw at 800'],
                                          ['#be795e', '#b9c7a8', '#397451', '#899eab'])):
        physics.barh(np.arange(len(configs)), values[:, i], left=left, height=.57, color=color, label=label)
        for y, width, start in zip(np.arange(len(configs)), values[:, i], left):
            if width >= 9:
                physics.text(start + width / 2, y, f'{width:.1f}%', ha='center', va='center',
                             color='white' if i == 2 else '#203c30', fontsize=9)
        left += values[:, i]
    physics.set(yticks=np.arange(len(configs)), yticklabels=[LABELS[n] for n in configs], xlim=(0, 100),
                xlabel='Share of all attacker-mirror rallies (%)',
                title='Physics candidates · same serves, rule-aware attacking scripts')
    physics.invert_yaxis()
    physics.legend(loc='upper left', bbox_to_anchor=(0, -.14), frameon=False, fontsize=9, ncol=2)
    for ax in [top, line, physics]:
        ax.grid(axis='x' if ax != line else 'y', color='#e1e5db', linewidth=.6)
        ax.set_axisbelow(True)
    fig.savefig(out / 'rule-study-chart.png', dpi=160)
    fig.savefig(out / 'rule-study-chart.svg')
    plt.close(fig)
    svg = out / 'rule-study-chart.svg'
    svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines()) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, default=Path('training/results/rule-study'))
    p.add_argument('--out', type=Path, default=Path('public/training'))
    args = p.parse_args()
    scan = read_complete(args.results / 'scan.json', 80)
    holdout = read_complete(args.results / 'holdout.json', 48)
    if set(scan['seeds']) & set(holdout['seeds']):
        raise ValueError('Scan and confirmation seed sets overlap')
    rows, scan_rows = index(holdout), index(scan)
    records = {(r['config'], r['pair']): r for r in holdout['records']}
    args.out.mkdir(parents=True, exist_ok=True)
    chart(holdout, rows, args.out)
    timeout_rows, duration_rows, match_rows = [], [], []
    for pair in ['NN', 'NA', 'NP']:
        for limit in [800, 1600, 3200]:
            r = rows['v2', pair, limit]
            timeout_rows.append([PAIRS[pair], f'{limit:,}', f'{r["wins"]} / {r["losses"]} / {r["draws"]}',
                                 pct(r['draws'], r['n']), r['cycles_by_limit'], r['draws_scoring_later']])
        base = rows['v2', pair, 800]
        proposed = rows['v2', pair, 1600]
        duration_rows.append([PAIRS[pair], f'{base["mean_ticks"] / 30:.2f} s',
                              f'{proposed["mean_ticks"] / 30:.2f} s',
                              f'{early_cycle_ticks(records["v2", pair], 1600).mean() / 30:.2f} s'])
        for limit in [800, 1600]:
            r = rows['v2', pair, limit]
            match_rows.append([PAIRS[pair], f'{limit:,}',
                               f'{r["match_wins"]} / {r["match_losses"]} / {r["match_draws"]}',
                               f'{r["paired_match_wins"]} / {r["paired_match_losses"]} / {r["paired_match_draws"]}'])
    physics_rows, other_rows = [], []
    for name, conf in holdout['configs'].items():
        aa, ad, ah, nn = [rows[name, pair, 800] for pair in ['AA', 'AD', 'AH', 'NN']]
        physics_rows.append([LABELS[name], f'{conf["paddle_speed"]:g}', f'{conf["min_vx"]}–{conf["max_vx"]}',
                             conf['radius'], median(aa), pct(aa['draws'], aa['n']),
                             '—' if aa['first_receiver_win_rate'] is None else pct(aa['first_receiver_win_rate']),
                             pct(aa['early_rate'])])
        other_rows.append([LABELS[name], pct(ad['wins'], ad['n']), pct(ad['draws'], ad['n']),
                           pct(ah['early_rate']), median(ah), pct(nn['draws'], nn['n']), median(nn)])
    scan_table = table(['Candidate', 'AA median returns¹', 'AA draws', 'AA early points²', 'AH early points²'],
                       [[LABELS[name], median(scan_rows[name, 'AA', 800]),
                         pct(scan_rows[name, 'AA', 800]['draws'], scan_rows[name, 'AA', 800]['n']),
                         pct(scan_rows[name, 'AA', 800]['early_rate']), pct(scan_rows[name, 'AH', 800]['early_rate'])]
                        for name in scan['configs']])
    max_point = max(max(r['point_tick']) for r in holdout['records'] if r['config'] == 'v2')
    late_after_1600 = sum(rows['v2', pair, 1600]['draws_scoring_later'] for pair in PAIRS)
    unresolved = sum(rows['v2', pair, 3200]['draws_unresolved_at_end'] for pair in PAIRS)
    nn = rows['v2', 'NN', 800]
    npair = rows['v2', 'NP', 800]
    faster = [rows[name, 'AA', 800] for name in ['paddle2.25-symmetric', 'paddle2.5-symmetric']]
    large, current = rows['larger-paddle', 'NN', 800], rows['v2', 'NN', 800]
    cap14 = rows['cap14', 'AA', 800]
    conclusion = (f'All scored rallies in the eight current-v2 pairings finished by tick {max_point:,}; '
                  'extending 1,600 to 3,200 recovered no additional points.' if not late_after_1600 else
                  f'Extending 1,600 to 3,200 recovered {late_after_1600} additional points across the eight current-v2 pairings.')
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rally Lab · Rule study</title><link rel="icon" href="/mark.svg">
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f6f0;color:#203c30;font:15px/1.65 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:40px 24px 64px}}nav{{display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:44px}}a{{color:#376e43}}h1{{font-size:clamp(36px,6vw,64px);line-height:1.06;letter-spacing:-.055em;max-width:800px;margin:12px 0 22px}}h2{{font-size:26px;letter-spacing:-.025em;margin:0 0 16px}}h3{{font-size:19px;margin:26px 0 10px}}p{{color:#5a6b5b;max-width:920px}}.eyebrow{{font:11px monospace;letter-spacing:.14em;color:#75935c}}.notice{{display:inline-block;font-size:12px;color:#466445;background:#e5eddc;padding:6px 12px;border-radius:30px}}.stats{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:28px 0}}section,.stat{{background:#fdfef9;border:1px solid #dce3d5;border-radius:12px;padding:23px}}section{{margin-top:20px;min-width:0}}.stat b{{display:block;font-size:30px;letter-spacing:-.04em}}.stat small{{font-size:11px;color:#70816c}}img{{width:100%;height:auto;border-radius:8px}}.table-wrap{{max-width:100%;overflow-x:auto;border-radius:5px}}table{{border-collapse:collapse;width:100%;font-size:13px;text-align:left}}th,td{{padding:11px 10px;border-bottom:1px solid #e2e7dc;white-space:nowrap}}th{{font-weight:500;color:#70816c}}td:first-child{{font-weight:500}}code{{background:#edf1e7;border-radius:4px;padding:2px 5px;overflow-wrap:anywhere}}details{{margin-top:22px}}summary{{cursor:pointer;color:#376e43;font-weight:600}}.note{{font-size:13px}}.recommendation{{border-left:4px solid #789461}}footer{{font-size:12px;color:#75816d;margin-top:28px}}@media(max-width:620px){{main{{padding:26px 14px 44px}}section{{padding:17px}}.stats{{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}.stat{{padding:16px}}.stat b{{font-size:26px}}nav{{margin-bottom:30px}}}}
</style><main><nav><a href="/">← Back to the arena</a><a href="report.html">Training &amp; learning curves ↗</a></nav>
<div class="eyebrow">RALLY LAB · EXPERIMENTAL RULE STUDY</div><h1>Give good rallies<br>time to finish.</h1><span class="notice">Evaluation only · live Pong v2 is unchanged</span>
<p>Long rallies can be unfinished contests or exact repeating exchanges. We continued identical serves to three deadlines, proved deterministic loops where possible, and compared modest changes to paddle movement, return speed and paddle size.</p>
<div class="stats"><div class="stat"><b>{len(scan['configs'])}</b><small>Physics candidates screened</small></div><div class="stat"><b>{len(holdout['configs'])}</b><small>Candidates on separate confirmation seeds</small></div><div class="stat"><b>{holdout['seed_count'] * 10:,}</b><small>Rallies per confirmed matchup and candidate</small></div><div class="stat"><b>3 deadlines</b><small>800 · 1,600 · 3,200 ticks</small></div></div>
<section><h2>What the measurements say</h2><p>{conclusion} At 800 ticks, {pct(nn['draws_proven_cyclic_by_end'], nn['draws'])} of the Selfplay-mirror draws were eventually proven repeating exchanges. In Selfplay vs Apprentice, however, {npair['draws_scoring_later']} of {npair['draws']} draws later became points. A longer deadline helps some matchups much more than others.</p><a href="rule-study-chart.svg"><img src="rule-study-chart.png" alt="800-tick rally outcomes split into scored, later scoring, proven loops and unresolved; draw rates at longer deadlines; attacker-mirror outcomes across physics candidates"></a><p class="note">Confirmation set only. A loop proven by tick 3,200 may not yet have been detected at tick 800. Stacked bars distinguish that eventual proof from a point that simply needed more time. There are {unresolved} unresolved, unproven rallies at 3,200 across the eight current-v2 matchups.</p></section>
<section><h2>Does a longer deadline help?</h2><p>At 30 decisions per second, the limits correspond to 26.7, 53.3 and 106.7 seconds of rally time. W / L / D is from the first named player’s perspective. Each row uses the same {holdout['seed_count'] * 10:,} openings.</p>{table(['Matchup', 'Limit', 'Rallies W / L / D', 'Draw rate', 'Loops proven by limit', 'Draws scoring later³'], timeout_rows)}
<p class="note">³ Later points through tick 3,200. “Loops proven by limit” is a subset of draws, not an additional outcome. The deadline changes no decisions: none of these policies observes the clock.</p>
<h3>Longer limit, with shorter proven stalemates</h3><p>A stateless deterministic model that repeats the complete physical state will make the same decisions forever. Ending that proven loop immediately preserves its eventual draw outcome. These are measured mean rally durations, excluding serves and display pauses.</p>{table(['Matchup', 'Current 800', '1,600 without loop exit', '1,600 with proven-loop exit'], duration_rows)}
<p class="note">Loop exit is an evaluation shortcut for deterministic controllers. It is not valid for a human, a stochastic training policy or a controller with untracked memory. The delayed proxy’s observation history is included in its cycle key. All players still need a finite maximum duration.</p></section>
<section><h2>Physics: scoring, recovery and defense</h2><p>These six candidates were selected for confirmation after the ten-candidate scan. Scripts know each candidate’s rules. The learned models keep their original v2 weights, so their results measure transfer, not how strong a retrained model could become.</p>{table(['Candidate', 'Paddle speed', 'Return VX', 'Hit radius', 'AA median returns¹', 'AA draws', 'First receiver wins⁴', 'AA early points²'], physics_rows)}
<p class="note">AA = attacker vs itself. ¹ Scored rallies only. ² A point before the second return, as a share of all rallies. ⁴ Among scored attacker-mirror rallies; a value near 50% indicates little first-receiver advantage for this script. Every row uses an 800-tick limit.</p>
<h3>Can an attacker still score?</h3>{table(['Candidate', 'Attacker wins vs defense', 'Draws vs defense', 'Delayed proxy early points²', 'Delayed proxy median¹', 'Selfplay-mirror draws', 'Selfplay-mirror median¹'], other_rows)}
<p>The delayed interceptor acts on a five-tick-old observation, including its own paddle position. It is a reaction-delay sensitivity test, not a human playtest. The attacker remains a greedy reference script, not a proof of optimal play.</p>
<details><summary>See all ten candidates from the initial scan</summary><p>{scan['seed_count']} seeds × five serves × both sides = {scan['seed_count'] * 10:,} rallies per candidate and matchup. These exploratory results selected the confirmation set; the larger tables above use different seeds.</p>{scan_table}</details>
<p class="note">Fractional paddle speeds use quarter-unit positions. “Symmetric” rounds half ties away from zero for vertical shots; the original JavaScript rule rounds toward positive infinity. This difference matters only once fractional offsets are reachable. The larger-paddle candidate changes hit radius 15→17, center bounds [15,145]→[17,143], and normalizes shot response to the new radius. It preserves the speed and angle at a full edge hit. Center-shot speed 6 changes the return range to 6–13; initial serves remain at horizontal speed 5.</p></section>
<section><h2>Fairer matches need both sides</h2><p>A five-rally game gives one player three initial receptions and the other two. For model ranking, replay the same five openings with the policies exchanging sides, then aggregate the ten point outcomes. Repeat over many seeds.</p>{table(['Matchup', 'Rally limit', 'Five-rally matches W / L / D', 'Paired ten-rally matches W / L / D'], match_rows)}
<p class="note">The columns contain {holdout['seed_count'] * 2} single games and {holdout['seed_count']} paired games, respectively. Identical-model paired matches tie by construction: they receive exactly the same opportunities. Those ties show symmetry rather than a failure of attacking play. Awarding each player half a point instead of zero for a drawn rally would leave these match margins unchanged.</p></section>
<section class="recommendation"><h2>A measured next step</h2><p>Retain the current physics for now. Faster paddles produced {pct(faster[0]['draws'], faster[0]['n'])} and {pct(faster[1]['draws'], faster[1]['n'])} attacker-mirror draws at speeds 2.25 and 2.5. A ball cap of 14 restored a {pct(cap14['first_receiver_win_rate'])} first-receiver advantage among scored attacker-mirror rallies. The larger paddle remains a possible human-playtest candidate, but frozen Selfplay-mirror draws rose from {pct(current['draws'], current['n'])} to {pct(large['draws'], large['n'])}; retraining could change that result.</p><p>Evaluate a 1,600-tick competition limit with early exits for proven model loops, and use paired side assignments for rankings. Keep a draw as a valid outcome. Physics changes deserve retraining and human playtesting before a new rules version is adopted; this study alone does not establish a universal optimum.</p><p>The existing six inputs omit the clock and match score. Time-dependent physics or deliberate end-of-match tactics would require a separate design decision. Training rewards and discounting can also change defensive behavior without changing the game.</p></section>
<section><h2>Method and reproduction</h2><p>The screen uses {scan['seed_count']} randomly selected seeds (sampler seed {scan['rng_seed']}); confirmation uses {holdout['seed_count']} separate seeds (sampler seed {holdout['rng_seed']}). Every configuration sees the same openings within its phase, all five serves and both policy-side assignments. Mirrored duplicates are retained. Counts are descriptive; individual rallies are not treated as independent statistical samples.</p><p>Frozen v2 PyTorch actors use deterministic first-argmax on one CPU thread. No models were retrained. Exact cycle keys contain ball position, velocity, both paddle positions, and any delayed-controller history. Scores and counters are excluded only because these policies and candidate physics do not use them.</p><p>Run <code>.venv/bin/python training/scripts/rule_study.py</code> for the screen. The confirmation command and verification results are in <code>training/results/rule-study/README.md</code>. Rebuild this page with <code>.venv/bin/python training/scripts/rule_study_report.py</code>. Full per-rally point times, loop times, seeds, parameters and source hashes are retained in <code>training/results/rule-study/scan.json</code> and <code>holdout.json</code>.</p></section>
<footer>Rally Lab · Experimental report · Published v1/v2 contracts and live gameplay remain unchanged</footer></main></html>'''
    (args.out / 'rule-study.html').write_text(html)
    print(args.out / 'rule-study.html')


if __name__ == '__main__':
    main()

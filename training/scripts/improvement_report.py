"""Render the fixed-v3 training investigation without modifying earlier reports."""
import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/rally-lab-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ('selfplay-v2', 'attacker')
NAMES = {'selfplay-v2': 'Original Selfplay v2', 'v2': 'Original Selfplay v2',
         'attacker': 'Scripted attacker', 'interceptor': 'Interceptor', 'selfplay-v3': 'Released Selfplay v3'}


def read(path):
    return json.loads(path.read_text())


def source(path):
    return ROOT / path


def score(row):
    return row['minimum_score'] if 'minimum_score' in row else row['selection_min']


def mean_score(row):
    return row['mean_score'] if 'mean_score' in row else row['selection_mean']


def table(headers, rows):
    head = ''.join(f'<th scope="col">{escape(h)}</th>' for h in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in row) + '</tr>' for row in rows)
    return f'<div class="table-wrap" tabindex="0"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def wld(row):
    return f'{row["wins"]} / {row["losses"]} / {row["draws"]}'


def verify_hash(path, expected, description):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'{description} does not match the frozen experiment.')


def plots(runs, diagnosis, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#536153', 'text.color': '#203c30',
                         'axes.edgecolor': '#ccd4c6', 'figure.facecolor': '#f5f6f0',
                         'axes.facecolor': '#fdfef9'})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    fig.suptitle('Rally Lab  /  Training under fixed Pong v3 rules', fontsize=21,
                 fontweight='bold', ha='left', x=.04)
    palette = ['#397451', '#b27836', '#44728d', '#9b554e', '#8c7a9f', '#6c8562', '#bc925e']
    for ax, family, title in zip(axes[0], ['ppo', 'distillation'],
                                 ['Focused PPO · raw development curves', 'Teacher distillation · raw development curves']):
        included = [run for run in runs if run['family'] == family]
        for i, run in enumerate(included):
            ax.plot([r['env_steps'] / 1e6 for r in run['rows']], [score(r) for r in run['rows']],
                    label=run['label'], color=palette[i % len(palette)], lw=1.6)
        ax.axhline(.5, color='#858a7b', lw=.8, linestyle='--')
        ax.set(title=title, xlabel='New training decisions in this run (millions)',
               ylabel='Weaker target score: wins + ½ draws', ylim=(0, 1))
        if included:
            ax.legend(loc='upper left', bbox_to_anchor=(0, -.2), fontsize=8, frameon=False)
        else:
            ax.text(.5, .5, 'No recorded runs', ha='center', va='center', transform=ax.transAxes)
    diagnostic = {(r['model'], r['opponent'], r['mode']): r for r in diagnosis['rows']}
    opponents = ['attacker', 'interceptor', 'v2']
    ax = axes[1, 0]
    x = np.arange(3)
    point = [diagnostic['v3-seed29', op, 'greedy-full']['mean_discounted_point_reward'] for op in opponents]
    bonus = [diagnostic['v3-seed29', op, 'greedy-full']['mean_discounted_return_bonus'] for op in opponents]
    ax.bar(x - .17, point, width=.34, color='#a76554', label='Discounted point rewards')
    ax.bar(x + .17, bonus, width=.34, color='#9aaf7d', label='Discounted return bonuses')
    ax.scatter(x, np.array(point) + bonus, marker='D', color='#264f39', s=30, label='Net training reward', zorder=3)
    ax.axhline(0, color='#858a7b', lw=.8)
    ax.set(title='Released v3 · what its reward measures', xticks=x,
           xticklabels=['Attacker', 'Interceptor', 'Original v2'], ylabel='Mean discounted reward per rally')
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1, 1]
    for shift, mode, label, color in [(-.18, 'greedy-full', 'Greedy deployment actions', '#397451'),
                                     (.18, 'sampled-full', 'Sampled training actions', '#b78b58')]:
        ax.bar(x + shift, [diagnostic['v3-seed29', op, mode]['mean_net_points'] for op in opponents],
               width=.36, label=label, color=color)
    ax.axhline(0, color='#858a7b', lw=.8)
    ax.set(title='Released v3 · sampling changes outcomes', xticks=x,
           xticklabels=['Attacker', 'Interceptor', 'Original v2'], ylabel='Rally wins minus losses / all rallies', ylim=(-1, 1))
    ax.legend(fontsize=8, frameon=False)
    for ax in axes.flat:
        ax.grid(axis='y', color='#e1e5db', linewidth=.6)
        ax.set_axisbelow(True)
    fig.savefig(out / 'improvement-curves.png', dpi=150)
    fig.savefig(out / 'improvement-curves.svg')
    plt.close(fig)
    svg = out / 'improvement-curves.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')


def result_rows(blocks):
    rows = []
    passed = len(blocks) == 3
    for label, data in blocks:
        if data['rules'] != 'pong-v3':
            raise ValueError('Final results must use the fixed v3 rules.')
        if data.get('candidate_runtime') != 'native_onnxruntime' or data.get('opponent_runtimes', {}).get('selfplay-v2') != 'native_onnxruntime':
            raise ValueError('Final results require the actual candidate and v2 ONNX runtimes.')
        if not all(name in data['opponents'] for name in TARGETS):
            passed = False
        for name, r in data['opponents'].items():
            if set(r['sides']) != {'0', '1'}:
                raise ValueError('Every final opponent must include both physical sides.')
            ci = r['point_margin_ci95']
            criteria = (r['paired_match_win_rate'] > .5 and ci[0] > 0
                        and all(s['wins'] > s['losses'] for s in r['sides'].values()))
            if name in TARGETS:
                passed = passed and criteria
            rows.append([label, NAMES.get(name, name), wld(r), r['loop_draws'], r['timeout_draws'],
                         f'{r["paired_match_wins"]} / {r["paired_match_losses"]} / {r["paired_match_draws"]}',
                         ' · '.join(f'{r["sides"][side]["wins"]}/{r["sides"][side]["losses"]}' for side in ['0', '1']),
                         f'{100 * ci[0]:+.1f}…{100 * ci[1]:+.1f} pp',
                         ('Pass' if criteria else 'Does not pass') if name in TARGETS else 'Guardrail'])
    return rows, bool(passed)


def pooled_rows(blocks):
    rows = []
    for name in TARGETS:
        values = [data['opponents'][name] for _, data in blocks if name in data['opponents']]
        if not values:
            continue
        counts = {key: sum(r[key] for r in values) for key in
                  ['wins', 'losses', 'draws', 'rallies', 'match_wins', 'match_losses', 'match_draws',
                   'paired_match_wins', 'paired_match_losses', 'paired_match_draws']}
        matches = sum(counts[key] for key in ['paired_match_wins', 'paired_match_losses', 'paired_match_draws'])
        rows.append([NAMES[name], wld(counts), f'{100*counts["draws"]/counts["rallies"]:.1f}%',
                     f'{counts["match_wins"]} / {counts["match_losses"]} / {counts["match_draws"]}',
                     f'{counts["paired_match_wins"]} / {counts["paired_match_losses"]} / {counts["paired_match_draws"]}',
                     f'{100*counts["paired_match_wins"]/matches:.1f}%'])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT / 'training/results/training-improvement')
    parser.add_argument('--out', type=Path, default=ROOT / 'public/training')
    args = parser.parse_args()
    manifest = read(args.results / 'manifest.json')
    protocol = read(args.results / 'protocol.json')
    diagnosis = read(ROOT / 'training/results/training-diagnosis/diagnosis.json')
    if manifest['rules'] != 'pong-v3':
        raise ValueError('This report is for fixed Pong v3 rules.')
    for path_key, hash_key in [('engine', 'sha256'), ('contract', 'contract_sha256')]:
        frozen = protocol['rules_unchanged']
        verify_hash(source(frozen[path_key]), frozen[hash_key], 'Game rules')
    reference = protocol['primary_targets']['selfplay-v2']
    verify_hash(source(reference['file']), reference['sha256'], 'Original v2 weights')
    attacker = protocol['primary_targets']['attacker']
    verify_hash(source(attacker['implementation'].split(':')[0]), attacker['source_sha256'], 'Scripted attacker source')
    runs = []
    for item in manifest['runs']:
        directory = source(item['path'])
        rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
        if not rows:
            raise ValueError(f'No learning measurements in {directory}')
        runs.append(dict(item, config=read(directory / 'config.json'), rows=rows,
                         best=max(rows, key=lambda r: (score(r), mean_score(r)))))
    blocks = [(item['name'], read(source(item['file']))) for item in manifest.get('holdouts', [])]
    final_rows, passed = result_rows(blocks)
    seed_file = args.results / 'holdout-seeds.json'
    verify_hash(seed_file, protocol['holdout']['sha256'], 'Reserved seeds')
    reserved = read(seed_file)['blocks']
    for label, result in blocks:
        expected = next((b for b in reserved if b['name'] == label), None)
        if expected is None or result['seeds'] != expected['seeds']:
            raise ValueError(f'{label} does not match the predeclared held-out seeds.')
        for key in ['samples', 'seed', 'confidence', 'method']:
            if result['bootstrap'][key] != protocol['evaluation']['bootstrap'][key]:
                raise ValueError(f'{label} does not use the predeclared bootstrap settings.')
    if len(set(label for label, _ in blocks)) != len(blocks):
        raise ValueError('Repeated holdout blocks cannot count as independent confirmation.')
    candidate = manifest.get('candidate')
    frozen_candidate = None
    if candidate:
        frozen_candidate = read(source(candidate['freeze_file']))
        for path_key, hash_key in [('checkpoint', 'checkpoint_sha256'), ('onnx', 'onnx_sha256')]:
            verify_hash(source(frozen_candidate[path_key]), frozen_candidate[hash_key], f'Selected {path_key}')
        verify_hash(source(candidate['model_path']), frozen_candidate['onnx_sha256'], 'Released candidate')
        for label, result in blocks:
            verify_hash(source(result['model']), frozen_candidate['onnx_sha256'], f'{label} candidate')
    elif blocks:
        raise ValueError('Final results require a frozen candidate.')
    comparisons = []
    development = manifest.get('development_result')
    if development:
        selected_development = read(source(development))
        for path in sorted(source(development).parent.parent.glob('*/evaluation.json')):
            result = read(path)
            if result['seeds'] != selected_development['seeds']:
                raise ValueError('The shared development comparison must use identical seed lists.')
            for name in TARGETS:
                r = result['opponents'][name]
                comparisons.append([path.parent.name, NAMES[name], wld(r),
                                    f'{r["paired_match_wins"]} / {r["paired_match_losses"]} / {r["paired_match_draws"]}',
                                    f'{r["score_rate"]:.3f}'])
    args.out.mkdir(parents=True, exist_ok=True)
    plots(runs, diagnosis, args.out)
    lookup = {(r['model'], r['opponent'], r['mode']): r for r in diagnosis['rows']}
    regression_rows = []
    for model, label in [('v2', 'Released v2'), ('v3-seed7', 'v3 seed 7'),
                         ('v3-seed17', 'v3 seed 17'), ('v3-seed29', 'Released v3 · seed 29')]:
        regression_rows.append([label] + [wld(lookup[model, op, 'greedy']) for op in ['attacker', 'interceptor', 'v2']])
    reward = lookup['v3-seed29', 'v2', 'greedy-full']
    experiment_rows = []
    for run in runs:
        best = run['best']
        targets = best['opponents']
        v2 = targets.get('selfplay-v2', targets.get('v2'))
        experiment_rows.append([run['label'], f'{run["rows"][-1]["env_steps"]:,}',
                                f'{score(best):.3f}', wld(v2), wld(targets['attacker']),
                                run.get('note', 'Development experiment')])
    total = sum(r['rows'][-1]['env_steps'] for r in runs)
    if blocks:
        verdict = ('The selected export passes the predeclared criteria against both targets in all three fresh blocks.' if passed else
                   'The recorded final evaluation does not satisfy the complete predeclared success criterion.')
    else:
        verdict = 'No final holdout evaluation has been recorded. Development improvements are not yet evidence of consistent wins on fresh tests.'
    candidate_text = '<p>No candidate has been released from this investigation.</p>'
    if candidate:
        candidate_text = f'<p><strong>{escape(candidate["name"])}</strong> · {escape(candidate["architecture"].rstrip("."))}. {escape(candidate["selection_note"])}</p>'
        candidate_text += f'<p>{escape(candidate.get("design_note", ""))} The frozen feature layers compute next-contact landing position and flight time from the original six observations. The learned head predicts a paddle target; fixed feedback converts its error into the three action scores. All inference stays inside the standalone ONNX graph.</p>'
        candidate_text += f'<p class="note">{frozen_candidate["trainable_parameters"]:,} learned parameters · {frozen_candidate["total_parameters"]:,} total parameters including fixed weights · {frozen_candidate["macs"]:,} multiply-accumulates · {frozen_candidate["bytes"]:,} bytes. The teacher supplies training targets; it does not run beside the submitted model.</p>'
        model_path = candidate['model_path']
        if model_path.startswith('public/'):
            candidate_text += f'<p><a class="download" href="/{escape(model_path[7:], quote=True)}" download>Download the standalone ONNX weights</a></p>'
    final_table = table(['Block', 'Opponent', 'Rally W / L / D', 'Loop draws', 'Timeout draws',
                         'Paired matches W / L / D', 'Left W/L · right W/L', 'Point-margin 95% interval', 'Protocol'], final_rows) if final_rows else ''
    pooled_table = table(['Opponent', 'All final rallies W / L / D', 'Rally draw rate',
                          'Five-rally games W / L / D', 'Paired matches W / L / D', 'Paired match win rate'], pooled_rows(blocks)) if blocks else ''
    selection_disclosure = ''
    if blocks:
        tested = manifest['final_candidates_tested']
        selection_disclosure = (f'{tested} frozen candidate was tested on these reserved blocks.' if tested == 1 else
                                f'{tested} candidates were tested on these reserved blocks; this reuse limits their independence from model selection.')
        selection_disclosure += ' The three blocks are confirmation of one chosen export, not three independent training runs.'
    development_table = table(['Candidate', 'Opponent', 'Rally W / L / D', 'Paired matches W / L / D',
                               'Score including half draws'], comparisons) if comparisons else ''
    verification = manifest.get('verification_note', 'Verification results will be recorded with the frozen release.')
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rally Lab · Training investigation</title><link rel="icon" href="/mark.svg">
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f6f0;color:#203c30;font:15px/1.65 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:42px 24px 64px}}nav{{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:46px}}a{{color:#376e43}}h1{{font-size:clamp(36px,6vw,64px);letter-spacing:-.055em;line-height:1.08;max-width:850px;margin:12px 0 22px}}h2{{font-size:26px;letter-spacing:-.03em;margin:0 0 16px}}h3{{font-size:19px;margin:26px 0 12px}}p{{color:#5a6b5b;max-width:940px}}.eyebrow{{font:11px monospace;color:#75935c;letter-spacing:.14em}}.badge{{display:inline-block;background:#e4ecdc;color:#45633f;border-radius:20px;padding:6px 12px;font-size:12px}}.stats{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:28px 0}}.stat,section{{padding:23px;background:#fdfef9;border:1px solid #dce3d5;border-radius:12px;min-width:0}}.stat b{{display:block;font-size:30px;letter-spacing:-.04em}}.stat small{{font-size:11px;color:#70816c}}section{{margin-top:20px}}img{{width:100%;height:auto;border-radius:8px}}.table-wrap{{max-width:100%;overflow-x:auto}}table{{border-collapse:collapse;width:100%;text-align:left;font-size:13px}}th,td{{padding:11px 10px;border-bottom:1px solid #e2e7dc;white-space:nowrap}}th{{color:#70816c;font-weight:500}}.note{{font-size:13px}}code{{background:#edf1e7;border-radius:4px;padding:2px 5px;overflow-wrap:anywhere}}.download{{display:inline-block;padding:9px 15px;border:1px solid #bfd0b7;border-radius:7px;text-decoration:none}}.verdict{{border-left:4px solid #789461}}footer{{font-size:12px;color:#75816d;margin-top:30px}}@media(max-width:620px){{main{{padding:26px 14px 44px}}section{{padding:17px}}.stats{{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}.stat{{padding:15px}}.stat b{{font-size:25px}}nav{{margin-bottom:30px}}}}
</style><main><nav><a href="/">← Back to the arena</a><a href="report.html">Original v3 training ↗</a><a href="rule-study.html">Frozen-rules study ↗</a></nav>
<div class="eyebrow">PONG V3 · TRAINING INVESTIGATION</div><h1>Train for the contest<br>we actually care about.</h1><span class="badge">Physics, observations and termination rules stay fixed</span>
<p>The goal is explicit: beat both original Selfplay v2 and the unchanged scripted attacker. We first measured why the previous training objective could reward unhelpful behavior, then tested focused reinforcement learning and teacher distillation. Failed experiments remain part of the record.</p>
<div class="stats"><div class="stat"><b>{len(runs)}</b><small>Recorded training runs</small></div><div class="stat"><b>{total/1e6:.2f}M</b><small>New training decisions across these runs</small></div><div class="stat"><b>2 targets</b><small>Original v2 and scripted attacker</small></div><div class="stat"><b>{len(blocks)} / 3</b><small>Fresh final blocks evaluated</small></div></div>
<section class="verdict"><h2>Result of the investigation</h2><p>{verdict}</p>{candidate_text}{pooled_table}<p class="note">{escape(selection_disclosure)} Rally draws are not wins: a model can win a paired ten-rally match by taking a few points while drawing the rest. The draw rate therefore matters alongside match wins. Paired matches combine both side assignments on the same five serves.</p>{final_table}<p class="note">The interceptor is reported as a guardrail, separately from the two primary targets. Pooled counts are descriptive; acceptance is checked independently in each block.</p></section>
<section><h2>Why v3 regressed</h2><p>The original v2 actor was absent from the training and validation opponent pool. Checkpoint selection averaged five opponents equally, three of them easier baselines. This was a different objective from consistently beating both requested targets.</p>{table(['Frozen policy', 'Attacker W / L / D', 'Interceptor W / L / D', 'Original v2 W / L / D'], regression_rows)}
<p class="note">These diagnostics use {diagnosis['seed_count']} dispersed development seeds, five serves and both sides: {diagnosis['seed_count']*10:,} rallies per row/opponent. They are not the new candidate’s final tests.</p>
<h3>A positive training reward can accompany losing play</h3><p>With the old discount of .99, a point 400 ticks away carries only {100*diagnosis['discounts']['400']:.3f}% of an immediate reward’s weight. For released v3 against v2, full-horizon greedy play loses more rallies than it wins, yet its mean discounted training reward is {reward['mean_discounted_training_return']:+.6f}: point rewards contribute {reward['mean_discounted_point_reward']:+.6f}, while return bonuses contribute {reward['mean_discounted_return_bonus']:+.6f}.</p>
<p>The sampled training policy also behaves differently from the deterministic browser policy. These are measured mismatches, not proof that any single change must fix learning. Greedy loop shortcuts preserve the same point outcomes as continuing the tested greedy games to the deadline, so loop detection itself does not explain the score regression.</p>
<p>A separate league-maintenance defect switched some opponents mid-rally after update 125. The released v3 checkpoint was update 100, so that later defect cannot explain its regression. The focused trainer keeps opponent assignments fixed until each rally ends.</p></section>
<section><h2>What we tried</h2><p>The curves below show raw development measurements. The score is the weaker of the two targets’ (wins + half draws) / rallies; .5 means zero net point advantage, not satisfaction of the full competition criterion. PPO and distillation use different fixed development seed sets, so their plotted scores are not a matched-test ranking.</p><a href="improvement-curves.svg"><img src="improvement-curves.png" alt="Raw focused PPO and distillation development curves, discounted reward components, and greedy versus sampled policy outcomes"></a>
{table(['Run', 'New decisions', 'Best weaker-target score', 'Best v2 W / L / D', 'Best attacker W / L / D', 'Interpretation'], experiment_rows)}
<p class="note">Each row shows that run’s best recorded development checkpoint by weaker-target score, breaking ties by mean target score. Continued stages count only their additional decisions. Values are unsmoothed; failed variants are included. Teacher-derived policies must still fit standalone neural weights and the existing browser graph contract.</p></section>
<section><h2>Choosing one candidate before final tests</h2><p>The finalists below played the same 128 fresh development seeds, with both sides and all five serves. This comparison includes the strongest focused PPO candidate as well as two geometry-assisted target models. These data were used for selection and are not held-out evidence. Raw-input action imitation and raw-input target regression remain visible above as unsuccessful alternatives.</p>{development_table}<p>The selected method learns a target from a minimax recovery teacher while progressively collecting the learner’s own states. The feature-assisted variants add exact game geometry as fixed neural layers. Their improvement cannot be attributed to more training alone: architecture and prior knowledge also changed. Lower target-prediction error alone did not select the best competitor.</p></section>
<section><h2>What counts as consistent improvement?</h2><p>The protocol reserves three disjoint blocks of 200 dispersed seeds each before final model evaluation. Every candidate faces all five serves from both physical sides. In <strong>every block, against both targets</strong>, it must win more than half the paired ten-rally matches including draws in the denominator, have a strictly positive lower endpoint of the seed-cluster 95% point-margin interval, and win more rallies than it loses on each physical side.</p>
<p>Uncertainty is estimated by resampling whole seeds, keeping their two sides and ten rallies together. The interval is not based on treating mirrored rallies as independent samples. These empirical thresholds do not prove optimal play or provide a family-wise statistical guarantee.</p><p>Final testing uses the actual exported ONNX candidate and original v2 ONNX file, deterministic first argmax, unchanged attacker code and the existing v3 loop/timeout rules. Checkpoint selection and tuning must finish before these outcomes are seen.</p></section>
<section><h2>Reproduce and inspect</h2><p>The frozen protocol, seed lists, run records and release manifest are in <code>training/results/training-improvement/</code>. The independent diagnosis is in <code>training/results/training-diagnosis/</code>. Run <code>.venv/bin/python training/scripts/improvement_report.py</code> to rebuild this page from the recorded data. Earlier v1/v2/v3 reports and weights remain available.</p><p>{escape(verification)}</p></section>
<footer>Rally Lab · Fixed Pong v3 · Evidence from development and reserved final tests is reported separately</footer></main></html>'''
    (args.out / 'improvement.html').write_text(html)
    print(args.out / 'improvement.html')


if __name__ == '__main__':
    main()

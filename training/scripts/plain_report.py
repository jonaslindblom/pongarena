"""Render the fully learned fixed-v3 experiment from retained evidence."""
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

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ('selfplay-v2', 'attacker')
NAMES = {'selfplay-v2': 'Original Selfplay v2', 'attacker': 'Scripted attacker',
         'interceptor': 'Interceptor guardrail'}


def read(path):
    return json.loads(path.read_text())


def source(path):
    return ROOT / path


def verify_hash(path, expected, label):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'{label} differs from the frozen experiment: {path}')


def table(headers, rows):
    head = ''.join(f'<th scope="col">{escape(value)}</th>' for value in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(str(value))}</td>' for value in row) + '</tr>' for row in rows)
    return f'<div class="table-wrap" tabindex="0"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def wld(row, prefix=''):
    return ' / '.join(str(row[prefix + suffix]) for suffix in ['wins', 'losses', 'draws'])


def rate(row, prefix='match_'):
    denominator = sum(row[prefix + suffix] for suffix in ['wins', 'losses', 'draws'])
    if denominator <= 0:
        raise ValueError('Match rates require completed games, including draws.')
    return row[prefix + 'wins'] / denominator


def minimum_game_rate(row):
    return min(rate(row['opponents'][name]) for name in TARGETS)


def passes(row):
    if set(row['sides']) != {'0', '1'}:
        raise ValueError('Both physical sides must be reported.')
    return (rate(row) > .5 and rate(row, 'paired_match_') > .5
            and row['point_margin_ci95'][0] > 0
            and all(side['wins'] > side['losses'] for side in row['sides'].values()))


def plots(runs, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#536153', 'text.color': '#203c30',
                         'axes.edgecolor': '#ccd4c6', 'figure.facecolor': '#f5f6f0',
                         'axes.facecolor': '#fdfef9'})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    fig.suptitle('Rally Lab  /  Fully learned policies, fixed Pong v3 rules',
                 fontsize=19, fontweight='bold', ha='left', x=.04)
    palette = ['#397451', '#b27836', '#44728d', '#9b554e', '#8c7a9f', '#6c8562']
    for ax, family, title in zip(axes, ['ppo', 'search'], ['PPO fine-tuning', 'Direct weight search']):
        included = [run for run in runs if run['family'] == family]
        for index, run in enumerate(included):
            ax.plot([row['env_steps'] / 1e6 for row in run['rows']],
                    [minimum_game_rate(row) for row in run['rows']], label=run['label'],
                    color=palette[index % len(palette)], lw=1.7)
        ax.axhline(.5, color='#858a7b', linewidth=.9, linestyle='--')
        ax.set(title=title, xlabel='Additional policy decisions in this run (millions)',
               ylabel='Weaker target: ordinary five-rally game win rate', ylim=(0, 1))
        ax.grid(axis='y', color='#e1e5db', linewidth=.6)
        ax.set_axisbelow(True)
        if included:
            ax.legend(loc='upper left', bbox_to_anchor=(0, -.18), fontsize=8, frameon=False)
        else:
            ax.text(.5, .5, 'No recorded runs', ha='center', va='center', transform=ax.transAxes)
    fig.savefig(out / 'fully-learned-curves.png', dpi=150)
    fig.savefig(out / 'fully-learned-curves.svg')
    plt.close(fig)
    svg = out / 'fully-learned-curves.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')


def final_tables(blocks):
    block_rows, pooled_rows = [], []
    success = len(blocks) == 3
    for label, data in blocks:
        if data['rules'] != 'pong-v3':
            raise ValueError('Final results must retain Pong v3 rules.')
        if data.get('candidate_runtime') != 'native_onnxruntime' or data.get('opponent_runtimes', {}).get('selfplay-v2') != 'native_onnxruntime':
            raise ValueError('Final evidence must use actual candidate and v2 ONNX files.')
        if any(name not in data['opponents'] for name in TARGETS):
            raise ValueError('Final blocks must include both primary opponents.')
        for name, row in data['opponents'].items():
            passed = passes(row)
            if name in TARGETS:
                success = success and passed
            ci = row['point_margin_ci95']
            block_rows.append([label, NAMES.get(name, name), wld(row),
                               f'{row["loop_draws"]} / {row["timeout_draws"]}', wld(row, 'match_'),
                               f'{100*rate(row):.1f}%', wld(row, 'paired_match_'),
                               f'{100*rate(row, "paired_match_"):.1f}%',
                               ' · '.join(f'{row["sides"][side]["wins"]}/{row["sides"][side]["losses"]}' for side in ['0', '1']),
                               f'{100*ci[0]:+.1f}…{100*ci[1]:+.1f} pp',
                               ('Pass' if passed else 'Does not pass') if name in TARGETS else 'Guardrail'])
    for name in (*TARGETS, 'interceptor'):
        values = [data['opponents'][name] for _, data in blocks if name in data['opponents']]
        if not values:
            continue
        totals = {key: sum(row[key] for row in values) for key in
                  ['rallies', 'wins', 'losses', 'draws', 'match_wins', 'match_losses', 'match_draws',
                   'paired_match_wins', 'paired_match_losses', 'paired_match_draws']}
        pooled_rows.append([NAMES[name], wld(totals), f'{100*totals["draws"]/totals["rallies"]:.1f}%',
                            wld(totals, 'match_'), f'{100*rate(totals):.1f}%',
                            wld(totals, 'paired_match_'), f'{100*rate(totals, "paired_match_"):.1f}%'])
    return block_rows, pooled_rows, bool(success)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT / 'training/results/fully-learned')
    parser.add_argument('--out', type=Path, default=ROOT / 'public/training')
    args = parser.parse_args()
    manifest = read(args.results / 'manifest.json')
    protocol = read(args.results / 'protocol.json')
    if manifest['rules'] != 'pong-v3' or protocol['rules'] != 'pong-v3':
        raise ValueError('This report is for the fixed Pong v3 experiment.')
    frozen_rules = protocol['rules_unchanged']
    for path, digest in [('engine', 'sha256'), ('contract', 'contract_sha256')]:
        verify_hash(source(frozen_rules[path]), frozen_rules[digest], 'Game rules')
    reference = protocol['primary_targets']['selfplay-v2']
    verify_hash(source(reference['file']), reference['sha256'], 'Original v2 weights')
    attacker = protocol['primary_targets']['attacker']
    verify_hash(source(attacker['implementation'].split(':')[0]), attacker['source_sha256'], 'Original attacker')
    seeds_file = args.results / protocol['holdout']['file']
    verify_hash(seeds_file, protocol['holdout']['sha256'], 'Reserved seed list')
    reserved = read(seeds_file)['blocks']
    runs = []
    for item in manifest['runs']:
        directory = source(item['path'])
        rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
        if not rows:
            raise ValueError(f'No measured learning curve: {directory}')
        runs.append(dict(item, rows=rows, selected=read(directory / 'selection.json')))
    blocks = [(item['name'], read(source(item['file']))) for item in manifest.get('holdouts', [])]
    if len({name for name, _ in blocks}) != len(blocks):
        raise ValueError('Duplicate blocks are not fresh independent confirmation.')
    for name, result in blocks:
        expected = next((block for block in reserved if block['name'] == name), None)
        if expected is None or expected['seeds'] != result['seeds']:
            raise ValueError(f'{name} differs from the predeclared seeds.')
        for key in ['samples', 'seed', 'confidence', 'method']:
            if result['bootstrap'][key] != protocol['evaluation']['bootstrap'][key]:
                raise ValueError(f'{name} differs from the predeclared bootstrap protocol.')
    block_rows, pooled_rows, success = final_tables(blocks)
    candidate = manifest.get('candidate')
    candidate_text = '<p>No final export has been selected.</p>'
    if candidate:
        frozen = read(source(candidate['freeze_file']))
        for path, digest in [('checkpoint', 'checkpoint_sha256'), ('onnx', 'onnx_sha256')]:
            verify_hash(source(frozen[path]), frozen[digest], f'Selected {path}')
        verify_hash(source(candidate['model_path']), frozen['onnx_sha256'], 'Published model')
        check = read(source(candidate['check_file']))
        if (not check['graph_matches_trainable_checkpoint'] or check['fixed_feature_or_feedback_parameters'] != 0
                or check['action_agreement'] != 1
                or check['onnx_sha256'] != frozen['onnx_sha256']
                or check['checkpoint_sha256'] != frozen['checkpoint_sha256']):
            raise ValueError('Selected export lacks a matching plain-network verification.')
        stats = check['arena_validator']
        if stats['params'] != check['trainable_parameters'] or stats['widths'][0] != 6 or stats['widths'][-1] != 3:
            raise ValueError('Every submitted parameter must belong to the six-input/three-score learned MLP.')
        for name, result in blocks:
            verify_hash(source(result['model']), frozen['onnx_sha256'], f'{name} evaluated candidate')
            if result.get('model_sha256') != frozen['onnx_sha256']:
                raise ValueError(f'{name} records a different model hash.')
        candidate_text = f'<p><strong>{escape(candidate["name"])}</strong> · {escape(candidate["architecture"].rstrip("."))}. {escape(candidate["selection_note"])}</p>'
        candidate_text += f'<p class="note">{stats["params"]:,} learned parameters · {stats["macs"]:,} MACs per decision · {stats["bytes"]:,} bytes · {stats["nodes"]} ONNX operators. The serialized graph exactly matches the trainable MLP checkpoint. Fixed geometry and paddle-feedback parameters: zero.</p>'
        if candidate['model_path'].startswith('public/'):
            candidate_text += f'<p><a class="download" href="/{escape(candidate["model_path"][7:], quote=True)}" download>Download standalone ONNX weights</a></p>'
    elif blocks:
        raise ValueError('Final results require a selected frozen candidate.')
    development_rows, development_seeds = [], None
    for item in manifest.get('development', []):
        result = read(source(item['file']))
        if development_seeds is None:
            development_seeds = result['seeds']
        elif result['seeds'] != development_seeds:
            raise ValueError('Shared development comparisons must use identical seeds.')
        for name in (*TARGETS, 'interceptor'):
            if name not in result['opponents']:
                continue
            row = result['opponents'][name]
            development_rows.append([item['name'], NAMES[name], wld(row), wld(row, 'match_'),
                                     f'{100*rate(row):.1f}%', wld(row, 'paired_match_')])
    experiment_rows = []
    for run in runs:
        selected = run['selected']
        position = (f'update {selected["update"]}' if 'update' in selected else f'generation {selected["generation"]}')
        experiment_rows.append([run['label'], f'{run["rows"][-1]["env_steps"]:,}', position,
                                f'{100*minimum_game_rate(selected):.1f}%',
                                wld(selected['opponents']['selfplay-v2'], 'match_'),
                                wld(selected['opponents']['attacker'], 'match_'), run.get('note', '')])
    verdict = ('The frozen fully learned export passes every predeclared criterion against both targets in all three reserved blocks.' if success else
               'The recorded results do not satisfy the complete predeclared criterion.' if blocks else
               'Final testing is not recorded yet. Development improvements alone do not establish consistent held-out wins.')
    final_summary = table(['Opponent', 'Rally W / L / D', 'Rally draws', 'Five-rally games W / L / D',
                           'Game win rate', 'Paired contests W / L / D', 'Paired win rate'], pooled_rows) if blocks else ''
    final_details = table(['Block', 'Opponent', 'Rally W / L / D', 'Loop / timeout draws', 'Five-rally games W / L / D',
                           'Game wins', 'Paired contests W / L / D', 'Paired wins', 'Left W/L · right W/L',
                           'Point-margin 95% interval', 'Protocol'], block_rows) if blocks else ''
    testing_note = ''
    if blocks:
        tested = manifest['final_candidates_tested']
        testing_note = ('One frozen candidate was tested on these reserved blocks.' if tested == 1 else
                        f'{tested} candidates were tested on these reserved blocks; reuse limits their independence from selection.')
        testing_note += ' The three blocks test the same chosen export; they are not three independent training runs.'
    total = sum(run['rows'][-1]['env_steps'] for run in runs)
    verification = manifest.get('verification_note', 'Final verification is pending.')
    args.out.mkdir(parents=True, exist_ok=True)
    plots(runs, args.out)
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rally Lab · Fully learned training</title><link rel="icon" href="/mark.svg">
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f6f0;color:#203c30;font:15px/1.65 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:42px 24px 64px}}nav{{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:46px}}a{{color:#376e43}}h1{{font-size:clamp(36px,6vw,64px);letter-spacing:-.055em;line-height:1.08;max-width:850px;margin:12px 0 22px}}h2{{font-size:26px;letter-spacing:-.03em;margin:0 0 16px}}p{{color:#5a6b5b;max-width:940px}}.eyebrow{{font:11px monospace;color:#75935c;letter-spacing:.14em}}.badge{{display:inline-block;background:#e4ecdc;color:#45633f;border-radius:20px;padding:6px 12px;font-size:12px}}.stats{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:28px 0}}.stat,section{{padding:23px;background:#fdfef9;border:1px solid #dce3d5;border-radius:12px;min-width:0}}.stat b{{display:block;font-size:30px;letter-spacing:-.04em}}.stat small{{font-size:11px;color:#70816c}}section{{margin-top:20px}}img{{width:100%;height:auto;border-radius:8px}}.table-wrap{{max-width:100%;overflow-x:auto}}table{{border-collapse:collapse;width:100%;text-align:left;font-size:13px}}th,td{{padding:11px 10px;border-bottom:1px solid #e2e7dc;white-space:nowrap}}th{{color:#70816c;font-weight:500}}.note{{font-size:13px}}code{{background:#edf1e7;border-radius:4px;padding:2px 5px;overflow-wrap:anywhere}}.download{{display:inline-block;padding:9px 15px;border:1px solid #bfd0b7;border-radius:7px;text-decoration:none}}.verdict{{border-left:4px solid #789461}}footer{{font-size:12px;color:#75816d;margin-top:30px}}@media(max-width:620px){{main{{padding:26px 14px 44px}}section{{padding:17px}}.stats{{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}.stat{{padding:15px}}.stat b{{font-size:25px}}nav{{margin-bottom:30px}}}}
</style><main><nav><a href="/">← Back to the arena</a><a href="report.html">Original v3 training ↗</a><a href="improvement.html">Earlier hybrid investigation ↗</a></nav>
<div class="eyebrow">PONG V3 · FULLY LEARNED POLICIES</div><h1>Six inputs.<br>Learned decisions.</h1><span class="badge">Same rules · No fixed trajectory calculations</span>
<p>The goal is a plain learned network that beats both original Selfplay v2 and the unchanged scripted attacker. Every affine weight and bias belongs to the trained policy. The submitted actor receives the six public observations and emits three action scores directly.</p>
<div class="stats"><div class="stat"><b>{len(runs)}</b><small>Recorded training experiments</small></div><div class="stat"><b>{total/1e6:.2f}M</b><small>Additional policy decisions across these runs</small></div><div class="stat"><b>2 targets</b><small>Original v2 and scripted attacker</small></div><div class="stat"><b>{len(blocks)} / 3</b><small>Reserved final blocks evaluated</small></div></div>
<section class="verdict"><h2>Result</h2><p>{verdict}</p>{candidate_text}{final_summary}<p class="note">Drawn games count in every win-rate denominator. Ordinary games are the arena’s five-rally matches; paired contests combine both side assignments on the same five serves. These are two aggregations of the same trajectories, not extra independent samples. {escape(testing_note)}</p>{final_details}<p class="note">Acceptance is checked in every block against each primary target. Pooled totals are descriptive. The interceptor is a separately reported guardrail.</p></section>
<section><h2>What “fully learned” means here</h2><p>The policy is a conventional feed-forward Tanh MLP: learned affine layers and nonlinearities, with no engineered flight-time or landing-height features, fixed paddle-error feedback, opponent identifier, timer, planner or external action rule. Its exported graph is checked against the selected trainable checkpoint and the existing arena upload budget.</p><p>The starting weights follow this lineage: original learned v2 actor → terminal-only focused PPO → undiscounted focused PPO → the current training experiments. Original v2 used a supervised warm start followed by PPO. The current candidates inherit learned weights; they do not inherit the hybrid Strategist’s geometry network. The training-only critic may use opponent identity and time to estimate returns; those inputs never enter the exported policy.</p><p>The earlier hybrid Strategist remains an archived experiment. It included fixed trajectory calculations and does not satisfy this experiment’s fully learned requirement. Its results are not counted as success here. The <a href="improvement.html">earlier report</a> retains the regression diagnosis and its separate evidence.</p></section>
<section><h2>Learning curves and unsuccessful runs</h2><p>These are raw development measurements of the weaker target’s ordinary five-rally game win rate. The dashed line is 50%, counting draws in the denominator. Crossing it on development data alone is not a final success claim.</p><a href="fully-learned-curves.svg"><img src="fully-learned-curves.png" alt="Raw development game-win curves for fully learned PPO fine-tuning and direct weight search"></a>
{table(['Run', 'Additional decisions', 'Selected checkpoint', 'Weaker game win rate', 'v2 games W / L / D', 'Attacker games W / L / D', 'Interpretation'], experiment_rows)}
<p class="note">Each table row uses its recorded selected checkpoint. PPO decision counts include critic warm-up. Search counts aggregate active decisions across tested population members. Counts cover completed logged checkpoints; they exclude earlier pretraining and any interrupted work after the last measurement, and are not elapsed-time comparisons. Failed and regressing runs remain visible.</p><p>PPO uses terminal point rewards, fixed opponents within each rally, conservative actor updates and a training-only value estimator. Deterministic weight search tests changes to learned MLP parameters against the actual greedy-deployment objective. Neither route adds a fixed geometric computation to inference. Sampled PPO training disables loop exits because a sampled action can escape a repeated position; deterministic evaluation uses the unchanged v3 loop rule.</p></section>
<section><h2>Shared development comparison</h2><p>The starting policy and candidate checkpoints below use the same {len(development_seeds) if development_seeds else 0} development seeds, all five serves and both sides. These results inform selection; they are not the reserved final tests. Selection targets v2 and the attacker. The interceptor is shown separately to expose specialization: a model can gain against both targets while turning wins against a passive defender into draws. Targeted improvement is not a claim of being universally stronger.</p>
{table(['Policy', 'Opponent', 'Rally W / L / D', 'Five-rally games W / L / D', 'Game win rate', 'Paired contests W / L / D'], development_rows) if development_rows else '<p>No shared comparison has been recorded.</p>'}</section>
<section><h2>A stricter consistency test</h2><p>The final protocol was reserved before evaluating a selected export. In <strong>each of three fresh blocks, against both primary opponents</strong>, it requires: more than half of ordinary five-rally games won; more than half of paired ten-rally contests won; a positive lower endpoint for the 95% seed-cluster point-margin interval; and more rally wins than losses separately on both physical sides. Draws remain in the game-win denominators.</p><p>Each block contains 200 dispersed seeds, 2,000 rallies per opponent and 400 ordinary games. Bootstrap resampling keeps both sides and all ten rallies for a seed together. These empirical criteria do not prove optimal play or give a family-wise statistical guarantee. Final testing runs the actual exported candidate and original v2 ONNX file through native ONNX Runtime, with deterministic first argmax. Published final seeds become historical evidence for future work.</p></section>
<section><h2>Reproduce and inspect</h2><p>Read <code>training/results/fully-learned/README.md</code> for exact training and evaluation commands. Run records, protocol, reserved seed lists, selection evidence and verification are retained under <code>training/results/fully-learned/</code>. Rebuild this report with <code>.venv/bin/python training/scripts/plain_report.py</code>. Earlier rules packages and model reports remain archived.</p><p>{escape(verification)}</p></section>
<footer>Rally Lab · Fixed Pong v3 · Fully learned inference · Development and final evidence reported separately</footer></main></html>'''
    (args.out / 'fully-learned.html').write_text(html)
    print(args.out / 'fully-learned.html')


if __name__ == '__main__':
    main()

"""Render the bounded burst-Pong model-size experiment without rerunning tests."""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from train_size_study import ROOT


def main():
    base = ROOT/'training/results/burst-study/models'
    release = base/'release'
    manifest = json.loads((release/'manifest.json').read_text())
    final = json.loads((release/'confirmation.json').read_text())
    assert final['complete']
    verification = json.loads((release/'verification.json').read_text())
    curves = {identity:[json.loads(line) for line in (base/'screen'/identity/'metrics.jsonl').read_text().splitlines()]
              for identity in manifest['models']}
    names = list(manifest['selected'])
    colors = ['#9c9270', '#84a45e', '#47886d', '#d38a4c', '#7a6eac']
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
        'axes.spines.right':False, 'figure.facecolor':'#fdfef9', 'axes.facecolor':'#fdfef9',
        'text.color':'#203c30', 'axes.labelcolor':'#516755', 'xtick.color':'#687c67', 'ytick.color':'#687c67'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for name, color in zip(names, colors):
        identities = [identity for identity, info in manifest['models'].items() if info['name'] == name]
        rounds = [row['round'] for row in curves[identities[0]]]
        scores = np.array([[row['development_margin'] for row in curves[identity]] for identity in identities])
        axes[0,0].plot(rounds, scores.mean(0), marker='o', ms=3, color=color, label=name.title())
        axes[0,0].fill_between(rounds, scores.min(0), scores.max(0), color=color, alpha=.10)
        chosen = manifest['selected'][name]
        params = manifest['models'][chosen]['params']
        tracker_rates = [final['models'][identity]['tracker']['summary']['score_rate']*100 for identity in identities]
        axes[0,1].plot([params]*len(tracker_rates), tracker_rates, 'o', color=color, ms=7)
        axes[0,1].plot([params, params], [min(tracker_rates), max(tracker_rates)], color=color)
        mirror = final['models'][chosen]['mirror']['summary']
        axes[1,0].bar(name.title(), mirror['median_decisive_returns'] or 0, color=color, width=.6)
        axes[1,0].text(names.index(name), (mirror['median_decisive_returns'] or 0)+.5,
                       f'{mirror["draws"]/mirror["rallies"]:.0%} draws', ha='center', fontsize=9)
        actual = final['models'][chosen]['interceptor']['summary']['score_rate']
        if name in ('tiny', 'small', 'large'):
            ablated = final['ablations'][name+':interceptor']['summary']['score_rate']
            index = ['tiny', 'small', 'large'].index(name)
            axes[1,1].bar(index-.17, actual*100, .32, color=color)
            axes[1,1].bar(index+.17, ablated*100, .32, color=color, alpha=.35, hatch='//')
    axes[0,0].set(title='Development score · identical data', xlabel='Training rounds', ylabel='Mean point margin')
    axes[0,0].set_ylim(-1.05, .15)
    axes[0,0].legend(frameon=False, ncol=2, fontsize=8, loc='lower right')
    axes[0,1].set(title='Fresh tests versus Tracker', xlabel='Learned parameters (log scale)', ylabel='Wins + half draws (%)')
    axes[0,1].set_xscale('log')
    axes[0,1].set_ylim(0, 100)
    axes[0,1].axhline(50, color='#9cae93', lw=.8)
    axes[1,0].set(title='Selected networks playing themselves', ylabel='Median returns in scored rallies')
    axes[1,0].set_ylim(0, 26)
    axes[1,1].set(title='Does learned burst use help?', ylabel='Score versus interceptor (%)')
    axes[1,1].set_xticks(range(3), ['Tiny', 'Small', 'Large'])
    axes[1,1].set_ylim(0, 100)
    axes[1,1].text(.02, .96, 'Solid: original policy\nHatched: burst mapped to ordinary movement',
                   transform=axes[1,1].transAxes, va='top', fontsize=8)
    for axis in axes.flat:
        axis.grid(axis='y', alpha=.10)
        axis.set_axisbelow(True)
    fig.suptitle('Small weights can play; tactical mastery is a separate test', fontweight='bold', x=.06, ha='left', fontsize=16)
    fig.tight_layout(rect=[0, .04, 1, .95])
    fig.text(.5, .025, 'Two initializations per size. Shared DAgger data and epoch budgets. No curve smoothing. Fresh tests: 32 paired seeds × five serves.',
             ha='center', fontsize=9, color='#687c67')
    for extension in ('png', 'svg'):
        path = base/f'size-curves.{extension}'
        fig.savefig(path, dpi=160)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        shutil.copyfile(path, ROOT/f'public/training/burst-size-curves.{extension}')
    plt.close(fig)
    def wld(row):
        return f'{row["wins"]} / {row["losses"]} / {row["draws"]}'
    rows = []
    for name in names:
        identity = manifest['selected'][name]
        info = manifest['models'][identity]
        result = final['models'][identity]
        mirror = result['mirror']['summary']
        rows.append(f'<tr><td><a href="/models/burst-{name}.onnx">{name.title()} ↓</a></td><td>{info["params"]:,}</td>'
            f'<td>{" → ".join(map(str, info["architecture"]))}</td><td>{wld(result["tracker"]["summary"])}</td>'
            f'<td>{wld(result["attack"]["summary"])}</td><td>{mirror["median_decisive_returns"]:g}</td><td>{mirror["draws"]/mirror["rallies"]:.1%}</td></tr>')
    cross = ''.join(f'<tr><td>{key.replace(":", " vs ").title()}</td><td>{wld(value["summary"])}</td>'
                    f'<td>{value["summary"]["median_decisive_returns"]}</td></tr>' for key, value in final['crossplay'].items())
    metrics = dict(protocol=manifest, learning_curves=curves,
        final_summaries={identity:{name:row['summary'] for name,row in rows.items()} for identity,rows in final['models'].items()},
        crossplay={key:row['summary'] for key,row in final['crossplay'].items()},
        ablations={key:row['summary'] for key,row in final['ablations'].items()}, verification=verification)
    followup_html = ''
    followup_path = base/'followup-release'
    if (followup_path/'confirmation.json').exists():
        focused = json.loads((followup_path/'confirmation.json').read_text())
        focus_protocol = json.loads((followup_path/'manifest.json').read_text())
        focus_curves = [json.loads(line) for line in (base/'followup/large-s641/metrics.jsonl').read_text().splitlines()]
        metrics['separate_followup'] = dict(protocol=focus_protocol, curves=focus_curves,
            summaries={name:row['summary'] for name,row in focused['pairs'].items()},
            tactic_probes=json.loads((followup_path/'tactic-probes.json').read_text()))
        fig, axes = plt.subplots(1,2,figsize=(12,3.8))
        for name,color in [('tracker','#84a45e'),('interceptor','#47886d'),('attack','#d38a4c')]:
            axes[0].plot([r['round'] for r in focus_curves], [r['opponents'][name]['point_margin'] for r in focus_curves],
                         marker='o',color=color,label=name.title())
            axes[1].plot([r['round'] for r in focus_curves], [100*r['opponents'][name]['point_rate'] for r in focus_curves],
                         marker='o',color=color,label=name.title())
        for axis in axes:
            axis.axvline(focus_protocol['selection']['round'],color='#98a88c',ls=':',lw=1)
            axis.grid(axis='y',alpha=.1)
            axis.set_xlabel('Additional training rounds')
        axes[0].set(title='Separate burst-focused follow-up',ylabel='Development point margin',ylim=(-1.05,1.05))
        axes[1].set(title='Development wins by opponent',ylabel='Rallies won (%)',ylim=(0,100))
        axes[0].legend(frameon=False,fontsize=9)
        fig.tight_layout()
        fig.savefig(base/'focus-curves.png',dpi=160)
        shutil.copyfile(base/'focus-curves.png',ROOT/'public/training/burst-focus-curves.png')
        plt.close(fig)
        show = [('focus:attack','Focus vs attacker'),('baseline:attack','Original Large vs attacker'),
                ('no_burst:attack','Focus with burst disabled vs attacker'),('focus:baseline','Focus vs original Large'),
                ('focus:safe','Focus vs safer defender'),('baseline:safe','Original Large vs safer defender')]
        frows=''.join(f'<tr><td>{label}</td><td>{wld(focused["pairs"][key]["summary"])}</td></tr>' for key,label in show)
        followup_html='''<section><h2>A separate test: learning to use burst</h2><p>One bounded follow-up continued the 18,309-parameter Large model for 60 additional rounds. Actual reachable game states where the teacher requested burst were oversampled. The selected checkpoint was round 48, chosen on a new development block; final tests used another 32 fresh seeds. This has a larger training budget and is deliberately excluded from the five-size comparison.</p><img src="burst-focus-curves.png" alt="Separate follow-up development curves showing improved attacking-opponent wins, with round48 selected before final testing"><p><strong>Learned burst use now has a measurable benefit.</strong> Focus wins 44 rallies against the attacker where the original Large wins none on the same seeds. Disabling Focus's burst actions removes all 44 wins. It remains much weaker than the attacker, and loses badly to the original Large and to the safer defender. This is a different specialist, not a universally stronger replacement.</p><div class="table-wrap"><table><thead><tr><th>Same fresh test seeds</th><th>Wins / losses / draws</th></tr></thead><tbody>'''+frows+'''</tbody></table></div><p>On the rule study's exact final-shot scenario, Focus catches the ball when given a full energy meter; all five original size-study models miss it. With the recorded depleted meter, the shot is mathematically unreachable and Focus also misses. This supports learned rescue timing. It still fails to choose the earlier placement counter that can avoid the trap, so the complete multi-exchange tactic remains unlearned.</p><p><a href="/models/burst-focus.onnx">Focus weights ↓</a> · Same plain 8→128→128→5 Tanh architecture · <a href="burst-focus-replays.json">Complete native replay fixtures</a></p></section>'''
    (ROOT/'public/training/burst-model-metrics.json').write_text(json.dumps(metrics, separators=(',', ':'))+'\n')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rally Lab · How small can a paddle be?</title><link rel="icon" href="/mark.svg">
<style>*{box-sizing:border-box}body{margin:0;background:#f5f6f0;color:#203c30;font:15px/1.65 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:36px 24px 64px}nav{display:flex;justify-content:space-between;gap:16px;margin-bottom:40px}a{color:#376e43}h1{font-size:clamp(36px,6vw,64px);line-height:1.05;letter-spacing:-.05em;margin:16px 0 24px}h2{font-size:25px;line-height:1.2;letter-spacing:-.02em}p{color:#5a6b5b;max-width:960px}.eyebrow{font:11px monospace;letter-spacing:.14em;color:#75935c}.notice{display:inline-block;font-size:12px;background:#e5eddc;padding:6px 12px;border-radius:24px}section{background:#fdfef9;border:1px solid #dce3d5;border-radius:12px;padding:24px;margin-top:24px;min-width:0}img{display:block;width:100%;height:auto}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12px;text-align:left}th,td{padding:12px 10px;border-bottom:1px solid #e2e7dc;white-space:nowrap}th{color:#70816c;font-weight:500}code,pre{background:#edf1e7;border-radius:4px}code{padding:2px 5px}pre{padding:16px;overflow:auto;font-size:12px}pre code{padding:0}.note,footer{font-size:12px;color:#70816c}footer{margin-top:28px}@media(max-width:650px){main{padding:24px 14px}section{padding:18px}}</style>
<main><nav><a href="/">← Arena</a><a href="burst-model-metrics.json">Raw measurements ↓</a></nav><div class="eyebrow">RALLY LAB / MODEL SIZE STUDY</div><h1>How small can<br>a paddle be?</h1><span class="notice">Burst prototype · bounded training experiment · two initialization seeds per size</span>
<p>Networks with 501 learned parameters can already beat a simple ball-following paddle. Larger networks improve prediction and survival. This experiment does <strong>not</strong> demonstrate tactical mastery: every selected network still loses every held-out rally to the resource-aware attacking teacher.</p>
<section><h2>Same data, different capacity</h2><p>Five plain Tanh architectures, each trained from two initializations. Every actor received the same 983,040 labels, exact minibatch orders and four epochs per round, over 24 rounds. The shared data came from 393,216 environment steps in pooled DAgger, both player perspectives, plus broad synthetic observations. Synthetic inputs are not claimed reachable. Geometric teacher calculations exist only in training; the exported networks have eight raw inputs and five action scores.</p><p>Weights were selected on eight development seeds, then frozen before 32 new paired test seeds. Training budgets are equal in examples and optimization passes, not FLOPs: larger models do more arithmetic. This is the smallest useful size demonstrated by this particular recipe, not a proof of the minimum possible network.</p><img src="burst-size-curves.png" alt="Learning curves for five model sizes, two initialization results versus Tracker, mirror rally lengths and draw rates, and burst-action ablations"><p class="note">Development bands span the two initializations. Mirror lengths include scored rallies only; draw rates are shown separately. Score is wins plus half draws. Numerical diagnostics do not establish subjective human enjoyment.</p></section>
<section><h2>Frozen selected weights</h2><p>Each row is 320 rallies: 32 seeds, five serves, both side assignments. The table uses frozen PyTorch checkpoints in the verified vector environment. Native ONNX and browser replay checks are separate. All use prototype configuration <code>burst96-cost8-half</code>.</p><div class="table-wrap"><table><thead><tr><th>Network</th><th>Parameters</th><th>Architecture</th><th>Tracker W / L / D</th><th>Attacker W / L / D</th><th>Mirror median returns</th><th>Mirror draws</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<p><strong>229 parameters:</strong> weak in this run, often missing early returns. <strong>501–1,509:</strong> a plausible beginner-opponent range, with clear weaknesses. <strong>5,061–18,309:</strong> better survival, but high mirror draw rates and continued losses to attacking scripts. Model size alone is not a difficulty level, and longer rallies do not prove better attacking tactics.</p><p>All models use fully learned weights. They require the burst contract, <code>observation[1,8]</code> and <code>action_scores[1,5]</code>; they are not compatible with the older six-input/three-output Pong files. These are provisional game rules, not a permanently frozen competition season.</p></section>
<section><h2>Size versus size</h2><p>All checkpoints were chosen before these tests. Rows are from the first model's perspective. These results describe this fixed opponent pool, not a universal ordering across every possible uploaded policy.</p><div class="table-wrap"><table><thead><tr><th>Pair</th><th>Wins / losses / draws</th><th>Median decisive returns</th></tr></thead><tbody>'''+cross+'''</tbody></table></div></section>
<section><h2>Can they learn the resource tactic?</h2><p>The current evidence is limited. Mapping learned burst actions to ordinary movement measures whether those actions help match outcomes. Changing an energy input and observing a changed action merely shows sensitivity; it is not proof that the new action is useful. Both results are retained in the raw data.</p><p>The greedy training teacher has weak opponent-energy dependence: at a fixed outgoing speed, some energy-dependent reach terms subtract the same value from every candidate shot. It does not provide a reliable multi-exchange planning target. Larger networks can learn some burst timing, but this first size screen has not taught the full pressure → depleted recovery → finishing-shot tactic. A separate bounded follow-up investigates rare reachable burst decisions without changing this comparison.</p></section>
<section><h2>Reproduce and inspect</h2><p>The complete protocol, all ten checkpoint selections, raw curves, test outcomes, model hashes and native parity probes are retained under <code>training/results/burst-study/models/</code>. Training games use seeds below one billion; development uses [2.81,2.82) billion, final tests [2.9,3.0) billion, and native replay fixtures use separate seeds above 3.01 billion. These final tests are consumed; future tuning needs new reserved seeds.</p><pre><code>.venv/bin/python training/scripts/train_size_study.py --help
.venv/bin/python training/scripts/report_size_study.py
.venv/bin/python training/scripts/verify_size_models.py</code></pre><p>The trained exports pass numeric parity on 23,376 observations with no action disagreements. Six full native ONNX matches are available for browser verification. The training method, limitations and exact reproduction commands are in the retained models README.</p></section><footer>Rally Lab · <a href="burst-model-metrics.json">Full measurements</a> · <a href="burst-onnx-replays.json">Native replay fixtures</a> · <a href="/">Watch or play →</a></footer></main></html>'''
    html=html.replace('<section><h2>Reproduce and inspect</h2>',followup_html+'<section><h2>Reproduce and inspect</h2>')
    (ROOT/'public/training/burst-models.html').write_text(html)
    print(ROOT/'public/training/burst-models.html')


if __name__ == '__main__':
    main()

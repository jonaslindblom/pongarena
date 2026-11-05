"""Render the retained Classic placement experiment; never train or select weights."""
import html
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/rally-classic-tactics-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT/'training/results/classic-tactics'
PUBLIC = ROOT/'public/training'


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return '<div class="table-wrap" tabindex="0"><table><thead><tr>'+''.join(
        '<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def court(key, label):
    return f'''<div><h3>{label}</h3><svg id="court-{key}" class="court" viewBox="0 0 360 160" role="img" aria-label="{label}">
<path d="M180 4V156" stroke="#688779" stroke-dasharray="3 4"/><rect x="12" width="6" height="30" fill="#87e8b8" data-left/>
<rect x="342" width="6" height="30" fill="#f6ac70" data-right/><circle r="4" fill="#f3ffd6" data-ball/></svg>
<div class="readout"><span id="returns-{key}"></span><span id="state-{key}"></span></div></div>'''


def plot_curves(logs, names, prefix, title):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'figure.facecolor':'#fdfef9','axes.facecolor':'#fdfef9',
        'text.color':'#203c30','axes.labelcolor':'#203c30','xtick.color':'#5a6b5b','ytick.color':'#5a6b5b'})
    fig, axes = plt.subplots(2,2,figsize=(12,7.6),layout='constrained')
    for name, color in [('control','#78966a'),('tactical','#b57646')]:
        rows = logs[name]
        x = [row['round'] for row in rows]
        loss = [row for row in rows if row['cross_entropy'] is not None]
        axes[0,0].plot([row['round'] for row in loss],[row['cross_entropy'] for row in loss],'.-',color=color,label=names[name])
        axes[0,1].plot(x,[row['development_margin'] for row in rows],'.-',color=color,label=names[name])
        for i, opponent in enumerate(('attack','defend')):
            axes[1,i].plot(x,[row['opponents'][opponent]['point_margin'] for row in rows],'.-',color=color,label=names[name])
    for ax,axis_title in zip(axes.flat,['Training label loss','Development selection metric','Development vs attacker','Development vs defender']):
        ax.set_title(axis_title,loc='left',fontweight='bold',pad=12)
        ax.set_xlabel('Training round');ax.grid(alpha=.15)
    axes[0,0].set_ylabel('Mean cross entropy')
    axes[0,0].legend(frameon=False,fontsize=8)
    for ax in (axes[0,1],axes[1,0],axes[1,1]):
        ax.set_ylabel('(wins − losses) / all rallies')
        ax.set_ylim(-1,1);ax.axhline(0,color='#aab2a3',lw=.8)
    fig.suptitle(title,fontweight='bold',x=.02,ha='left')
    fig.supxlabel('Unsmoothed development evaluations. Checkpoint selection uses attacker + defender; final seeds are separate.',fontsize=9)
    for suffix in ('png','svg'):
        path = STUDY/f'{prefix}-curves.{suffix}'
        fig.savefig(path,dpi=160)
        if suffix == 'svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        shutil.copyfile(path,PUBLIC/f'{prefix}-curves.{suffix}')
    plt.close(fig)


def main():
    train, final_probes = (read(STUDY/'probes'/f'{phase}.json') for phase in ('train', 'final'))
    evaluation = STUDY/'learning/evaluation'
    final = read(evaluation/'confirmation.json')
    assert final['complete']
    manifest = read(evaluation/'manifest.json')
    run = ROOT/manifest['source']
    config = read(run/'config.json')
    verification = read(STUDY/'verification.json')
    probes = read(evaluation/'tactic-probes.json')
    breadth = read(STUDY/'probes/breadth.json')
    examples = [read(STUDY/'probes'/name) for name in ('representative-broad.json','representative.json')]
    logs = {name:[json.loads(line) for line in (run/name/'metrics.jsonl').read_text().splitlines()]
            for name in ('control', 'tactical')}
    first_logs = {name:[json.loads(line) for line in (STUDY/'learning/run'/name/'metrics.jsonl').read_text().splitlines()]
                  for name in ('control','tactical')}
    names = dict(control='Original-label control', tactical='Tactical-label treatment', pacer='Unchanged Pacer',
                 attack='Adaptive attacker', defend='Adaptive defender', **{'control-last':'Trained control · final round'})
    plot_curves(logs,names,'classic-tactics','Restrained updates with an anchor to the original Pacer')
    plot_curves(first_logs,names,'classic-tactics-first','First attempt: larger tactical updates lose overall strength')
    probe_rows = [[phase,len(data['seeds']),data['summary']['snapshots'],data['summary']['branches'],
                   data['summary']['states_with_better_outcome'],data['summary']['loss_to_win_states']]
                  for phase,data in [('Training discovery',train),('Held-out probes',final_probes)]]
    match_rows = []
    for pair, record in final['pairs'].items():
        a,b = pair.split(':'); s = record['summary']
        match_rows.append([names[a],names[b],f'{s["wins"]} / {s["losses"]} / {s["draws"]}',
                           s['median_decisive_returns'] if s['median_decisive_returns'] is not None else '—'])
    gains = []
    for key,row in final['paired_improvements'].items():
        comparison,opponent = key.split(':')
        a,b = comparison.split('-minus-')
        lo,hi = row['ci95']
        gains.append([names[a]+' − '+names[b],names[opponent],f'{row["mean_point_margin_gain"]:+.3f}',f'{lo:+.3f} to {hi:+.3f}'])
    local_rows = []
    for name in ('pacer','tactical','control-last'):
        record = probes['models'][name]
        summary = record['summary']
        local_rows.append([names[name],f'{summary["strict_opportunities_won"]} / {summary["strict_loss_to_win_opportunities"]}',
            f'{100*record["path_label_agreement"]["all"]["accuracy"]:.2f}%'])
    local_html = '<section><h2>Imitation accuracy is not tactical success</h2><p>On fresh probe seeds, the model controls the receiving paddle until its next contact. The original scripts then resume, isolating the placement choice. The opportunity count includes only positions where the scripted baseline loses and a legal alternative wins.</p>'+table(
        ['Model','Winning placements found / available','Agreement with selected path actions'],local_rows)+'''<p>The trained network agrees with more action labels, but finds one fewer winning placement than unchanged Pacer. There are often several valid action paths to a good contact, and a label sequence is optimized for the scripted continuation. A network playing the rest of the rally can make different choices.</p><p class="note">Fresh seeds do not guarantee unseen states: these scripted trajectories often converge. State-overlap and separate seen/unseen probe summaries are retained in the measurements. These correlated probes are not independent match trials.</p></section>'''
    output = dict(rules='pong-v4',protocol=manifest,config=config,learning_curves=logs,first_attempt_curves=first_logs,
        probe_summaries={phase:data['summary'] for phase,data in [('train',train),('final',final_probes)]},
        full_match_summaries={pair:record['summary'] for pair,record in final['pairs'].items()},
        paired_improvements=final['paired_improvements'],verification=verification,
        tactical_probe_overlap=probes['overlap'],
        tactical_probe_summaries={name:{key:record[key] for key in ('summary','start_partitions','path_label_agreement')}
                                 for name,record in probes['models'].items()},
        placement_breadth={phase:{key:value for key,value in record.items() if key not in ('rows','broad_mid_rally_candidates')}
                          for phase,record in breadth['phases'].items()})
    (PUBLIC/'classic-tactics-metrics.json').write_text(json.dumps(output,indent=2)+'\n')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rally Lab · Can a different return win?</title><link rel="icon" href="/mark.svg">
<style>*{box-sizing:border-box}body{margin:0;background:#f5f6f0;color:#203c30;font:15px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:32px 22px 60px}nav{display:flex;justify-content:space-between;gap:14px;margin-bottom:36px}a{color:#326b43}h1{font-size:clamp(34px,6vw,62px);line-height:1.07;letter-spacing:-.045em;max-width:850px}h2{font-size:25px;line-height:1.2}h3{font-size:15px;margin-bottom:8px}p{max-width:940px;color:#576b5d}section{background:#fdfff9;border:1px solid #dce3d6;border-radius:12px;padding:24px;margin-top:22px;min-width:0}.tag{font:11px monospace;letter-spacing:.1em;color:#6c875c}.note,footer{font-size:12px;color:#61745e}.court-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.court{display:block;background:#17382d;width:100%;aspect-ratio:2.25}.readout{font:12px monospace;display:flex;justify-content:space-between;gap:8px;margin-top:7px;flex-wrap:wrap}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:18px 0}button,select{min-height:42px;border:1px solid #bdcdb9;background:#f0f5e9;color:#203c30;border-radius:6px;padding:8px 13px;font:inherit;cursor:pointer}input[type=range]{width:100%;min-height:36px;accent-color:#386f4d}.table-wrap{overflow-x:auto;max-width:100%}table{border-collapse:collapse;min-width:580px;width:100%;font-size:12px}th,td{text-align:left;padding:11px;border-bottom:1px solid #e0e6d9}th{color:#61745e;font-weight:500}code{background:#ebf0e3;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}pre{white-space:pre-wrap;font-size:12px;background:#edf2e7;padding:14px;border-radius:6px}img{display:block;width:100%;height:auto}.callout{border-left:4px solid #789661;padding-left:16px}.badge{display:inline-block;background:#e7eedf;padding:5px 10px;border-radius:20px;font-size:12px}@media(max-width:650px){main{padding:24px 14px 40px}section{padding:17px}.court-grid{grid-template-columns:1fr}nav{margin-bottom:26px}h2{font-size:22px}}</style>
<main><nav><a href="/">← Arena</a><a href="pong-v4.zip">Frozen rules package ↓</a></nav><div class="tag">RALLY LAB / CLASSIC V4</div><h1>Can a different<br>return win?</h1><span class="badge">Unchanged rules · legal alternatives · controlled learning test</span>
<p>First, test whether a strong script leaves a better reachable shot available. Then test whether a plain neural network learns those choices and improves against opponents in complete matches. These are separate claims.</p>
<section><h2>One reachable choice, two continuations</h2><p id="example-description"></p><div class="court-grid">__COURTS__</div><div class="controls"><button id="play">Play</button><button id="branch">Changed decision</button><button id="ending">Outcomes</button><label>Speed <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><span id="tick" class="tag"></span></div><label for="timeline" class="note">Recorded simulation tick</label><input id="timeline" type="range" min="0" step="1"><p class="note">The panes share their entire prefix. The alternative executes ordinary up/stay/down actions until the selected contact, then the same scripts resume. Local probes use the original 1,600-tick deadline without the deterministic-loop shortcut, because this one-off intervention changes the controller. Full network matches below use normal loop detection.</p><p class="callout">This demonstrates an exploitable choice by this script, under this continuation. It is not a proof of a forced win against every possible defense.</p></section>
<section><h2>All alternatives count</h2>__PROBE_TABLE__<p class="note">All unique flight-start physical states in the specified script matches are inspected, including states without an improvement. Every exactly reachable legal contact center is enumerated. These are conditional opportunities, not independent match win rates. Training examples select one loss-to-win path per qualifying state; held-out seeds never provide training labels.</p></section>
<section><h2>Change the labels, hold the budget fixed</h2><p>Both arms start from the identical Pacer weights: <strong>6 → 128 → 128 → 3</strong>, 17,795 learned parameters. Both see the same pooled observations, minibatches and optimization budget. The control uses the original adaptive attacker's labels; treatment substitutes successful branch actions on the oversampled tactical rows. No trajectory calculations or scripted decisions are embedded in either export.</p><p>Checkpoint selection uses development performance against the attacker and defender. Loss and development curves below are not final-test results. A winning action path is only one way to reach its contact; exact action imitation is not the same as learning useful placement.</p><a href="classic-tactics-curves.svg"><img src="classic-tactics-curves.png" alt="Controlled learning curves for original-teacher and tactical-placement labels"></a></section>
__LOCAL_RESULTS__
<section><h2>Complete matches on fresh serves</h2><p>Each pairing uses __FINAL_COUNT__ fresh seeds, all five serves and both side assignments. The models were frozen before these tests. Scores use the released arena rules, including exact-loop draws.</p>__MATCH_TABLE__<p class="note">W / L / D is from the first model's perspective. Median returns includes scored rallies only. Results use frozen PyTorch weights in the verified vector engine; ONNX and browser parity are implementation checks performed separately.</p></section>
<section><h2>Did the tactical labels improve play?</h2>__GAIN_TABLE__<p class="note">Point margin is (wins − losses) / all rallies. Each difference compares the exact same serve seeds. The 95% intervals resample whole seeds, retaining both sides and all five rallies; they are not corrected for multiple comparisons. A positive local result does not override a full-match regression.</p>__CONCLUSION__</section>
<section><h2>Reproduce the test</h2><p>The retained experiment contains the complete action prefixes and every candidate branch, shared training data, raw logs, selected checkpoints, source hashes and fresh evaluation seeds. The independent scalar checker verifies the legal transitions and terminal outcomes. Published final seeds are now consumed; future selection requires new seeds.</p><pre>.venv/bin/python training/scripts/check_classic_tactics.py
.venv/bin/python training/scripts/report_classic_tactics.py</pre><p>Exact discovery, training, freezing and evaluation commands are in <code>training/results/classic-tactics/README.md</code>. A failed continuation is evidence about this training method and budget, not a proof that a network cannot beat the scripts. This experiment does not test human enjoyment or change the game rules.</p></section><footer>Rally Lab · <a href="classic-tactics-metrics.json">Measurements and protocol</a> · <a href="/">Arena</a></footer></main>
<script>
const examples=__EXAMPLE__;let data=examples[0];
const range=document.getElementById('timeline'),play=document.getElementById('play'),speed=document.getElementById('speed');
let at=Math.max(0,data.start_tick-20),playing=false,last=0,debt=0;
let end=Math.max(data.baseline.end_tick,data.alternative.end_tick);range.max=end;
const result=outcome=>outcome===2?'Draw':outcome===data.side?'Controlled player wins':'Controlled player loses';
const description=document.getElementById('example-description');
const label=document.createElement('label');label.textContent='Example ';const selector=document.createElement('select');selector.setAttribute('aria-label','Replay example');['A wider winning placement range','A one-unit contact change'].forEach((text,i)=>{const o=document.createElement('option');o.value=i;o.textContent=text;selector.append(o);});label.append(selector);description.before(label);
const selection=document.createElement('p');selection.className='note';description.after(selection);
function describe(){description.textContent=`Seed ${data.seed}, rally ${data.rally}. The ${data.side===0?'left':'right'} paddle changes its contact center from Y=${data.baseline_contact} to Y=${data.alternative_contact}. ${result(data.baseline.outcome)} with the original continuation; ${result(data.alternative.outcome).toLowerCase()} with the alternative.`;selection.textContent=data.selection+(data.winning_centers?` Winning contact centers in this continuation: ${data.winning_centers.join(', ')}.`:'');}
selector.addEventListener('change',()=>{pause();data=examples[Number(selector.value)];end=Math.max(data.baseline.end_tick,data.alternative.end_tick);range.max=end;at=Math.max(0,data.start_tick-20);describe();draw();});describe();
function pane(key,record){const frames=record.frames,f=frames[Math.min(frames.length-1,Math.max(0,at-frames[0].tick))],svg=document.getElementById('court-'+key);svg.querySelector('[data-left]').setAttribute('y',f.paddles[0]-15);svg.querySelector('[data-right]').setAttribute('y',f.paddles[1]-15);const ball=svg.querySelector('[data-ball]');ball.setAttribute('cx',f.ball[0]);ball.setAttribute('cy',f.ball[1]);document.getElementById('returns-'+key).textContent=f.returns+' returns';document.getElementById('state-'+key).textContent=at>=record.end_tick?result(record.outcome):'In play';}
function draw(){range.value=at;document.getElementById('tick').textContent='TICK '+at;pane('a',data.baseline);pane('b',data.alternative);}
function pause(){playing=false;play.textContent='Play';}
function seek(t){pause();at=t;draw();}
range.addEventListener('input',()=>seek(Number(range.value)));document.getElementById('branch').addEventListener('click',()=>seek(data.start_tick));document.getElementById('ending').addEventListener('click',()=>seek(end));
play.addEventListener('click',()=>{if(playing){pause();return;}if(at>=end)at=Math.max(0,data.start_tick-20);playing=true;play.textContent='Pause';last=performance.now();debt=0;requestAnimationFrame(tick);});
function tick(now){if(!playing)return;debt+=Math.min(100,now-last)*Number(speed.value);last=now;while(debt>=1000/30&&at<end){at++;debt-=1000/30;}draw();if(at>=end)pause();else requestAnimationFrame(tick);}
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});draw();
</script></html>'''
    replacements = {
        '__COURTS__':court('a','Original scripted choice')+court('b','Alternative contact placement'),
        '__PROBE_TABLE__':table(['Phase','Seeds','Flight states','Branches incl. baseline','Better outcome available','Loss → win available'],probe_rows),
        '__MATCH_TABLE__':table(['Model','Opponent','W / L / D','Median scored returns'],match_rows),
        '__GAIN_TABLE__':table(['Comparison','Opponent','Point-margin gain','95% paired interval'],gains),
        '__FINAL_COUNT__':str(len(final['seeds'])),
        '__EXAMPLE__':json.dumps(examples,separators=(',',':')),
        '__LOCAL_RESULTS__':local_html,
        '__CONCLUSION__':'<p><strong>The scripts leave exploitable choices; this training method did not produce an overall stronger player.</strong> The selected treatment improves against the defender but regresses clearly against the attacker. Its head-to-head difference against Pacer is uncertain. The arena keeps its existing models.</p>',
    }
    page = page.replace('<p>First, test whether', '<p class="callout"><strong>Result: legal placement improvements exist, but the trained continuation is not an overall upgrade.</strong> It improves against the defender and regresses against the attacker.</p><p>First, test whether')
    page = page.replace('All unique flight-start physical states', 'On the final set, 269 of 410 loss-to-win opportunities have at least three winning contact centers. All unique flight-start physical states')
    page = page.replace('Checkpoint selection uses development performance', 'The first 24-round attempt regressed in both arms. A single follow-up reduced the learning rate and tactical-row frequency and anchored outputs to the original Pacer. Treatment selected round 44 of 48; control selected round 0, so its primary export is unchanged Pacer. The trained control at round 48 is a secondary diagnostic. Checkpoint selection uses development performance')
    page = page.replace('</a></section>\n__LOCAL_RESULTS__', '</a><details><summary>First attempt curves</summary><a href="classic-tactics-first-curves.svg"><img src="classic-tactics-first-curves.png" alt="First attempt learning curves show regression from the initial Pacer in both arms"></a></details></section>\n__LOCAL_RESULTS__')
    for marker,value in replacements.items():page=page.replace(marker,value)
    (PUBLIC/'classic-tactics.html').write_text(page+'\n')
    print('Published classic-tactics.html, learning curves and measurements')


if __name__ == '__main__':
    main()

"""Publish the retained gradual-speed study and raw DAgger learning curves."""
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/rally-pace-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT/'training/results/pace-study'
TRAIN = STUDY/'training'
PUBLIC = ROOT/'public/training'


def read(path):
    return json.loads(path.read_text())


def pct(value):
    return '—' if value is None else f'{100*value:.1f}%'


def number(value):
    return '—' if value is None else f'{value:g}'


def wld(row):
    return f'{row["wins"]} / {row["losses"]} / {row["draws"]}'


def table(headers, rows):
    return '<div class="table-wrap" tabindex="0"><table><thead><tr>'+''.join(f'<th>{x}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    rule = read(STUDY/'confirmation.json')
    final = read(TRAIN/'confirmation.json')
    assert final['complete']
    verification = read(TRAIN/'verification.json')
    logs = []
    for name, offset in [('dagger-s411',0),('dagger-continuation-s412',24)]:
        rows = [json.loads(line) for line in (TRAIN/name/'metrics.jsonl').read_text().splitlines()]
        logs.append([dict(run=name,total_round=row['round']+offset,**row) for row in rows])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'figure.facecolor':'#fdfef9','axes.facecolor':'#fdfef9',
        'text.color':'#203c30','axes.labelcolor':'#203c30','xtick.color':'#5a6b5b','ytick.color':'#5a6b5b'})
    fig, axes = plt.subplots(2,2,figsize=(12,7.4),layout='constrained')
    colors = {'attack':'#b57646','defend':'#9475aa','contender':'#397451','tracker':'#7b985a'}
    for rows in logs:
        x = [r['total_round'] for r in rows]
        measured = [r for r in rows if r['cross_entropy'] is not None]
        axes[0,0].plot([r['total_round'] for r in measured],[r['cross_entropy'] for r in measured],'.-',color='#397451')
        axes[0,1].plot(x,[r['development_margin'] for r in rows],'.-',color='#203c30')
        for name,color in colors.items():
            axes[1,0].plot(x,[r['opponents'][name]['point_margin'] for r in rows],'.-',color=color,label=name if rows is logs[0] else None)
            axes[1,1].plot(x,[100*r['opponents'][name]['point_rate'] for r in rows],'.-',color=color)
    labels = [('Teacher imitation loss','Mean cross entropy'),('Development selection metric','Mean point margin'),
        ('Development performance by opponent','(wins − losses) / all rallies'),('Development wins by opponent','Wins / all rallies (%)')]
    for ax,(title,ylabel) in zip(axes.flat,labels):
        ax.set_title(title,loc='left',fontweight='bold',pad=12)
        ax.set_xlabel('Total training rounds');ax.set_ylabel(ylabel)
        ax.grid(alpha=.14);ax.axvline(24,color='#aab2a3',linestyle=':',linewidth=1.3)
        ax.set_xlim(0,60)
    for ax in [axes[0,1],axes[1,0]]:ax.axhline(0,color='#aab2a3',linewidth=.8);ax.set_ylim(-1.05,1.05)
    axes[1,1].set_ylim(0,100)
    axes[1,0].legend(ncol=2,fontsize=9,frameon=False,loc='lower right')
    fig.suptitle('Pacer v4 · learned from raw six-input observations',fontsize=17,fontweight='bold',x=.02,ha='left')
    fig.supxlabel('Dots are retained evaluations, every 3 rounds. Dotted line: continuation with a larger replay buffer; no curve smoothing.',fontsize=9,color='#5a6b5b')
    for suffix in ('png','svg'):
        target=TRAIN/f'learning-curves.{suffix}'
        fig.savefig(target,dpi=160)
        if suffix=='svg':target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
        shutil.copyfile(target,PUBLIC/f'pace-learning-curves.{suffix}')
    plt.close(fig)
    clean = lambda row:{k:v for k,v in row.items() if k not in ('per_seed','sides')}
    data = dict(rule_seeds=rule['seeds'],model_seeds=final['seeds'],
        development_seeds=read(TRAIN/'dagger-s411/config.json')['development_seeds'],
        rule_confirmation={name:{pair:clean(r['summary']) for pair,r in pairs.items()} for name,pairs in rule['configs'].items()},
        model_confirmation={pair:clean(r['summary']) for pair,r in final['pairs'].items()},
        learning_curves=[[dict(**{k:v for k,v in row.items() if k!='opponents'},
            opponents={name:clean(r) for name,r in row['opponents'].items()}) for row in rows] for rows in logs],
        export_verification=verification)
    (PUBLIC/'pace-metrics.json').write_text(json.dumps(data,indent=2)+'\n')
    rule_rows=[]
    for name,label in [('baseline','Previous v3'),('pace13-v9','Gradual speed, unscaled angle'),('pace13-v9-scaled','Pong v4 · selected')]:
        pairs=rule['configs'][name];s=pairs['attack:attack']['summary'];d=pairs['defend:defend']['summary']
        rule_rows.append([label,pct(1-s['draws']/s['rallies']),number(s['median_decisive_returns']),pct(s['first_receiver_win_rate']),wld(pairs['attack:defend']['summary']),pct(d['first_receiver_win_rate'])])
    names={'pacer':'Pacer v4','apprentice':'Apprentice v4','attack':'Adaptive attacker','defend':'Adaptive defender',
        'contender':'Contender (old weights)','v2':'Selfplay v2 (old weights)','tracker':'Tracker','anticipator':'Anticipator'}
    model_rows=[]
    for pair,record in final['pairs'].items():
        a,b=pair.split(':');s=record['summary'];lo,hi=s['point_margin_ci95']
        model_rows.append([names[a],names[b],wld(s),number(s['median_decisive_returns']),f'{lo:+.3f} to {hi:+.3f}'])
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rally Lab · Give the rally room to grow</title><link rel="icon" href="/mark.svg">
<style>*{box-sizing:border-box}body{margin:0;background:#f5f6f0;color:#203c30;font:15px/1.65 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:36px 24px 64px}nav{display:flex;justify-content:space-between;gap:16px;margin-bottom:44px}a{color:#376e43}h1{font-size:clamp(36px,6vw,66px);line-height:1.04;letter-spacing:-.055em;max-width:880px;margin:12px 0 22px}h2{font-size:26px;line-height:1.2;letter-spacing:-.025em;margin:0 0 16px}p{color:#5a6b5b;max-width:930px}.eyebrow{font:11px monospace;letter-spacing:.14em;color:#75935c}.notice{display:inline-block;font-size:12px;color:#466445;background:#e5eddc;padding:6px 12px;border-radius:30px}section{background:#fdfef9;border:1px solid #dce3d5;border-radius:12px;padding:24px;margin-top:22px;min-width:0}.table-wrap{overflow-x:auto;max-width:100%}table{border-collapse:collapse;width:100%;font-size:12px;text-align:left}th,td{padding:12px 10px;border-bottom:1px solid #e2e7dc;white-space:nowrap}th{font-weight:500;color:#70816c}.note,footer{font-size:12px;color:#70816c}img{display:block;width:100%;height:auto}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:25px}.stat{background:#eaf0e2;border-radius:10px;padding:18px}.stat strong{display:block;font-size:29px;line-height:1.2}.stat span{font-size:12px;color:#64765f}.conclusion{border-left:4px solid #789461}code{background:#edf1e7;border-radius:4px;padding:2px 5px;overflow-wrap:anywhere}pre{background:#edf1e7;padding:16px;overflow:auto;font-size:12px;border-radius:8px}pre code{padding:0}footer{margin-top:30px}summary{cursor:pointer;color:#376e43;font-weight:600}@media(max-width:650px){main{padding:26px 14px 44px}section{padding:18px}.stats{grid-template-columns:1fr}nav{margin-bottom:30px}h2{font-size:23px}}</style>
<main><nav><a href="/">← Arena</a><a href="pong-v4.zip">Training package ↓</a></nav><div class="eyebrow">RALLY LAB / PONG V4</div><h1>Give the rally<br>room to grow.</h1><span class="notice">Rules tuned · new learned weights · separate confirmation sets</span>
<p>Start gently, build speed with each return, and use paddle placement to move the opponent. Pong v4 makes sustained exchanges possible before pressure builds. Pacer and Apprentice are new plain neural networks trained for these rules.</p>
<div class="stats"><div class="stat"><strong>1.83 s</strong><span>First return, across the court</span></div><div class="stat"><strong>11 returns</strong><span>Median, strong attacker mirror</span></div><div class="stat"><strong>17,795</strong><span>Learned parameters per new model</span></div></div>
<section class="conclusion"><h2>A practical step toward tactical rallies.</h2><p>On 64 reserved seeds, strong attacker mirrors score every rally after 7–19 returns, with a median of 11. The attacker beats the safer defender 384–119, with 137 draws. In all three strong-script pairings, every scoring return is already unreachable from the receiver’s position when the ball leaves the opposing paddle. Placement and recovery have consequences.</p><p>These scripts are reference controllers, not optimal players. First-receiver advantage remains: 60.3% in attacker mirrors and 71.6% in defender mirrors. Side-paired evaluation remains necessary. Actual keyboard and touch playtesting is still needed to assess enjoyment and accessibility.</p></section>
<section><h2>One return rule. No hidden counter.</h2><p>The serve has horizontal speed 5. Every successful return adds 1, capped at 13, regardless of where it hits the paddle. Higher and lower contacts determine its vertical direction. The angle scales with horizontal speed, so early edge shots stay readable. A center hit no longer resets the pace.</p><pre><code>d = ball_y − updated_paddle_y
s = min(13, abs(incoming_vx) + 1)
vx = outgoing_sign × s
vy = floor(9 × s × d / 195 + 0.5)</code></pre><p>The first return has horizontal speed 6 and maximum vertical speed 4. It takes eight returns to reach the cap; the fastest crossing takes 26 ticks (0.87 seconds). Paddle speed remains 2 units per tick. All state stays integer-valued, with the same six observations and three actions. Velocity inputs remain divided by 6, without clipping; current velocity already exposes the speed stage.</p><p class="note">Court, collision order, seeded serves, 1,600-tick deadline and exact post-return loop detection retain their established definitions. Full definitions and Python/TypeScript references are in the <a href="pong-v4.zip">standalone v4 package</a>.</p></section>
<section><h2>Same serves, stronger reference players</h2>'''+table(['Rules','Attacker mirror scored','Median returns¹','First receiver wins¹','Attack vs defense W/L/D','Defender mirror: first receiver¹'],rule_rows)+'''<p class="note">Each cell summarizes 64 seeds × five serves × both assignments = 640 rallies. ¹ Among scored rallies. Both speed candidates use cap 13 and vertical factor 9; the unscaled candidate applies the full vertical factor even at the first return. Mirrors necessarily tie in aggregate when both roles are swapped.</p><p>The search screened caps 11–14 and vertical factors 5–9, then compared scaled factors 8 and 9, using 32 diagnostic seeds. The selected settings were locked before this 64-seed confirmation. No strong-script pairing under the selected rules scores before seven returns. Attacker mirrors last a median of 13.67 seconds.</p><p>A stage-aware safety calculation still proves never-concede play for some milder candidates. It does not establish a universally safe defense for v4. That failed sufficient test is not proof that optimal attack always wins.</p></section>
<section><h2>Learning from six numbers</h2><p>Both new models use <strong>6 → 128 → 128 → 3</strong>, with Tanh between learned linear layers: 17,795 parameters, 17,536 multiply-accumulates, five ONNX nodes and 71,658 bytes. Inference contains learned weights only. The three output scores choose up, stay or down by first argmax.</p><p>Training uses PyTorch DAgger: a geometric attacker labels states encountered by the learner, then cross entropy teaches the network to reproduce those choices. The teacher is used only during training. Opponents include stronger scripts, older weights and simpler baselines. Broad synthetic component-valid states supplement game trajectories; they are not assumed to be reachable.</p><p>Pacer uses 24 initial rounds plus 36 continuation rounds: 3,932,160 active environment steps, with a larger replay buffer in the continuation. Apprentice is the initial run’s round-six checkpoint, chosen as an easier opponent. Checkpoints were selected on development seeds before either model saw the final set. This is supervised imitation with dataset aggregation; no PPO run is claimed.</p><a href="pace-learning-curves.svg"><img src="pace-learning-curves.png" alt="Unsmoothed Pacer learning curves: imitation loss, mean development margin, per-opponent margin and win rate across 60 training rounds"></a><p class="note">Loss and accuracy are training measurements. Match curves use 32 development seeds, five serves and both assignments. Dots show retained evaluations every three rounds; lines connect observations, with no smoothing. <a href="pace-metrics.json">Download the plotted values, final metrics and seed lists</a>.</p></section>
<section><h2>What the new weights can actually beat</h2>'''+table(['Model','Opponent','W / L / D','Median returns¹','Point margin · 95% CI'],model_rows)+'''<p class="note">Separate model confirmation: 64 untouched seeds, 640 rallies per pairing, using frozen PyTorch checkpoints in the verified vector environment. Native and browser ONNX inference were checked separately against those weights; this table is not an all-ONNX run. ¹ Among scored rallies. Point margin is (wins − losses) / all rallies. Confidence intervals resample whole paired seeds. Older weights are evaluated under v4 rules, so these are transfer comparisons, not their original leaderboards.</p><p>Pacer clearly improves on older weights under the new rules, and Pacer mirrors sustain a median of 11 returns. It still loses to the adaptive attacker and defender. These results establish a playable learned baseline, not a training ceiling or a claim to have beaten optimal scripted play.</p><p><a href="/models/pacer-v4.onnx">Pacer v4 weights ↓</a> · <a href="/models/apprentice-v4.onnx">Apprentice v4 weights ↓</a> · <a href="/">Watch or play in the arena →</a></p></section>
<section><h2>Reproducible, with limits visible</h2><p>Rule verification compares 45,493 independent scalar/vector transitions and 1,082,322 exhaustive contact transitions. Fifteen complete action fixtures agree with the independent v4 reference. Both exports pass the arena validator and native ONNX checks; 8,192 numeric probes per model preserve every PyTorch action. Browser WASM parity also passed all six complete native-ONNX matches: 10,904 frames and 21,760 decisions, with zero action or state mismatches and maximum score difference 0.00001145.</p><p>Training game seeds are below 1 billion. Development seeds are in [1.9, 2.0) billion, rule screening in [2.3, 2.4), rule confirmation in [2.4, 2.5), and final model confirmation in [2.6, 2.7). Exact lists, frozen hashes, checkpoints, scripts and raw logs live in <code>training/results/pace-study/</code>.</p><details><summary>Rebuild the report and verification</summary><pre><code>.venv/bin/python training/results/pace-study/verify.py
.venv/bin/python training/scripts/verify_pace_models.py
.venv/bin/python training/scripts/pace_report.py</code></pre><p>Full training commands and the exact historical source snapshots are documented in <code>training/results/pace-study/training/README.md</code>. The frozen final evaluation intentionally refuses to overwrite its results. New tuning requires new reserved seeds.</p></details><p class="note">A delayed-observation controller was a timing sensitivity check, not a human simulation. Numerical success does not establish subjective fun. Persistent serve advantage and the learned models’ losses to strong scripts remain useful targets for future evaluation.</p></section><footer>Rally Lab · Gradual-speed rules and fully learned v4 baselines · <a href="pace-metrics.json">Measurements</a> · <a href="pace-onnx-replays.json">Native ONNX parity fixtures</a></footer></main></html>'''
    (PUBLIC/'pace-study.html').write_text(html+'\n')
    print('Published pace-study.html, curves and raw metrics')


if __name__ == '__main__':
    main()

"""Publish the completed return-law comparison and synchronized recorded replays."""
from html import escape
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'training/results/offense-study'
OUT = ROOT / 'public/training'
LABELS = {'baseline': 'Current rules', 'angle8': 'Steeper returns · 8',
          'angle9': 'Steeper returns · 9', 'motion2': 'Moving-paddle deflection'}


def pct(value):
    return '—' if value is None else f'{value:.1%}'


def table(headers, rows):
    return '<div class="table-wrap" tabindex="0"><table><thead><tr>' + ''.join(
        f'<th>{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def main():
    final = json.loads((STUDY / 'evaluation/confirmation.json').read_text())
    assert final.get('complete'), 'Publish only a complete frozen comparison.'
    invariant = json.loads((STUDY / 'verification/invariant.json').read_text())
    training = STUDY / 'training/search'
    strong, learned, detail, replay = [], [], [], {}
    total = 0
    for name, pairs in final['configs'].items():
        aa, ad = [pairs[p]['summary'] for p in ('attack:attack', 'attack:defend')]
        n = aa['rallies']
        strong.append([LABELS[name], pct((aa['wins'] + aa['losses']) / n),
                       aa['median_decisive_returns'] if aa['median_decisive_returns'] is not None else '—',
                       pct(aa['first_receiver_win_rate']), f"{ad['wins']} / {ad['losses']} / {ad['draws']}",
                       'Proven' if invariant['configs'][name]['proves_never_concede_from_all_standard_serves'] else 'Not established'])
        for opponent in ('attack', 'defend'):
            before = pairs[f'initial:{opponent}']['summary']
            after = pairs[f'trained:{opponent}']['summary']
            learned.append([LABELS[name], 'Adaptive attacker' if opponent == 'attack' else 'Adaptive defender',
                            f"{before['wins']} / {before['losses']} / {before['draws']}",
                            f"{after['wins']} / {after['losses']} / {after['draws']}",
                            f"{after['point_margin_ci95'][0]:.3f} to {after['point_margin_ci95'][1]:.3f}"])
        for pair, result in pairs.items():
            s = result['summary']
            detail.append([LABELS[name], pair, f"{s['wins']} / {s['losses']} / {s['draws']}",
                           pct(s['early_finish_rate']), pct(s['long_scored_rate']),
                           s['loop_draws'], s['timeout_draws']])
        replay[name] = {pair: pairs[pair]['replays'] for pair in ('trained:attack', 'attack:defend')}
        total += json.loads((training / name / 'completed.json').read_text())['training_decisions']
    OUT.mkdir(exist_ok=True)
    for ext in ('png', 'svg'):
        shutil.copyfile(training / f'learning-curves.{ext}', OUT / f'offense-learning-curves.{ext}')
    # Compact arrays keep the browser artifact small; these are recorded states,
    # not a second simulator or a different inference path.
    for pairs in replay.values():
        for matches in pairs.values():
            for match in matches:
                match['frames'] = [[f['tick'], *f['ball'], *f['paddles'], f['returns']] for f in match['frames']]
    (OUT / 'offense-replays.json').write_text(json.dumps(replay, separators=(',', ':')) + '\n')
    html = TEMPLATE.replace('__STRONG__', table(
        ['Rules', 'Attack mirror: scored', 'Median returns¹', 'First receiver wins¹', 'Attack vs defense W/L/D', 'Never-concede defense'], strong))
    html = html.replace('__LEARNED__', table(
        ['Rules', 'Opponent', 'Starting weights W/L/D', 'Retrained weights W/L/D', 'Retrained point margin · 95% CI'], learned))
    html = html.replace('__DETAIL__', table(
        ['Rules', 'Pair', 'W / L / D', 'Before 2 returns²', 'Scores after 6+ returns²', 'Loop draws', 'Timeout draws'], detail))
    html = html.replace('__SEEDS__', str(final['seed_count'])).replace('__RALLIES__', str(final['seed_count'] * 10))
    html = html.replace('__DECISIONS__', f'{total:,}').replace('__SEED__', str(final['seeds'][0]))
    (OUT / 'offense-study.html').write_text(html)
    print(OUT / 'offense-study.html')


TEMPLATE = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rally Lab · The shape of a good rally</title><link rel="icon" href="/mark.svg">
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f6f0;color:#203c30;font:15px/1.65 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:36px 24px 64px}nav{display:flex;justify-content:space-between;gap:16px;margin-bottom:44px}a{color:#376e43}h1{font-size:clamp(36px,6vw,66px);line-height:1.04;letter-spacing:-.055em;max-width:800px;margin:12px 0 22px}h2{font-size:26px;line-height:1.2;letter-spacing:-.025em;margin:0 0 16px}p{color:#5a6b5b;max-width:900px}.eyebrow{font:11px monospace;letter-spacing:.14em;color:#75935c}.notice{display:inline-block;font-size:12px;color:#466445;background:#e5eddc;padding:6px 12px;border-radius:30px}section{background:#fdfef9;border:1px solid #dce3d5;border-radius:12px;padding:24px;margin-top:22px;min-width:0}.table-wrap{overflow-x:auto;max-width:100%}table{border-collapse:collapse;width:100%;font-size:12px;text-align:left}th,td{padding:12px 10px;border-bottom:1px solid #e2e7dc;white-space:nowrap}th{font-weight:500;color:#70816c}.note,footer{font-size:12px;color:#70816c}img{width:100%;height:auto}.courts{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:20px}.court{background:#eef2e8;padding:16px;border-radius:8px;min-width:0}.court h3{font-size:14px;font-weight:600;margin:0 0 10px}.court p{font-size:12px;margin:5px 0 0}canvas{width:100%;aspect-ratio:360/160;display:block;background:#e4ebdc;border-radius:4px}.controls{display:flex;flex-wrap:wrap;align-items:end;gap:10px}.controls label{font-size:12px;color:#64765f;display:flex;flex-direction:column;gap:4px}button,select{font:inherit;color:#203c30;border:1px solid #c6d2be;border-radius:6px;background:#f8faf4;padding:8px 11px}button{cursor:pointer}button:hover{background:#e5eddc}button:disabled{opacity:.45;cursor:default}input[type=range]{width:100%;accent-color:#397451;margin-top:16px}.progress{display:flex;justify-content:space-between;gap:12px;font:12px monospace;color:#63745b}.conclusion{border-left:4px solid #789461}code{background:#edf1e7;border-radius:4px;padding:2px 5px;overflow-wrap:anywhere}summary{cursor:pointer;font-weight:600;color:#376e43}footer{margin-top:30px}ul{padding-left:22px;color:#5a6b5b}@media(max-width:650px){main{padding:26px 14px 44px}section{padding:18px}.courts{grid-template-columns:1fr}.controls{gap:8px}nav{margin-bottom:30px}}
</style><main><nav><a href="/">← Arena</a><a href="fully-learned.html">Contender training ↗</a></nav>
<div class="eyebrow">RALLY LAB / OFFENSIVE BALANCE</div><h1>A different bounce.<br>A better game?</h1><span class="notice">Experimental comparison · live Pong v3 unchanged</span>
<p>Four return laws, the same serves, stronger scripts and equally budgeted learned-weight training. The goal is meaningful placement and recovery over sustained exchanges.</p>
<section class="conclusion"><h2>No candidate earns a rules change yet.</h2><p>The milder angle change still allows a guaranteed never-concede defense. The steeper-angle and moving-paddle candidates create some points against adaptive scripts, but their script mirror matches lean toward short rallies and first-receiver advantage. This is evidence against promoting these exact settings, not a claim that all Pong variants lack depth.</p><p>All four preserve horizontal return speeds 5–13, paddle speed 2, the same five serves and the 1,600-tick limit. The shortest cross-court flight remains 26 ticks (0.87 seconds). Real keyboard playability has not been tested.</p></section>
<section><h2>Watch the same opening under four rules</h2><p>Recorded evaluation trajectories. All panels use seed <code>__SEED__</code>, the first reserved seed, and the selected rally. The left player is the first named controller. Ended rallies stay visible while the others continue.</p>
<div class="controls"><label>Players<select id="pair"><option value="trained:attack">Retrained model vs adaptive attacker</option><option value="attack:defend">Adaptive attacker vs adaptive defender</option></select></label><label>Serve<select id="rally"><option value="0">1</option><option value="1">2</option><option value="2">3</option><option value="3">4</option><option value="4">5</option></select></label><label>Speed<select id="speed"><option value="1">1×</option><option value="4" selected>4×</option><option value="8">8×</option></select></label><button id="play" disabled>Play</button><button id="step" disabled>Next tick</button><button id="restart" disabled>Restart</button></div>
<div class="courts" id="courts"></div><input id="scrub" type="range" aria-label="Replay tick" min="0" max="1600" value="0" disabled><div class="progress"><span id="clock">Loading recorded matches…</span><span>30 ticks / second</span></div><p class="note">The model in each panel is a separately adapted 6 → 64 → 64 → 3 Tanh network. These experimental weights are not new arena presets. Replays visualize measured states; no inference is being performed by this page.</p></section>
<section><h2>Can stronger scripts create useful exchanges?</h2>__STRONG__<p class="note">__SEEDS__ held-out seeds × five rallies × both side assignments = __RALLIES__ rallies per row. ¹ Among scored rallies only. W/L/D is from the attacker’s perspective. A mirror match gives both players the same policy; its side-paired game score is necessarily tied, but individual rallies can still be scored. “Not established” means this defensive proof failed to cover every serve—not that optimal attack is proven to win.</p><p>The angle-8 defense chooses a safe return and recovers into a verified safe region. This repairs the apparent weakness found by inspecting arbitrary returns. Its guarantee covers every standard serve and every legal opponent action sequence, with exact discrete movement and clamping.</p></section>
<section><h2>A useful difference at intermediate skill</h2><p>In a separate diagnostic test, moving-paddle deflection lets the attacker beat simple interception in 314 of 320 rallies, with a median of 15 returns among scored rallies. Against a defender that deliberately chooses safer returns, it gets 12 wins, 2 losses and 306 draws on those same diagnostic seeds. Return selection matters, but the strong mirror results above still expose early serve bias.</p><p class="note">This paragraph uses the 32-seed exploratory set, not the 64-seed final set. The interceptor predicts incoming landing and recovers sensibly; it does not optimize its outgoing return.</p></section>
<section><h2>What the learned networks actually gained</h2><p>Each candidate starts from the same fully learned Contender and receives 20 search generations, nine population members, identical training/development seeds and three opponents: adaptive attack, adaptive defense and frozen Contender. All __DECISIONS__ active training decisions are recorded. Different rally lengths mean equal search opportunities, not identical decision counts.</p>__LEARNED__<p class="note">Positive point margin favors the learned model. Intervals resample whole paired seeds, not individual rallies. These four short adaptation runs test learnability; they do not establish a training ceiling or reproduce full self-play training from scratch.</p><a href="offense-learning-curves.svg"><img src="offense-learning-curves.png" alt="Development learning curves for four candidate rules across twenty generations of plain neural weight search"></a><p class="note">Curves use development seeds. Final table uses a separate reserved set. The mean scoring curve includes frozen Contender; these gains do not establish stronger attack against the scripts. The model remains a plain network with 4,803 learned parameters: no trajectory calculations or scripted decisions are embedded in its inference graph.</p></section>
<section><h2>How far is this from normal Pong?</h2><p>Angle-only returns preserve the familiar rule: hit higher or lower on the paddle to change the outgoing direction. The experiment merely makes those directions steeper. In a published interview, Nolan Bushnell describes this placement tradeoff as central to Pong. <a href="https://www.raspberrypi.com/news/the-history-of-pong-code-the-classics/">Pong design interview ↗</a></p><p>Moving-paddle deflection adds one stroke mechanic: the paddle’s actual direction at contact adds −2, 0 or +2 vertical units per tick. There is no curved flight, extra button, persistent spin or hidden state. It is a modest Pong variant, but it adds a timing demand that angle-only play avoids.</p><p>Our bounded paddle speed, five seeded rallies and loop/timeout draws are already custom competition choices. Original Pong used knob controls, and Al Alcorn describes increasing horizontal speed in three steps as a rally continued. A rally speed increase would therefore be historically grounded, though a reproducible model arena would need to expose any new speed-stage state. <a href="https://mads.si.edu/mads/id/NMAH-AC1498_Transcript_AllanAlcorn">Alcorn’s Smithsonian oral history ↗</a></p></section>
<section><h2>Methods and limits</h2><p>The candidates change only <code>vy = floor(V × offset / 15 + 0.5) + K × actual_movement_direction</code>: current (V=5,K=0), angle 8, angle 9, or motion (V=5,K=2). Python and an independent TypeScript implementation agree on explicit edge cases and complete rally fixtures. The existing arena engine is unchanged.</p><p>Adaptive scripts enumerate feasible paddle contact positions and final movement, then plan placement and recovery using each rule’s physics. They are stronger reference opponents, not optimal game solvers. A five-tick delayed-observation defender is included only as a timing sensitivity test; it is not a substitute for a human playtest. Loop exits are disabled for that stateful controller.</p><details><summary>All matchups and draw reasons</summary>__DETAIL__<p class="note">² Share of all rallies. Initial = original Contender; trained = the selected experimental adaptation; v2 = the frozen original Selfplay v2. A transfer loss by an old model is not itself evidence of better rules.</p></details><p>Source, frozen hashes, seed lists, learning logs, weights, fixtures and every point result are under <code>training/results/offense-study/</code>. Rebuild with <code>.venv/bin/python training/scripts/offense_report.py</code>.</p></section><footer>Rally Lab · Controlled return-law study · No published rules or arena presets changed</footer></main>
<script>
const labels={baseline:'Current rules',angle8:'Steeper returns · 8',angle9:'Steeper returns · 9',motion2:'Moving-paddle deflection'};
const $=id=>document.getElementById(id);let data=null,tick=0,playing=false,last=0,carry=0;
for(const [name,label] of Object.entries(labels)){const card=document.createElement('div');card.className='court';card.innerHTML='<h3>'+label+'</h3><canvas width="720" height="320" aria-label="'+label+' recorded match"></canvas><p></p>';$('courts').appendChild(card);card.dataset.name=name;}
function matches(){return Object.fromEntries(Object.keys(labels).map(name=>[name,data[name][$('pair').value][+$('rally').value]]));}
function stop(){playing=false;$('play').textContent='Play';}
function draw(){if(!data)return;const ms=matches();for(const card of $('courts').children){const m=ms[card.dataset.name],i=Math.min(tick,m.frames.length-1),f=m.frames[i],ctx=card.querySelector('canvas').getContext('2d');ctx.setTransform(2,0,0,2,0,0);ctx.clearRect(0,0,360,160);ctx.strokeStyle='#c3d0b9';ctx.setLineDash([3,5]);ctx.beginPath();ctx.moveTo(180,0);ctx.lineTo(180,160);ctx.stroke();ctx.setLineDash([]);ctx.strokeStyle='#a7ba95';ctx.beginPath();for(let j=Math.max(0,i-12);j<=i;j++){const g=m.frames[j];if(j===Math.max(0,i-12))ctx.moveTo(g[1],g[2]);else ctx.lineTo(g[1],g[2]);}ctx.stroke();ctx.fillStyle='#397451';ctx.fillRect(10,f[5]-11,5,22);ctx.fillStyle='#ad7547';ctx.fillRect(345,f[6]-11,5,22);ctx.fillStyle='#253b2d';ctx.fillRect(f[1]-3,f[2]-3,6,6);const end=tick>=m.frames.length-1;card.querySelector('p').textContent=end?(m.outcome===2?'Draw · '+(m.draw_reason===1?'exact loop':'time limit'):(m.outcome===0?'Left':'Right')+' scores')+' · '+m.returns+' returns':f[7]+' returns · tick '+f[0];}
$('scrub').value=tick;$('clock').textContent='Tick '+tick+' / '+$('scrub').max+' · '+(tick/30).toFixed(1)+' s';}
function reset(){stop();tick=0;carry=0;$('scrub').max=Math.max(...Object.values(matches()).map(m=>m.frames.length-1));draw();}
$('pair').onchange=reset;$('rally').onchange=reset;$('restart').onclick=reset;$('scrub').oninput=()=>{stop();tick=+$('scrub').value;draw();};$('step').onclick=()=>{stop();tick=Math.min(tick+1,+$('scrub').max);draw();};$('play').onclick=()=>{if(tick>=+$('scrub').max)reset();playing=!playing;$('play').textContent=playing?'Pause':'Play';last=performance.now();};
function animate(now){if(playing){carry+=(now-last)*30*(+$('speed').value)/1000;const steps=Math.floor(carry);carry-=steps;tick=Math.min(tick+steps,+$('scrub').max);if(tick>=+$('scrub').max)stop();if(steps)draw();}last=now;requestAnimationFrame(animate);}requestAnimationFrame(animate);
fetch('offense-replays.json').then(r=>{if(!r.ok)throw Error('Replay request failed');return r.json();}).then(d=>{data=d;for(const id of ['play','step','restart','scrub'])$(id).disabled=false;reset();}).catch(e=>{$('clock').textContent='Could not load replays: '+e.message;});
</script></html>'''


if __name__ == '__main__':
    main()

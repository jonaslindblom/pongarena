import { useState, type MouseEvent } from 'react'
import { ArrowRight, Download, Play } from 'lucide-react'
import type { Controller } from './models'
import { OneShotExplainer } from './OneShotExplainer'
import { findingsData } from './findingsData'
import { assetPath } from './assets'
import './Findings.css'

export type FindingId = 'training' | 'oneshot'
const repository = 'https://github.com/jonaslindblom/pongarena'
const percent = (value: number) => (value * 100).toFixed(1) + '%'

function TrainingFinding({ models, onPlay, onWatch }: {
  models: Controller[]
  onPlay: (id: string) => void
  onWatch: (first: string, second: string) => void
}) {
  const [opponent, setOpponent] = useState<'quill' | 'counter'>('quill')
  const opponentName = opponent === 'quill' ? 'Quill' : 'Counter'
  const primary = findingsData.matchups.find(row => row.model === 'small' && row.opponent === 'quill')!
  const comparisonIds = ['quill', 'small', 'oneshot', 'large', 'counter'].filter(id => id !== opponent)
  const ready = models.some(model => model.id === 'small') && models.some(model => model.id === opponent)

  return <article className="training-finding">
    <header className="finding-heading">
      <span className="section-code">01 / TRAINING TINY NETWORKS</span>
      <h1>The architecture can do it.<br />Can training find the weights?</h1>
      <p>Quill plays with just 163 programmed parameters. We gave a trained network exactly the same layers, then tried distillation, reinforcement learning, and evolutionary search.</p>
    </header>
    <section className="finding-takeaway" aria-labelledby="training-takeaway-title">
      <h2 id="training-takeaway-title">What we learned</h2>
      <p><strong>Our training attempts fell short at 163 parameters.</strong> The selected model trailed Quill and Counter after distillation, reinforcement learning and weight search. Quill’s own weights show that this architecture can play better.</p>
    </section>
    <div className="finding-architecture" aria-label="Shared architecture: 6, 12, 4, 3, 3">
      <span className="section-code">SAME NETWORK</span>
      <strong>6 → 12 → 4 → 3 → 3</strong>
      <span>163 parameters · ReLU · 141 MACs per decision</span>
    </div>
    <div className="finding-models">
      {[
        ['Quill', 'Programmed', 'Geometry and tactics expressed directly as weights. Developed with Codex and refined through play.'],
        ['Small', 'Trained', 'Learned from Large, then refined through play and a search over weight values.'],
        ['OneShot', 'Prompted', 'Numeric weights from one Astra reply to the rules and a tactical brief. No match feedback.'],
      ].map(([name, method, description]) => <div key={name}>
        <span className="section-code">{method}</span><h2>{name}</h2><p>{description}</p>
      </div>)}
    </div>

    <section className="panel finding-result" aria-labelledby="training-result-title">
      <div className="finding-result-intro">
        <span className="section-code">THE RESULT</span>
        <h2 id="training-result-title">Small falls short of Quill.</h2>
        <p>On this fresh comparison, Small scores <strong>{percent(primary.rallyScore)}</strong> of the available rally points against Quill. Matching the layer layout did not close the gap.</p>
        <div className="finding-actions">
          <button className="import-btn" disabled={!ready} onClick={() => onWatch('small', opponent)}><Play size={15} /> Watch Small vs {opponentName}</button>
          <button className="secondary-btn" disabled={!ready} onClick={() => onPlay('small')}>Play against Small <ArrowRight size={15} /></button>
        </div>
      </div>
      <div className="finding-comparison">
        <div className="finding-comparison-heading">
          <h3>How does each model fare?</h3>
          <div className="finding-switch" role="group" aria-label="Benchmark opponent">
            {(['quill', 'counter'] as const).map(id => <button key={id} aria-pressed={opponent === id} onClick={() => setOpponent(id)}>vs {id === 'quill' ? 'Quill' : 'Counter'}</button>)}
          </div>
        </div>
        <p className="finding-chart-key">Rally score · 50% is even · shaded range: 95% confidence interval</p>
        <div className="finding-bars" role="list" aria-label={`Rally scores against ${opponentName}`}>
          {comparisonIds.map(id => {
            const model = findingsData.models.find(model => model.id === id)!
            const result = findingsData.matchups.find(row => row.model === id && row.opponent === opponent)!
            const [low, high] = result.rallyScoreCi95
            return <div className={'finding-bar-row' + (id === 'small' ? ' is-small' : '')} key={id} role="listitem">
              <div className="finding-bar-label"><strong>{model.name}</strong><span>{model.params.toLocaleString()} params</span><b>{percent(result.rallyScore)}</b></div>
              <div className="finding-bar-track" aria-hidden="true">
                <div className="finding-bar-fill" style={{ width: percent(result.rallyScore) }} />
                <div className="finding-bar-interval" style={{ left: percent(low), width: percent(high - low) }} />
              </div>
              <span className="finding-bar-caption">95% CI {percent(low)}–{percent(high)} · paired set score {percent(result.leagueScore)}</span>
            </div>
          })}
        </div>
      </div>
      <p className="finding-benchmark-note">{findingsData.pairedServeSets} fresh serve sets per matchup, played from both sides: {findingsData.ralliesPerMatchup.toLocaleString()} rallies. Rally score counts a win as 1 and a draw as ½. Paired set score instead rewards the winner across ten rallies. These frozen results are separate from the League.</p>
    </section>

    <section className="finding-section" aria-labelledby="training-method-title">
      <span className="section-code">A REASONABLE ATTEMPT</span>
      <h2 id="training-method-title">More than one training run.</h2>
      <p>We ran four Large-distillation starts and two script-imitation controls from random weights. Three students then received PPO reinforcement learning, with 8.4 million environment steps each. Four CMA-ES searches explored 9,216 candidate weight vectors in total.</p>
      <ol className="finding-method">
        <li><strong>Distill a teacher</strong><p>Use a strong, larger network’s action scores to give the small student a useful starting policy.</p></li>
        <li><strong>Learn by playing</strong><p>Refine actions through reward, against scripts and recent copies of the student. The student can move beyond imitation.</p></li>
        <li><strong>Search the weights</strong><p>Try evolutionary refinement of the tiny network directly, without relying on a gradient through the game.</p></li>
      </ol>
      <p>Small was selected using a separate development benchmark before the held-out evaluation. <strong>Quill and Counter were excluded from training and checkpoint selection.</strong> The student received no extra observations at play time, and neither opponent’s weights were copied.</p>
      <details className="panel finding-details">
        <summary>Selection, uncertainty, and reproducibility</summary>
        <div>
          <p>The development benchmark used three scripts and Large on 32 serve sets. Checkpoints were ranked by mean paired set score, with rally score as the tie-breaker. All 13 saved stage outputs were then evaluated on 512 held-out sets. None beat Quill or Counter. Those outputs are related checkpoints, not 13 independent training runs.</p>
          <p>The comparison above uses another shared set of 512 fresh seeds after the five model hashes were frozen. Confidence intervals resample entire paired serve sets, keeping both sides together. These intervals describe variation across serves, not across training runs. The repository includes model hashes, seeds, individual outcomes, recipes, and the unsuccessful candidates’ results.</p>
          <a href={repository + '/tree/main/experiments/quill-training'}>Read the experiment and reproduce the comparison <ArrowRight size={14} /></a>
        </div>
      </details>
    </section>
    <section className="finding-section finding-interpretation" aria-labelledby="training-meaning-title">
      <span className="section-code">WHAT THIS TELLS US</span>
      <h2 id="training-meaning-title">The surprising part is finding the policy.</h2>
      <p>Quill is constructive evidence that this architecture can play well. Our training methods did not find equally effective weights within this experiment’s budget. Large does much better, so learning the game itself is possible.</p>
      <p>That supports a narrow, interesting finding: <strong>training this tiny network to beat the programmed players proved difficult despite several serious attempts.</strong> It does not establish that programmed weights are generally better, or that a better training scheme could not win. The programmed policies also benefited from human and AI iteration.</p>
      <div className="finding-actions">
        <a className="secondary-btn" href={assetPath('models/small.onnx')} download="small.onnx"><Download size={15} /> Download Small</a>
        <a className="secondary-btn" href={repository + '/tree/main/training'}>Try another training scheme <ArrowRight size={15} /></a>
      </div>
    </section>
  </article>
}

export function Findings({ models, selected, onSelect, onLesson, onPlay, onWatch }: {
  models: Controller[]
  selected: FindingId
  onSelect: (event: MouseEvent<HTMLAnchorElement>, id: FindingId) => void
  onLesson: (event: MouseEvent<HTMLAnchorElement>) => void
  onPlay: (id: string) => void
  onWatch: (first: string, second: string) => void
}) {
  return <div className="findings">
    <div className="reading-picker" role="group" aria-label="Choose a finding">
      <p className="section-code">FINDINGS · TWO EXPERIMENTS</p>
      <div className="reading-options">
        {([
          ['training', 'Training tiny networks', 'Can learning match a programmed design?', '#findings'],
          ['oneshot', 'Prompting the weights', 'Put a design into words, then test the weights.', '#findings/oneshot'],
        ] as const).map(([id, title, description, href], index) => <a key={id} href={href}
          aria-label={`${index + 1} · ${title}`} aria-describedby={`finding-${id}-description`}
          aria-current={selected === id ? 'page' : undefined} onClick={event => onSelect(event, id)}>
          <span className="reading-position">EXPERIMENT {index + 1} OF 2 {selected === id && <span>READING</span>}</span>
          <strong>{title}<ArrowRight size={17} aria-hidden="true" /></strong>
          <span id={`finding-${id}-description`} className="reading-description">{description}</span>
        </a>)}
      </div>
    </div>
    {selected === 'training' ? <TrainingFinding models={models} onPlay={onPlay} onWatch={onWatch} />
      : <OneShotExplainer model={models.find(model => model.id === 'oneshot')} onPlay={() => onPlay('oneshot')}
        onWatch={models.some(model => model.id === 'small') ? () => onWatch('oneshot', 'small') : undefined} onLesson={onLesson} />}
    {selected === 'training'
      ? <a className="lesson-next" href="#findings/oneshot" onClick={event => onSelect(event, 'oneshot')}>
        <span><span className="section-code">NEXT FINDING · 2 OF 2</span><strong>Prompting the weights</strong><span>One reply produced a capable policy in the same architecture. See how it compares with Small.</span></span>
        <span>Continue <ArrowRight size={18} aria-hidden="true" /></span>
      </a>
      : <a className="lesson-next" href="#findings" onClick={event => onSelect(event, 'training')}>
        <span><span className="section-code">ALSO IN FINDINGS · 1 OF 2</span><strong>Can training find the weights?</strong><span>See how a learned policy fares in exactly the same architecture as the programmed design.</span></span>
        <span>Explore training <ArrowRight size={18} aria-hidden="true" /></span>
      </a>}
  </div>
}

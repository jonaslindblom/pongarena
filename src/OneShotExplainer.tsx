import { ArrowLeft, ArrowRight, Download, Play } from 'lucide-react'
import { useEffect, useState } from 'react'
import { choice, observation, serve, type Scores } from './game'
import { infer, loadModel, type Controller } from './models'
import { actionLabels as moves } from './observationLabels'
import { oneShotPrompt, oneShotRalliesAgainstLarge, oneShotResults, oneShotUrl, rulesOnlyPrompt, rulesOnlyUrl } from './oneShotExperiment'
import './OneShotExplainer.css'

const opponentPlacements = ['left', 'right'] as const
const opponents = [38, 122] as const
const shot = { x: 80, y: 99, vx: -5, vy: 0.95, paddle: 118 }
const field = ['Follow', 'Small', 'Predict', 'Attack', 'Counter', 'Quill', 'Large']

type ShotSample = { oneShot: Scores; rulesOnly: Scores }

function Court({ opponentPlacement }: { opponentPlacement: 0 | 1 }) {
  const screenBallX = 20 + shot.y
  const screenBallY = 380 - shot.x
  return <svg className="attack-court" viewBox="0 0 200 400" role="img"
    aria-label={`Frozen incoming shot with the opponent on the ${opponentPlacements[opponentPlacement]}.`}>
    <defs>
      <marker id="oneshot-ball-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M 0 0 L 8 4 L 0 8" fill="none" className="attack-velocity" />
      </marker>
    </defs>
    <text className="attack-court-label" x="100" y="13" textAnchor="middle">OPPONENT</text>
    <rect className="attack-boundary" x="20" y="20" width="160" height="360" rx="4" />
    <line className="attack-midline" x1="20" x2="180" y1="200" y2="200" />
    <line className="attack-centerline" x1="100" x2="100" y1="25" y2="370" />
    <rect className="attack-opponent" x={20 + opponents[opponentPlacement] - 15} y="33" width="30" height="5" rx="1.5" />
    <line className="attack-velocity" x1={screenBallX} y1={screenBallY} x2={screenBallX + shot.vy * 5} y2={screenBallY - shot.vx * 5}
      markerEnd="url(#oneshot-ball-arrow)" />
    <circle className="attack-ball" cx={screenBallX} cy={screenBallY} r="4" />
    <rect className="attack-own-paddle" x={20 + shot.paddle - 15} y="363" width="30" height="5" rx="1.5" />
    <text className="attack-court-label" x="100" y="397" textAnchor="middle">YOUR PADDLE</text>
  </svg>
}

function Decision({ name, prompt, scores }: { name: string; prompt: string; scores: Scores }) {
  const move = choice(scores)
  return <article className="oneshot-decision">
    <h3>{name}</h3>
    <p>{prompt}</p>
    <div className="oneshot-move">
      {move === 0 ? <ArrowLeft size={18} aria-hidden="true" /> : move === 2 ? <ArrowRight size={18} aria-hidden="true" /> : null}
      <output aria-label={`${name} chosen move`}>{moves[move]}</output>
    </div>
    <dl className="oneshot-scores" aria-label={`${name} action scores`}>
      {moves.map((label, index) => <div key={label} className={move === index ? 'is-winning' : undefined}>
        <dt>{label}</dt><dd>{scores[index].toFixed(3)}</dd>
      </div>)}
    </dl>
  </article>
}

export function OneShotExplainer({ model, onPlay }: { model?: Controller; onPlay: () => void }) {
  const [opponentPlacement, setOpponentPlacement] = useState<0 | 1>(0)
  const [samples, setSamples] = useState<ShotSample[]>([])
  const [error, setError] = useState('')

  useEffect(() => {
    if (!model) return
    const request = new AbortController()
    setSamples([])
    setError('')
    async function compare(oneShot: Controller) {
      let rulesOnly: Controller | undefined
      try {
        const response = await fetch(rulesOnlyUrl, { signal: request.signal })
        if (!response.ok) throw new Error('Could not load the rules-only model.')
        rulesOnly = await loadModel(new Uint8Array(await response.arrayBuffer()), {
          id: 'rules-only', name: 'Rules only', kind: 'Prompt experiment', description: 'Rules A from the one-prompt experiment.',
        })
        const results: ShotSample[] = []
        for (const opponent of opponents) {
          if (request.signal.aborted) return
          const input = observation({
            ...serve(0), x: shot.x, y: shot.y, vx: shot.vx, vy: shot.vy, paddles: [shot.paddle, opponent],
          }, 0)
          results.push({ oneShot: await infer(oneShot, input), rulesOnly: await infer(rulesOnly, input) })
        }
        if (!request.signal.aborted) setSamples(results)
      } finally {
        await rulesOnly?.session?.release()
      }
    }
    void compare(model).catch(reason => {
      if (!request.signal.aborted) setError(reason instanceof Error ? reason.message : 'Could not compare the models.')
    })
    return () => request.abort()
  }, [model])

  const sample = samples[opponentPlacement]
  return <div className="weights-lab">
    <header className="weights-heading">
      <div>
        <span className="section-code">03 / ONESHOT · 163 PARAMETERS</span>
        <h1>One prompt. One Pong player.</h1>
        <p>GPT-6 Astra returned the weight values for these two controllers in one reply each. Both got the rules and the same network layout. OneShot also got a tactical brief.</p>
      </div>
      <button className="import-btn" onClick={onPlay} disabled={!model}><Play size={15} fill="currentColor" />Play against OneShot</button>
    </header>
    <section className="panel behavior-workbench" aria-label="OneShot comparison">
      <div className="behavior-prompt">
        <h2>Same shot. Move the opponent.</h2>
        <p>Switch the orange paddle to the other side. Compare the two controllers’ next moves.</p>
      </div>
      <div className="behavior-controls">
        <div className="behavior-direction" role="group" aria-label="Opponent position">
          <button aria-pressed={opponentPlacement === 0} onClick={() => setOpponentPlacement(0)}>Opponent left</button>
          <button aria-pressed={opponentPlacement === 1} onClick={() => setOpponentPlacement(1)}>Opponent right</button>
        </div>
      </div>
      <div className="behavior-workspace">
        <div className="behavior-decisions oneshot-scene">
          <Court opponentPlacement={opponentPlacement} />
          <p className="behavior-court-note">The ball and your paddle stay fixed. The arrow shows the ball’s direction.</p>
        </div>
        <div className="behavior-observation oneshot-observation">
          <span className="section-code">NEXT MOVE · HIGHEST SCORE WINS</span>
          {error ? <p className="oneshot-feedback" role="alert">{error}</p> : !sample
            ? <p className="oneshot-feedback" role="status">Comparing the two models…</p>
            : <>
              <div className="oneshot-choices">
                <Decision name="OneShot" prompt="Rules + brief · run 1" scores={sample.oneShot} />
                <Decision name="Rules only" prompt="Run 1" scores={sample.rulesOnly} />
              </div>
              <p className="oneshot-feedback"><strong>OneShot changes its move.</strong> The rules-only controller chooses left in both cases.</p>
              <p className="oneshot-note">Moving the paddle changes where the ball hits it. Shifting the paddle left can send the return right, and vice versa.</p>
            </>}
          <p className="oneshot-note">This compares one decision. The match results below measure play across many serves.</p>
        </div>
      </div>
    </section>
    <section className="oneshot-section" aria-labelledby="oneshot-results-title">
      <span className="section-code">THE EXPERIMENT</span>
      <h2 id="oneshot-results-title">How did the four replies play?</h2>
      <p>We ran each prompt twice. Each reply produced a different controller. OneShot came from run 1 of the rules plus tactical brief. Run 2 used that exact prompt and returned different weights.</p>
      <p>These results use the experiment’s 64 sets against each opponent. The League uses its own shared serves. Each set is two games of five rallies, one from each side. Whoever wins more rallies across both games gets one point. A tie earns half a point.</p>
      <p className="oneshot-scroll-hint">Scroll across to see all opponents.</p>
      <div className="panel oneshot-results" role="region" aria-label="Scrollable results table" tabIndex={0}>
        <table aria-label="OneShot experiment results">
          <thead><tr>
            <th scope="col">Prompt and run</th>
            {field.map(opponent => <th scope="col" key={opponent}>{opponent}</th>)}
          </tr></thead>
          <tbody>{oneShotResults.map(row => <tr key={row.label} className={row.label.includes('OneShot') ? 'is-oneshot' : undefined}>
            <th scope="row">{row.label}</th>
            {row.points.map((points, i) => <td key={field[i]}>{points} / 64</td>)}
          </tr>)}</tbody>
        </table>
      </div>
      <p>OneShot lost all 64 sets against Large, so it earned 0 set points. It still won <strong>{oneShotRalliesAgainstLarge} of 640 individual rallies</strong>.</p>
      <p>OneShot won one set and drew one against Quill. Both replies to the rules plus brief scored higher against Attack than either rules-only reply. Two replies per prompt do not establish how reliably the brief improves a new controller.</p>
    </section>
    <section className="oneshot-section" aria-labelledby="oneshot-prompts-title">
      <span className="section-code">THE INPUTS</span>
      <h2 id="oneshot-prompts-title">Read the exact prompts</h2>
      <p>The arena’s handwritten models also involved AI. We used Codex to develop formulas that calculate their weights and refined the designs through play. For this experiment, each Astra reply supplied the numeric weights directly, with no tools, match feedback or repairs. Neither approach uses training.</p>
      <p>Both prompts used GPT-6 Astra at low reasoning effort, a 6,000-token output limit, and the same numeric JSON response schema.</p>
      <p>These are the original prompts, preserved verbatim with the technical notation used in the experiment.</p>
      <details className="panel oneshot-prompt"><summary>Rules-only prompt</summary><pre>{rulesOnlyPrompt}</pre></details>
      <details className="panel oneshot-prompt"><summary>OneShot prompt · rules + tactical brief</summary><pre>{oneShotPrompt}</pre></details>
      <div className="weights-downloads">
        <a href={oneShotUrl} download="oneshot.onnx"><Download size={15} aria-hidden="true" />Download OneShot</a>
      </div>
    </section>
  </div>
}

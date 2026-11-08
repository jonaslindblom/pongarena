import { Download, Play, ArrowLeft, ArrowRight, RotateCcw } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import * as ort from 'onnxruntime-web/wasm'
import { assetPath } from './assets'
import type { Controller } from './models'
import { observation, serve, type Frame, type Scores } from './game'
import { createCounterIntervention, getAttackExplanation, inspectAttackModel } from './attackExplanation'
import './CounterWeightsLab.css'

const actions = ['Left', 'Stay', 'Right']
const inputs = ['Ball depth', 'Ball position', 'Depth velocity', 'Sideways velocity', 'Your paddle', 'Opponent paddle']
type LabFrame = Pick<Frame, 'x' | 'y' | 'vx' | 'vy' | 'paddles'>
type Inspection = Awaited<ReturnType<typeof inspectAttackModel>>
type Explanation = ReturnType<typeof getAttackExplanation>
interface ModelSample { inspection: Inspection; explanation: Explanation }
interface Snapshot { frame: LabFrame; original: ModelSample; muted: ModelSample }
const number = (n: number, digits = 2) => (Math.abs(n) < .5 * 10 ** -digits ? 0 : n).toFixed(digits)
const frameKey = (frame: LabFrame) => [frame.x, frame.y, frame.vx, frame.vy, ...frame.paddles].join(',')
const startingFrame: LabFrame = { x: 184, y: 120, vx: 13, vy: 0, paddles: [100, 80] }
const neuronValue = (explanation: Explanation, layer: number, index: number) => explanation.neurons.find(n => n.layer === layer && n.index === index)!.value
const layerJobs = [
  { title: 'Read the scene', labels: ['Forecast ramp +', 'Forecast ramp −', 'Carry ball information', 'Carry paddle position', 'Detect departure', 'Read opponent side'] },
  { title: 'Choose a target', labels: ['Gate the forecast', 'Combine paddle + aim', 'Detect approach', 'Limit aim + recover'] },
  { title: 'Account for walls', labels: ['Form the target gap', 'Measure right overshoot', 'Measure left overshoot'] },
  { title: 'Score the moves', labels: ['Left score', 'Stay score', 'Right score'] },
]

function Decision({ sample, frame, muted = false }: { sample: ModelSample; frame: LabFrame; muted?: boolean }) {
  const name = muted ? 'Muted detector' : 'Counter'
  const e = sample.explanation
  const target = Math.max(15, Math.min(145, e.target))
  const ballDepth = 360 - frame.x
  const tipDepth = ballDepth - frame.vx * 3
  return <article className={'attack-model' + (muted ? ' is-muted' : '')} aria-label={`${name} decision`}>
    <div className="attack-model-heading"><h3>{muted ? 'Neuron switched off' : 'Counter'}</h3><span>{muted ? 'Same scene · one change' : 'Original network'}</span></div>
    <svg className="attack-court" viewBox="0 0 200 400" role="img" aria-label={`${name}: ball on the right, moving ${frame.vx > 0 ? 'away from' : 'toward'} your paddle. Move ${actions[e.chosen].toLowerCase()}.`}>
      <text className="attack-court-label" x="100" y="13" textAnchor="middle">OPPONENT</text>
      <rect className="attack-boundary" x="20" y="20" width="160" height="360" rx="4" />
      <line className="attack-midline" x1="20" x2="180" y1="200" y2="200" />
      <line className="attack-centerline" x1="100" x2="100" y1="25" y2="370" />
      <rect className="attack-opponent" x={20 + frame.paddles[1] - 15} y="28" width="30" height="5" rx="1.5" />
      <line className="attack-velocity" x1={20 + frame.y} y1={20 + ballDepth} x2={20 + frame.y} y2={20 + tipDepth} />
      <path className="attack-velocity" d={`M ${15 + frame.y} ${20 + tipDepth + (frame.vx > 0 ? 7 : -7)} l 5 ${frame.vx > 0 ? -7 : 7} l 5 ${frame.vx > 0 ? 7 : -7}`} />
      <circle className="attack-ball" cx={20 + frame.y} cy={20 + ballDepth} r="4" />
      <rect className="attack-target" x={20 + target - 15} y="361" width="30" height="16" rx="3" />
      <rect className="attack-own-paddle" x={20 + frame.paddles[0] - 15} y="367" width="30" height="5" rx="1.5" />
      <text className="attack-court-label" x="100" y="397" textAnchor="middle">YOUR PADDLE</text>
    </svg>
    <div className="attack-decision">{e.chosen === 0 ? <ArrowLeft size={22} /> : e.chosen === 2 ? <ArrowRight size={22} /> : null}<output aria-label={`${name} chosen move`}>{actions[e.chosen]}</output></div>
    <p className="attack-decision-caption">{e.activeForecast === null ? 'Return toward center' : 'Move toward the predicted return'}</p>
    <output className="visually-hidden" aria-label={`${name} paddle target`}>{number(e.target)}</output>
  </article>
}

function NeuronMap({ sample, onInspect }: { sample: ModelSample; onInspect: (neuron: string) => void }) {
  return <section className="neuron-map" aria-label="Counter neuron roles">
    <div className="neuron-map-heading"><span className="section-code">THE REST OF THE NETWORK</span><h2>13 hidden neurons. A job for each.</h2><p>Counter’s weights were written by hand, so we know what each neuron was built to do. Some detect a condition; others carry or combine numbers. Together they predict, aim and recover.</p></div>
    <div className="neuron-layers">{sample.inspection.denseLayers.map((dense, l) => <div className="neuron-layer" key={l}>
      <span className="section-code">{dense.activation ? `HIDDEN LAYER ${l + 1} · ${dense.width} NEURONS` : 'OUTPUT · 3 SCORES'}</span>
      <h3>{layerJobs[l].title}</h3>
      <ul>{dense.roles.map((role, i) => <li key={i}><button className={l === 0 && i === 4 ? 'is-featured' : undefined} onClick={() => onInspect(`${l}:${i}`)} aria-label={`Inspect ${dense.activation ? `H${l + 1}.${i + 1}` : actions[i]}: ${role}`}>
        <span className="neuron-id">{dense.activation ? `H${l + 1}.${i + 1}` : `A${i + 1}`}</span><span>{layerJobs[l].labels[i]}</span>{l === 0 && i === 4 && <small>THIS LESSON</small>}
      </button></li>)}</ul>
    </div>)}</div>
    <p className="neuron-map-note">Select a job to inspect its weights. These are designed roles, not evidence that every neuron in a trained network has one neat purpose. A behavior can need several neurons, and a neuron can contribute to several behaviors.</p>
  </section>
}

function NeuronInspector({ sample, neuron, onNeuron }: { sample: ModelSample; neuron: string; onNeuron: (value: string) => void }) {
  const { inspection, explanation } = sample
  const [layer, index] = neuron.split(':').map(Number)
  const current = explanation.neurons.find(value => value.layer === layer && value.index === index)!
  const dense = inspection.denseLayers[layer]
  return <section className="attack-neuron" aria-label={'Counter weight inspection'}>
    <label className="attack-neuron-picker">Neuron
      <select aria-label="Neuron" value={neuron} onChange={event => onNeuron(event.target.value)}>
        {inspection.denseLayers.map((dense, l) => <optgroup key={l} label={dense.activation ? `Hidden layer ${l + 1}` : 'Action scores'}>
          {dense.roles.map((role, i) => <option key={i} value={`${l}:${i}`}>{dense.activation ? `H${l + 1}.${i + 1}` : actions[i]} · {role}</option>)}
        </optgroup>)}
      </select>
    </label>
    <div className="attack-neuron-description"><strong>{current.role}</strong><span>Stored weights × live input values, then add the bias.</span></div>
    {layer === 0 && <p className="attack-caption">Position inputs are centered at 0. Velocity inputs are court units per tick divided by 6.</p>}
    <p className="attack-table-hint">Scroll the table sideways to read every contribution.</p>
    <div className="attack-table-scroll">
      <table aria-label="Neuron weight contributions">
        <thead><tr><th scope="col">Source</th><th scope="col">Input</th><th scope="col">Weight</th><th scope="col">Contribution</th></tr></thead>
        <tbody>{current.contributions.map(contribution => <tr key={contribution.inputIndex} className={contribution.weight === 0 ? 'is-zero' : undefined}>
          <th scope="row">{layer === 0 ? inputs[contribution.inputIndex] : `H${layer}.${contribution.inputIndex + 1}`}</th>
          <td>{number(contribution.input, 6)}</td><td>{number(contribution.weight, 6)}</td><td>{number(contribution.product, 6)}</td>
        </tr>)}</tbody>
      </table>
    </div>
    <div className="attack-neuron-sum">
      <div><span>Bias</span><output aria-label="Neuron bias">{number(current.bias, 6)}</output></div>
      <div><span>Weighted sum + bias</span><output aria-label="Neuron weighted sum">{number(current.preActivation, 6)}</output></div>
      <div><span>{dense.activation ? 'After ReLU' : 'Linear output'}</span><output aria-label="Neuron activation">{number(current.value, 6)}</output></div>
    </div>
    <p className="attack-caption">{dense.activation ? 'ReLU keeps positive sums and clips negative sums to zero.' : 'The output layer keeps the signed score.'} Zero weights ignore their inputs. Displayed products are rounded; the sum and activation come directly from ONNX.</p>
    <details className="attack-all-weights"><summary>See all 97 parameters</summary>
      {inspection.denseLayers.map((dense, l) => <div key={l} className="attack-table-scroll">
        <table aria-label={`Layer ${l + 1} stored parameters`}><caption>Layer {l + 1} · {dense.inputWidth} → {dense.width} · {dense.activation ?? 'Linear'}</caption>
          <thead><tr><th scope="col">Neuron</th>{Array.from({ length: dense.inputWidth }, (_, i) => <th key={i} scope="col">{l === 0 ? inputs[i] : `H${l}.${i + 1}`}</th>)}<th scope="col">Bias</th></tr></thead>
          <tbody>{dense.weights.map((row, i) => <tr key={i}><th scope="row">{dense.activation ? `H${l + 1}.${i + 1}` : actions[i]}</th>{row.map((value, j) => <td key={j}>{number(value, 6)}</td>)}<td>{number(dense.biases[i], 6)}</td></tr>)}</tbody>
        </table>
      </div>)}
    </details>
  </section>
}

export function CounterWeightsLab({ model, onPlay }: { model: Controller | undefined; onPlay: () => void }) {
  const [frame, setFrame] = useState<LabFrame>(startingFrame)
  const [muted, setMuted] = useState(false)
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [error, setError] = useState('')
  const [neuron, setNeuron] = useState('0:4')
  const advanced = useRef<HTMLDetailsElement>(null)
  const latest = useRef(frame)
  latest.current = frame

  useEffect(() => {
    let disposed = false, busy = false, previous = ''
    let timer: ReturnType<typeof setInterval> | undefined
    let pending: Promise<unknown> | undefined
    const opened: { inspection: Inspection; session: ort.InferenceSession }[] = []
    setSnapshot(null); setError('')
    if (!model) return
    async function open() {
      try {
        if (!model?.modelBytes) throw new Error('Counter’s model bytes are unavailable.')
        const original = await inspectAttackModel(model.modelBytes, 'counter97')
        if (disposed) return
        for (const inspection of [original, createCounterIntervention(original)]) {
          const session = await ort.InferenceSession.create(inspection.bytes, { executionProviders: ['wasm'] })
          if (disposed) { await session.release(); return }
          opened.push({ inspection, session })
        }
        async function sample() {
          if (disposed || busy) return
          const current = latest.current
          const key = frameKey(current)
          if (key === previous) return
          busy = true
          try {
            const input = observation({ ...serve(1), ...current }, 0)
            const result: ModelSample[] = []
            for (const { inspection, session } of opened) {
              pending = session.run({ observation: new ort.Tensor('float32', input, [1, 6]) })
              const outputs = await pending as ort.InferenceSession.ReturnType
              if (disposed) return
              const values = Object.fromEntries(Object.entries(outputs).map(([name, tensor]) => [name, tensor.data as Float32Array]))
              result.push({ inspection, explanation: getAttackExplanation('counter97', input, values, Array.from(values.action_scores) as Scores, inspection) })
            }
            if (key !== frameKey(latest.current)) return
            previous = key
            setSnapshot({ frame: current, original: result[0], muted: result[1] })
          } catch (reason) {
            if (!disposed) { clearInterval(timer); setSnapshot(null); setError(reason instanceof Error ? reason.message : 'Could not inspect Counter.') }
          } finally { busy = false }
        }
        timer = setInterval(() => { void sample() }, 50)
        await sample()
      } catch (reason) {
        if (!disposed) setError(reason instanceof Error ? reason.message : 'Could not open model inspection.')
      }
    }
    void open()
    return () => {
      disposed = true
      clearInterval(timer)
      void (pending ?? Promise.resolve()).catch(() => {}).then(() => Promise.all(opened.map(({ session }) => session.release())))
    }
  }, [model])

  const current = snapshot
  const away = current ? current.frame.vx > 0 : frame.vx > 0
  const detectorOn = current ? neuronValue(current.original.explanation, 0, 4) > 0 : false
  function inspect(neuron: string) {
    setNeuron(neuron)
    if (advanced.current) {
      advanced.current.open = true
      advanced.current.scrollIntoView({ block: 'start' })
    }
  }
  return <div className="weights-lab behavior-lab">
    <header className="weights-heading">
      <div><span className="section-code">02 / COUNTER · 97 PARAMETERS</span><h1>Know when to stop chasing.</h1><p>The ball is on the right. Should the paddle follow it? Counter uses the ball’s direction to choose between meeting a return and getting ready for the next one.</p></div>
      <button className="import-btn" onClick={onPlay} disabled={!model}><Play size={15} fill="currentColor" />Play against Counter</button>
    </header>
    <section className="panel behavior-workbench" aria-label="Counter behavior experiment">
      <div className="behavior-prompt"><h2>Same position. Different direction.</h2><p>Change only the ball’s direction. Watch the move change.</p></div>
      <div className="behavior-controls">
        <div className="behavior-direction" role="group" aria-label="Ball direction">
          <button aria-pressed={frame.vx > 0} onClick={() => setFrame(f => ({ ...f, vx: 13 }))}>Away from you</button>
          <button aria-pressed={frame.vx < 0} onClick={() => setFrame(f => ({ ...f, vx: -13 }))}>Toward you</button>
        </div>
        <button className="behavior-reset" onClick={() => { setFrame(startingFrame); setMuted(false); setNeuron('0:4') }}><RotateCcw size={13} />Reset</button>
      </div>
      {error ? <p className="weights-status" role="alert">{error}</p> : !current ? <p className="weights-status" role="status">Reading Counter’s signals…</p> : <>
        <div className="behavior-workspace" aria-busy={frameKey(current.frame) !== frameKey(frame)}>
          <div className={'behavior-decisions' + (muted ? ' has-comparison' : '')}>
            <Decision sample={current.original} frame={current.frame} />
            {muted && <Decision sample={current.muted} frame={current.frame} muted />}
            <div className="attack-court-legend"><span><i className="legend-paddle" />Paddle now</span><span><i className="legend-target" />Target</span></div>
            <p className="behavior-court-note">One frozen moment. The arrow shows ball direction; the outline shows the paddle’s target.</p>
          </div>
          <div className="behavior-observation">
            <span className="section-code">COUNTER’S ORIGINAL DECISION</span>
            <ol className="behavior-reasoning">
              <li><span className="reasoning-number">1</span><div><h3>Is the ball leaving?</h3><p>A single neuron reads its direction.</p><span className={'detector-signal' + (detectorOn ? ' is-on' : '')}><i /><output aria-label="Outgoing detector state">{detectorOn ? 'On · ball moving away' : 'Off · ball coming back'}</output></span></div></li>
              <li><span className="reasoning-number">2</span><div><h3>{away ? 'Pause the chase' : 'Use the landing forecast'}</h3><p>{away ? 'That signal tells later neurons to set aside the landing forecast and aim back toward center.' : 'With no departure signal, later neurons use the landing forecast to position the paddle for a return.'}</p></div></li>
              <li><span className="reasoning-number">3</span><div><h3>{away ? 'Move left to recover' : 'Move right to intercept'}</h3><p>{away ? 'The paddle is right of center. Moving left gets it ready for the next return.' : 'The predicted return is to the paddle’s right. Moving right gets it into position.'}</p></div></li>
            </ol>
            <div className="behavior-intervention">
              <h3>Does that neuron really matter?</h3><p>Switch it off and compare. The ball, paddles and every other weight stay the same.</p>
              <button className="secondary-btn" aria-pressed={muted} onClick={() => setMuted(value => !value)}>{muted ? 'Restore the departure neuron' : 'Switch off the departure neuron'}</button>
              {muted && <p className="behavior-test-result" role="status">{away ? <><strong>The move flips.</strong> Without the signal, the network moves right toward the predicted return even though the ball is leaving.</> : <><strong>No change this time.</strong> The departure neuron was already off. Both copies move right to meet the return.</>}</p>}
            </div>
          </div>
        </div>
      </>}
    </section>
    {current && <>
      <NeuronMap sample={current.original} onInspect={inspect} />
      <details className="panel behavior-advanced" ref={advanced}>
        <summary>Inspect weights and formulas <span>6 inputs → 13 hidden neurons → 3 action scores</span></summary>
        <div className="behavior-formulas">
          <p><strong>The departure neuron, H1.5:</strong> its only nonzero weight reads the velocity between paddles. Its bias is zero.</p>
          <p><code>departure signal = ReLU(depth velocity ÷ 6)</code></p>
          <p>ReLU keeps positive values and clips negative ones to zero. Here the signal is {number(neuronValue(current.original.explanation, 0, 4), 6)}. Later weights use it to suppress the forecast and cancel the aiming offset.</p>
          <p>This example uses speed 13, where recovery is exactly centered. At slower legal speeds the target can remain slightly off-center. It demonstrates this circuit’s contribution to one decision, not its effect on a whole match.</p>
          {muted && <p>The comparison sets all input weights and the bias of H1.5 to zero in a separate ONNX copy. The inspector below shows the original model.</p>}
        </div>
        <div className="behavior-action-scores"><span className="section-code">ORIGINAL ACTION SCORES · HIGHEST WINS</span><div className="attack-scores">{current.original.explanation.scores.map((value, index) => <div key={index}><span>{actions[index]}</span><output aria-label={`Counter ${actions[index]} score`}>{number(value, 6)}</output></div>)}</div></div>
        <NeuronInspector sample={current.original} neuron={neuron} onNeuron={setNeuron} />
      </details>
    </>}
    <section className="weights-context"><div><h2>A signal becomes a strategy.</h2><p>The departure neuron does not move the paddle by itself. Its connections let later neurons change the target. Follow shares one signal between actions; Counter combines signals into a strategy. Both are programs written directly in weights.</p></div><div className="weights-downloads"><a href={assetPath('models/counter97.onnx')} download="counter97.onnx"><Download size={15} />Download Counter</a></div></section>
  </div>
}

import { Code2, Download, Play, ArrowRight, RotateCcw } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import * as ort from 'onnxruntime-web/wasm'
import counterSourceUrl from '../handwritten/counter97.py?url'
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
interface Snapshot { experiment: Experiment; frame: LabFrame; original: ModelSample; muted: ModelSample }
const number = (n: number, digits = 2) => (Math.abs(n) < .5 * 10 ** -digits ? 0 : n).toFixed(digits)
const frameKey = (frame: LabFrame) => [frame.x, frame.y, frame.vx, frame.vy, ...frame.paddles].join(',')
const clampPaddle = (n: number) => Math.max(15, Math.min(145, n))
const experiments = [
  { id: 'wall', title: 'Detect a wall', question: 'How can a neuron detect a boundary?',
    task: 'Move the ball until its forecast crosses the right wall. Watch the wall neuron wake up and the landing estimate turn back.',
    frame: { x: 240, y: 80, vx: -13, vy: 3, paddles: [100, 80] } },
  { id: 'recovery', title: 'Switch behavior', question: 'What makes the network stop chasing?',
    task: 'Reverse the ball direction, then mute the outgoing detector. Test whether this neuron helps cause recovery.',
    frame: { x: 184, y: 120, vx: 13, vy: 0, paddles: [100, 80] } },
  { id: 'aim', title: 'Aim a return', question: 'How does interception become an attacking shot?',
    task: 'Move the opponent across center. Keep the ball fixed and watch a small paddle offset change the intended corner.',
    frame: { x: 16, y: 80, vx: -13, vy: 0, paddles: [80, 80] } },
] satisfies { id: string; title: string; question: string; task: string; frame: LabFrame }[]
type Experiment = typeof experiments[number]['id']
const startingNeuron: Record<Experiment, string> = { wall: '2:1', recovery: '0:4', aim: '0:5' }
const neuronValue = (explanation: Explanation, layer: number, index: number) => explanation.neurons.find(n => n.layer === layer && n.index === index)!.value

function Court({ name, frame, explanation, experiment }: { name: string; frame: LabFrame; explanation: Explanation; experiment: Experiment }) {
  const target = clampPaddle(explanation.target)
  const tipX = Math.max(4, Math.min(156, frame.y + frame.vy * 2))
  const tipY = Math.max(4, Math.min(356, 360 - frame.x - frame.vx * 2))
  return <div className="attack-court-wrap">
    <span className="attack-end-label">OPPONENT</span>
    <svg className="attack-court" viewBox="0 0 160 360" role="img" aria-label={`${name}: ball at ${number(frame.y)}, paddle at ${number(frame.paddles[0])}, target ${number(explanation.target)}. ${actions[explanation.chosen]}.`}>
      <rect className="attack-boundary" x="1" y="1" width="158" height="358" rx="3" />
      <line className="attack-midline" x1="1" x2="159" y1="180" y2="180" />
      <line className="attack-centerline" x1="80" x2="80" y1="1" y2="359" />
      {explanation.reflected !== null && <>
        <line className="attack-estimate-line" x1={frame.y} y1={360 - frame.x} x2={explanation.reflected} y2="331" />
        <circle className="attack-estimate" cx={explanation.reflected} cy="331" r="4" />
      </>}
      {experiment === 'aim' && explanation.cornerTarget !== null && <path className="attack-corner" d={`M ${explanation.cornerTarget} 25 l 5 6 l -5 6 l -5 -6 Z`} />}
      <rect className="attack-opponent" x={frame.paddles[1] - 15} y="8" width="30" height="5" rx="1.5" />
      <rect className="attack-own-paddle" x={frame.paddles[0] - 15} y="347" width="30" height="5" rx="1.5" />
      <rect className="attack-target" x={target - 15} y="341" width="30" height="16" rx="3" />
      <line className="attack-velocity" x1={frame.y} y1={360 - frame.x} x2={tipX} y2={tipY} />
      <circle className="attack-ball" cx={frame.y} cy={360 - frame.x} r="4" />
      <circle className="attack-velocity-tip" cx={tipX} cy={tipY} r="1.5" />
    </svg>
    <span className="attack-end-label">{frame.vx > 0 ? 'BALL MOVING AWAY' : 'BALL APPROACHING'}</span>
  </div>
}

function Decision({ sample, frame, experiment, muted = false }: { sample: ModelSample; frame: LabFrame; experiment: Experiment; muted?: boolean }) {
  const name = muted ? 'Muted detector' : 'Counter97'
  const e = sample.explanation
  return <article className={'attack-model' + (muted ? ' is-muted' : '')} aria-label={`${name} decision`}>
    <div className="attack-model-heading"><h3>{name}</h3><span>{muted ? 'H1.5 held at zero' : 'Original weights'}</span></div>
    <Court name={name} frame={frame} explanation={e} experiment={experiment} />
    <div className="attack-decision"><output aria-label={`${name} chosen move`}>{actions[e.chosen]}</output><span>Paddle target <output aria-label={`${name} paddle target`}>{number(e.target)}</output>{e.target < 15 || e.target > 145 ? ' · beyond reach' : ''}</span></div>
    <p className="attack-decision-caption">{e.activeForecast === null ? 'Forecast off · recovering toward center' : 'Forecast on · tracking and aiming'}</p>
  </article>
}

function Readout({ label, value, note, active }: { label: string; value: string; note: string; active?: boolean }) {
  return <div className={'behavior-readout' + (active ? ' is-active' : '')}>
    <span>{label}</span><output aria-label={label}>{value}</output><small>{note}</small>
  </div>
}

function ExperimentReadouts({ experiment, sample }: { experiment: Experiment; sample: ModelSample }) {
  const e = sample.explanation
  if (experiment === 'wall') {
    const wall = neuronValue(e, 2, 1)
    return <>
      <Readout label="Unfolded forecast" value={number(e.forecast)} note="Before the wall correction · court units" />
      <Readout label="Wall neuron activation" value={number(wall, 6)} note="H3.2 · right wall overshoot, normalized" active={wall > 0} />
      <Readout label="Reflected forecast" value={number(e.reflected!)} note="After subtracting twice the overshoot" />
    </>
  }
  if (experiment === 'recovery') return <>
    <Readout label="Outgoing detector activation" value={number(neuronValue(e, 0, 4), 6)} note="H1.5 · ReLU of depth velocity ÷ 6" active={neuronValue(e, 0, 4) > 0} />
    <Readout label="Incoming gate activation" value={number(e.incomingGate!, 6)} note="H2.3 · 1 on approach, 0 on departure" active={e.incomingGate! > 0} />
    <Readout label="Forecast carrier activation" value={number(neuronValue(e, 1, 0), 6)} note="H2.1 · carries the forecast + 4, or zero" active={neuronValue(e, 1, 0) > 0} />
  </>
  return <>
    <Readout label="Aim signal" value={number(e.aimSignal!, 6)} note="Two ramps clip opponent preference to −1…+1" active={e.aimSignal !== 0} />
    <Readout label="Aim offset" value={number(e.aimOffset)} note="Paddle target minus landing estimate · court units" />
    <Readout label="Intended corner" value={e.cornerTarget! < 79 ? 'Left' : e.cornerTarget! > 81 ? 'Right' : 'Center'} note="Aiming assumes a fast return; it is not a guarantee" />
  </>
}

function Lesson({ experiment }: { experiment: Experiment }) {
  switch (experiment) {
    case 'wall': return <>
      <h3>A threshold becomes a geometric operation.</h3>
      <p>The wall neuron is quiet until the forecast passes 156, the ball-center boundary. Beyond it, ReLU measures the overshoot. Later weights subtract twice that amount to reflect the estimate.</p>
      <p className="behavior-design"><strong>Design idea</strong> Turn “beyond a boundary” into a threshold neuron, then let the next layer use its signal. A second neuron handles the other wall.</p>
    </>
    case 'recovery': return <>
      <h3>Test a role by removing its signal.</h3>
      <p>The outgoing detector feeds several later neurons. Together they suppress the forecast and cancel the aiming offset on departure. Muting H1.5 tests its contribution while keeping all six inputs fixed.</p>
      <p className="behavior-design"><strong>Design idea</strong> Detect a condition, then use it to gate a calculation. Reverse to approach: muting makes no difference because H1.5 is already zero.</p>
      <details className="behavior-comparison-note"><summary>Why spend a neuron on direction?</summary><p>Attack75 gives depth velocity zero weight everywhere, so reversing only that input cannot change its decision. Counter97 uses a direction detector and extra downstream structure to recover. Muting one Counter97 neuron does not recreate Attack75.</p></details>
    </>
    case 'aim': return <>
      <h3>Aiming adjusts an existing target.</h3>
      <p>The forecast stays fixed. The opponent ramp and a later clipping neuron choose a corner away from the opponent. A small shift of our paddle changes where the ball strikes it, angling the return.</p>
      <p className="behavior-design"><strong>Design idea</strong> Reuse the landing estimate and add an aiming correction. One task can need several neurons; one signal can serve several tasks.</p>
    </>
  }
}

function Formulas({ experiment }: { experiment: Experiment }) {
  return <div className="behavior-formulas">
    <p>Formula positions use centered, normalized units: <code>f = (court forecast − 80) / 80</code>. The main readouts show court units unless marked normalized.</p>
    {experiment === 'wall' ? <><p><code>right hinge = ReLU(f − 0.95)</code></p><p><code>reflected f = f − 2 × right hinge + 2 × left hinge</code></p><p>Counter97 approximates flight time with two forecast ramps and an 18-tick lead cap. It does not divide distance by incoming speed. This fold does not handle arbitrarily many wall bounces.</p></>
      : experiment === 'recovery' ? <><p><code>outgoing = ReLU(depth velocity ÷ 6)</code></p><p><code>forecast carrier = ReLU(f + 4 − 20 × outgoing)</code></p><p>Other connections remove the carry offset and cancel aiming. Recovery is exactly centered at departure speed 13; at slower legal speeds its remaining offset is less than three court units.</p></>
        : <><p><code>s = clip(80 × opponent input + 4 × ball input, −1, 1)</code></p><p><code>target = (83 / 78) × reflected f + (19 / 312) × s</code></p><p>The target formula uses centered, normalized positions. The ball term breaks ties near center. The aiming constants assume a fast return.</p></>}
    <p>Left and right scores have opposite signs; Stay has bias 1.1 / 80. The highest score wins, giving a 1.1-unit deadband. The court clamps unreachable paddle targets to 15–145.</p>
  </div>
}
function NeuronInspector({ sample, neuron, onNeuron }: { sample: ModelSample; neuron: string; onNeuron: (value: string) => void }) {
  const { inspection, explanation } = sample
  const [layer, index] = neuron.split(':').map(Number)
  const current = explanation.neurons.find(value => value.layer === layer && value.index === index)!
  const dense = inspection.denseLayers[layer]
  return <section className="attack-neuron" aria-label={'Counter97 weight inspection'}>
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
  const [experiment, setExperiment] = useState<Experiment>('wall')
  const [frame, setFrame] = useState<LabFrame>(experiments[0].frame)
  const [muted, setMuted] = useState(false)
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [error, setError] = useState('')
  const [neuron, setNeuron] = useState('2:1')
  const latest = useRef({ frame, experiment })
  latest.current = { frame, experiment }

  useEffect(() => {
    let disposed = false, busy = false, previous = ''
    let timer: ReturnType<typeof setInterval> | undefined
    let pending: Promise<unknown> | undefined
    const opened: { inspection: Inspection; session: ort.InferenceSession }[] = []
    setSnapshot(null); setError('')
    if (!model) return
    async function open() {
      try {
        if (!model?.modelBytes) throw new Error('Counter97’s model bytes are unavailable.')
        const original = await inspectAttackModel(model.modelBytes, 'counter97')
        if (disposed) return
        for (const inspection of [original, createCounterIntervention(original)]) {
          const session = await ort.InferenceSession.create(inspection.bytes, { executionProviders: ['wasm'] })
          if (disposed) { await session.release(); return }
          opened.push({ inspection, session })
        }
        async function sample() {
          if (disposed || busy) return
          const request = latest.current, current = request.frame
          const key = request.experiment + ':' + frameKey(current)
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
            if (key !== latest.current.experiment + ':' + frameKey(latest.current.frame)) return
            previous = key
            setSnapshot({ experiment: request.experiment, frame: current, original: result[0], muted: result[1] })
          } catch (reason) {
            if (!disposed) { clearInterval(timer); setSnapshot(null); setError(reason instanceof Error ? reason.message : 'Could not inspect Counter97.') }
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

  const current = snapshot?.experiment === experiment ? snapshot : null
  const preset = experiments.find(value => value.id === experiment)!
  const reset = (id: Experiment) => {
    setExperiment(id); setFrame(experiments.find(value => value.id === id)!.frame)
    setMuted(false); setNeuron(startingNeuron[id])
  }
  return <div className="weights-lab behavior-lab">
    <header className="weights-heading">
      <div><span className="section-code">BUILDING BEHAVIOR / COUNTER97</span><h1>Small jobs. A working strategy.</h1><p>Three experiments in a handwritten network: detect a boundary, switch behavior, and adjust a target. See what each signal contributes.</p></div>
      <button className="import-btn" onClick={onPlay} disabled={!model}><Play size={15} fill="currentColor" />Play against Counter97</button>
    </header>
    <div className="behavior-experiments" role="group" aria-label="Choose an experiment">{experiments.map((value, i) =>
      <button key={value.id} aria-pressed={experiment === value.id} onClick={() => reset(value.id)}><span>0{i + 1}</span>{value.title}<ArrowRight size={14} /></button>)}
    </div>
    <section className="panel behavior-workbench" aria-label="Counter97 behavior experiment">
      <div className="behavior-prompt"><span className="section-code">{experiment === 'wall' ? 'DETECT' : experiment === 'recovery' ? 'GATE' : 'COMBINE'}</span><h2>{preset.question}</h2><p>{preset.task}</p></div>
      <div className="behavior-controls">
        {experiment === 'wall' && <label>Ball position <output>{frame.y}</output><input aria-label="Ball position" type="range" min="40" max="150" step="1" value={frame.y} onChange={event => setFrame(f => ({ ...f, y: Number(event.target.value) }))} /><span>The ball moves sideways; depth and velocity stay fixed.</span></label>}
        {experiment === 'recovery' && <div className="behavior-switches"><button className="secondary-btn" onClick={() => setFrame(f => ({ ...f, vx: -f.vx }))}><RotateCcw size={14} />Reverse ball direction</button><button className="secondary-btn" aria-pressed={muted} onClick={() => setMuted(value => !value)}>Mute outgoing detector</button><span>Same position and speed · only direction changes</span></div>}
        {experiment === 'aim' && <label>Opponent position <output>{frame.paddles[1]}</output><input aria-label="Opponent position" type="range" min="15" max="145" step="1" value={frame.paddles[1]} onChange={event => setFrame(f => ({ ...f, paddles: [f.paddles[0], Number(event.target.value)] }))} /><span>0 left · 80 center · 160 right</span></label>}
        <button className="behavior-reset" onClick={() => reset(experiment)}><RotateCcw size={13} />Reset experiment</button>
      </div>
      {error ? <p className="weights-status" role="alert">{error}</p> : !current ? <p className="weights-status" role="status">Reading Counter97’s signals…</p> : <>
        <div className={'behavior-workspace' + (muted ? ' has-comparison' : '')} aria-busy={frameKey(current.frame) !== frameKey(frame)}>
          <div className={'behavior-decisions' + (muted ? ' has-comparison' : '')}>
            <Decision sample={current.original} frame={current.frame} experiment={experiment} />
            {muted && <Decision sample={current.muted} frame={current.frame} experiment={experiment} muted />}
            <div className="attack-court-legend"><span><i className="legend-target" />Paddle target</span><span><i className="legend-estimate" />Landing estimate</span>{experiment === 'aim' && <span><i className="legend-corner" />Intended corner</span>}</div>
            <p className="behavior-court-note">Frozen decision, not a match. The dotted guide is a landing estimate, not a simulated flight path.</p>
          </div>
          <div className="behavior-observation">
            <span className="section-code">ORIGINAL NETWORK · LIVE SIGNALS</span>
            <div className="behavior-readouts"><ExperimentReadouts experiment={experiment} sample={current.original} /></div>
            {muted && <p className="behavior-test-result" role="status">{current.frame.vx > 0 ? <>With H1.5 muted, the forecast stays on: target <strong>{number(current.muted.explanation.target)}</strong> instead of <strong>{number(current.original.explanation.target)}</strong>. The network keeps chasing.</> : <>Both copies give the same result on approach. The detector was already zero.</>}</p>}
            <div className="behavior-lesson"><Lesson experiment={experiment} /></div>
          </div>
        </div>
        <details className="behavior-advanced" key={experiment}>
          <summary>Inspect weights and formulas <span>6 → 6 → 4 → 3 → 3 · 97 parameters</span></summary>
          <Formulas experiment={experiment} />
          <div className="behavior-action-scores"><span className="section-code">ORIGINAL ACTION SCORES</span><div className="attack-scores">{current.original.explanation.scores.map((value, index) => <div key={index}><span>{actions[index]}</span><output aria-label={`Counter97 ${actions[index]} score`}>{number(value, 6)}</output></div>)}</div></div>
          <NeuronInspector sample={current.original} neuron={neuron} onNeuron={setNeuron} />
          {muted && <p className="behavior-formulas">The inspector shows the original network. The comparison copy sets all six input weights and the bias of H1.5 to zero; every downstream weight stays fixed. It runs in ONNX and is used only for this experiment.</p>}
        </details>
      </>}
    </section>
    <section className="weights-context"><div><h2>Design a role. Then test it.</h2><p>These roles were deliberately written into Counter97. In trained networks, tasks can be spread across neurons and a neuron can support several tasks. Activation suggests a role; removing a signal tests its contribution in that situation.</p></div><div className="weights-downloads"><a href={assetPath('models/counter97.onnx')} download="counter97.onnx"><Download size={15} />Download Counter97</a><a href={counterSourceUrl} download="counter97.py"><Code2 size={15} />Read Counter97’s constructor</a></div></section>
  </div>
}

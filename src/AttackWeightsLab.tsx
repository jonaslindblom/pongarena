import { Code2, Download, Play } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import * as ort from 'onnxruntime-web/wasm'
import attackSourceUrl from '../handwritten/attack75.py?url'
import counterSourceUrl from '../handwritten/counter97.py?url'
import { assetPath } from './assets'
import type { Controller } from './models'
import { observation, serve, type Frame, type Scores } from './game'
import { ATTACK_SCENARIOS, getAttackExplanation, inspectAttackModel, type AttackModelId } from './attackExplanation'
import './AttackWeightsLab.css'

const ids = ['attack75', 'counter97'] as const
const names = { attack75: 'Attack75', counter97: 'Counter97' }
const actions = ['Left', 'Stay', 'Right']
const inputs = ['Ball depth', 'Ball position', 'Depth velocity', 'Sideways velocity', 'Your paddle', 'Opponent paddle']
const stages = ['Prediction', 'Wall reflection', 'Corner choice', 'Recovery', 'Move'] as const
type Stage = typeof stages[number]
type LabFrame = Pick<Frame, 'x' | 'y' | 'vx' | 'vy' | 'paddles'>
type Inspection = Awaited<ReturnType<typeof inspectAttackModel>>
type Explanation = ReturnType<typeof getAttackExplanation>
interface ModelSample { inspection: Inspection; explanation: Explanation }
interface Snapshot { frame: LabFrame; models: Record<AttackModelId, ModelSample> }
const number = (n: number, digits = 2) => (Math.abs(n) < .5 * 10 ** -digits ? 0 : n).toFixed(digits)
const frameKey = (frame: LabFrame) => [frame.x, frame.y, frame.vx, frame.vy, ...frame.paddles].join(',')
const clampPaddle = (n: number) => Math.max(15, Math.min(145, n))

function Court({ name, frame, explanation, stage }: { name: string; frame: LabFrame; explanation: Explanation; stage: Stage }) {
  const recovered = explanation.activeForecast === null
  const target = clampPaddle(explanation.target)
  const corner = explanation.cornerTarget
  const tipX = Math.max(4, Math.min(156, frame.y + frame.vy * 2))
  const tipY = Math.max(4, Math.min(356, 360 - frame.x - frame.vx * 2))
  return <div className="attack-court-wrap">
    <span className="attack-end-label">OPPONENT</span>
    <svg className={'attack-court stage-' + stages.indexOf(stage)} viewBox="0 0 160 360" role="img" aria-label={`${name}: ball at ${number(frame.y)}, paddle at ${number(frame.paddles[0])}, target ${number(explanation.target)}. ${actions[explanation.chosen]}.`}>
      <rect className="attack-boundary" x="1" y="1" width="158" height="358" rx="3" />
      <line className="attack-midline" x1="1" x2="159" y1="180" y2="180" />
      <line className="attack-centerline" x1="80" x2="80" y1="1" y2="359" />
      {!recovered && explanation.reflected !== null && <>
        <line className="attack-estimate-line" x1={frame.y} y1={360 - frame.x} x2={explanation.reflected} y2="345" />
        <circle className="attack-estimate" cx={explanation.reflected} cy="331" r="4" />
      </>}
      {corner !== null && !recovered && <path className="attack-corner" d={`M ${corner} 25 l 5 6 l -5 6 l -5 -6 Z`} />}
      <rect className="attack-opponent" x={frame.paddles[1] - 15} y="8" width="30" height="5" rx="1.5" />
      <rect className="attack-own-paddle" x={frame.paddles[0] - 15} y="347" width="30" height="5" rx="1.5" />
      <rect className="attack-target" x={target - 15} y="341" width="30" height="16" rx="3" />
      <line className="attack-velocity" x1={frame.y} y1={360 - frame.x} x2={tipX} y2={tipY} />
      <circle className="attack-ball" cx={frame.y} cy={360 - frame.x} r="4" />
      <circle className="attack-velocity-tip" cx={tipX} cy={tipY} r="1.5" />
    </svg>
    <span className="attack-end-label">MODEL · {frame.vx > 0 ? 'BALL MOVING AWAY' : 'BALL APPROACHING'}</span>
  </div>
}

function stageValue(stage: Stage, explanation: Explanation) {
  switch (stage) {
    case 'Prediction': return explanation.activeForecast === null ? 'Forecast suppressed' : `Unfolded ${number(explanation.forecast)}`
    case 'Wall reflection': return explanation.reflected === null ? 'No landing target' : `Landing ${number(explanation.reflected)}`
    case 'Corner choice': return explanation.cornerTarget === null ? 'Aiming suppressed' : `${explanation.cornerTarget < 79 ? 'Left' : explanation.cornerTarget > 81 ? 'Right' : 'Center'} · offset ${number(explanation.aimOffset)}`
    case 'Recovery': return explanation.activeForecast === null ? `Recover to ${number(explanation.target)}` : 'Keep tracking the forecast'
    case 'Move': return `Target ${number(explanation.target)} · error ${number(explanation.error)}`
  }
}

function StageExplanation({ stage, selected }: { stage: Stage; selected: AttackModelId }) {
  const counter = selected === 'counter97'
  switch (stage) {
    case 'Prediction': return <><h2>Two ramps anticipate the landing.</h2><p>The first two neurons use sideways-velocity weights of <strong>{counter ? '+1.5 and −1.5' : '+1 and −1'}</strong>. Their difference bounds the look-ahead by distance and speed. {counter ? 'Counter97 extends the lead cap from 12 to 18 ticks.' : 'Attack75 caps the lead at 12 ticks.'} A shared 0.75-tick correction accounts approximately for contact between ticks.</p><p>Carrier neurons use positive biases to preserve signed positions through ReLU; later layers remove those offsets. This is an approximate forecast; incoming depth speed is not used to divide distance by time.</p></>
    case 'Wall reflection': return <><h2>ReLU detects how far the forecast crosses a wall.</h2><p>One neuron activates beyond +0.95, the other below −0.95. The output weights subtract twice the overshoot to reflect the forecast back into the court. Both models use this idea; a longer forecast can make Counter97 anticipate a bounce sooner.</p><p>The dotted guide connects the ball to the landing estimate. It is not a simulated flight path.</p></>
    case 'Corner choice': return <><h2>A small paddle offset makes an angled return.</h2><p>The <strong>80 × opponent position</strong> weight dominates corner selection. A <strong>4 × ball position</strong> term breaks near-center ties. Two ReLU ramps clip that signal between −1 and +1.</p><p>Both models share this aiming rule. They move the paddle slightly to the same side as the opponent so the ball hits the opposite part of the paddle and travels toward the open corner. The aiming weights assume a fast return; they do not guarantee a corner hit on every rally.</p></>
    case 'Recovery': return <><h2>{counter ? 'Ball direction switches off the chase.' : 'A zero weight means no direction awareness.'}</h2><p>{counter ? <>The departure neuron reads depth velocity with weight <strong>+1</strong>. ReLU makes it zero for incoming balls. On departure, a <strong>−20</strong> connection suppresses the forecast and other weights cancel the aiming offset.</> : <>Every depth-velocity weight in Attack75 is <strong>zero</strong>. Incoming and outgoing balls at the same position produce the same decision. The selected neuron shows this ignored input; Counter97 spends a neuron on reading it.</>}</p><p>Counter97 recovers exactly to center at maximum depth speed, and within three court units at slower legal speeds. It still uses the shared aggressive aiming rule on approach.</p></>
    case 'Move': return <><h2>Three scores turn target error into movement.</h2><p>The left and right scores have opposite signs. Stay has a constant <strong>1.1 / 80</strong> bias, creating a 1.1-unit deadband around the target. The largest score chooses the move; ties go to the first action.</p><p>The network can request a target beyond paddle reach. The arena clamps the paddle to 15–145; the dashed target on the court shows that reachable position.</p></>
  }
}

function NeuronInspector({ sample, selected, neuron, onNeuron }: { sample: ModelSample; selected: AttackModelId; neuron: string; onNeuron: (value: string) => void }) {
  const { inspection, explanation } = sample
  const [layer, index] = neuron.split(':').map(Number)
  const current = explanation.neurons.find(value => value.layer === layer && value.index === index)!
  const dense = inspection.denseLayers[layer]
  return <section className="attack-neuron" aria-label={`${names[selected]} weight inspection`}>
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
    <details className="attack-all-weights"><summary>See all {selected === 'attack75' ? 75 : 97} parameters</summary>
      {inspection.denseLayers.map((dense, l) => <div key={l} className="attack-table-scroll">
        <table aria-label={`Layer ${l + 1} stored parameters`}><caption>Layer {l + 1} · {dense.inputWidth} → {dense.width} · {dense.activation ?? 'Linear'}</caption>
          <thead><tr><th scope="col">Neuron</th>{Array.from({ length: dense.inputWidth }, (_, i) => <th key={i} scope="col">{l === 0 ? inputs[i] : `H${l}.${i + 1}`}</th>)}<th scope="col">Bias</th></tr></thead>
          <tbody>{dense.weights.map((row, i) => <tr key={i}><th scope="row">{dense.activation ? `H${l + 1}.${i + 1}` : actions[i]}</th>{row.map((value, j) => <td key={j}>{number(value, 6)}</td>)}<td>{number(dense.biases[i], 6)}</td></tr>)}</tbody>
        </table>
      </div>)}
    </details>
  </section>
}

export function AttackWeightsLab({ selected, attack, counter, onPlay }: { selected: AttackModelId; attack: Controller | undefined; counter: Controller | undefined; onPlay: () => void }) {
  const [scenario, setScenario] = useState(ATTACK_SCENARIOS[0].id)
  const [frame, setFrame] = useState<LabFrame>(ATTACK_SCENARIOS[0].frame)
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [error, setError] = useState('')
  const [stage, setStage] = useState<Stage>('Recovery')
  const [neuron, setNeuron] = useState('0:2')
  const latest = useRef(frame)
  latest.current = frame

  useEffect(() => {
    const mapping: Record<Stage, string> = selected === 'attack75'
      ? { Prediction: '0:0', 'Wall reflection': '1:1', 'Corner choice': '0:4', Recovery: '0:2', Move: '2:2' }
      : { Prediction: '0:0', 'Wall reflection': '2:1', 'Corner choice': '0:5', Recovery: '0:4', Move: '3:2' }
    setNeuron(mapping[stage])
  }, [selected, stage])

  useEffect(() => {
    let disposed = false, busy = false, previous = ''
    let timer: ReturnType<typeof setInterval> | undefined
    let pending: Promise<unknown> | undefined
    const opened: { id: AttackModelId; inspection: Inspection; session: ort.InferenceSession }[] = []
    setSnapshot(null); setError('')
    if (!attack || !counter) return
    async function open() {
      try {
        for (const [id, model] of [['attack75', attack], ['counter97', counter]] as const) {
          if (!model?.modelBytes) throw new Error(`${names[id]}’s model bytes are unavailable.`)
          const inspection = await inspectAttackModel(model.modelBytes, id)
          if (disposed) return
          const session = await ort.InferenceSession.create(inspection.bytes, { executionProviders: ['wasm'] })
          if (disposed) { await session.release(); return }
          opened.push({ id, inspection, session })
        }
        async function sample() {
          if (disposed || busy) return
          const current = latest.current, key = frameKey(current)
          if (key === previous) return
          busy = true
          try {
            const input = observation({ ...serve(1), ...current }, 0)
            const result: Partial<Snapshot['models']> = {}
            for (const { id, inspection, session } of opened) {
              const run = session.run({ observation: new ort.Tensor('float32', input, [1, 6]) })
              pending = run
              const outputs = await run
              if (disposed) return
              const values = Object.fromEntries(Object.entries(outputs).map(([name, tensor]) => [name, tensor.data as Float32Array]))
              result[id] = { inspection, explanation: getAttackExplanation(id, input, values, Array.from(values.action_scores) as Scores, inspection) }
            }
            if (key !== frameKey(latest.current)) return
            previous = key
            setSnapshot({ frame: current, models: result as Snapshot['models'] })
          } catch (reason) {
            if (!disposed) { clearInterval(timer); setSnapshot(null); setError(reason instanceof Error ? reason.message : 'Could not inspect the models.') }
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
  }, [attack, counter])

  const current = snapshot && frameKey(snapshot.frame) === frameKey(frame) ? snapshot : null
  const sample = current?.models[selected]
  const model = selected === 'attack75' ? attack : counter
  const setPaddle = (side: 0 | 1, value: number) => setFrame(frame => ({ ...frame, paddles: side === 0 ? [value, frame.paddles[1]] : [frame.paddles[0], value] }))
  return <div className="weights-lab attack-lab">
    <header className="weights-heading">
      <div><span className="section-code">UNDERSTANDING THE WEIGHTS / {names[selected].toUpperCase()}</span><h1>{selected === 'attack75' ? 'Aiming, written in weights.' : 'Know when to attack. Know when to recover.'}</h1><p>Two handwritten networks, the same situation. Follow the signal from six observations to a move, then inspect the weights that made the difference.</p></div>
      <button className="import-btn" onClick={onPlay} disabled={!model}><Play size={15} fill="currentColor" />Play against {names[selected]}</button>
    </header>
    <section className="panel attack-workbench" aria-label="Compare Attack75 and Counter97">
      <div className="weights-workbench-heading"><strong>Same situation. Two decisions.</strong><span>Live ONNX values · no training</span></div>
      <div className="attack-controls">
        <label>Situation<select aria-label="Situation" value={scenario} onChange={event => {
          const preset = ATTACK_SCENARIOS.find(value => value.id === event.target.value)!
          setScenario(preset.id); setFrame(preset.frame)
          setStage(preset.id === 'outgoing' ? 'Recovery' : preset.id === 'long' ? 'Prediction' : preset.id === 'bounce' ? 'Wall reflection' : 'Corner choice')
        }}>{ATTACK_SCENARIOS.map(value => <option key={value.id} value={value.id}>{value.title}</option>)}</select></label>
        <label>Opponent position <output>{frame.paddles[1]}</output><input aria-label="Opponent position" type="range" min="15" max="145" step="1" value={frame.paddles[1]} onChange={event => setPaddle(1, Number(event.target.value))} /></label>
        <label>Your paddle position <output>{frame.paddles[0]}</output><input aria-label="Your paddle position" type="range" min="15" max="145" step="1" value={frame.paddles[0]} onChange={event => setPaddle(0, Number(event.target.value))} /></label>
      </div>
      <p className="attack-situation">{ATTACK_SCENARIOS.find(value => value.id === scenario)!.description} Positions run from 0 left to 160 right.</p>
      {error ? <p className="weights-status" role="alert">{error}</p> : !current ? <p className="weights-status" role="status">Reading both models’ decisions…</p> : <>
        <div className="attack-comparison">{ids.map(id => {
          const { explanation, inspection } = current.models[id]
          return <article key={id} className={'attack-model' + (selected === id ? ' is-inspected' : '')} aria-label={`${names[id]} decision`}>
            <div className="attack-model-heading"><h2>{names[id]}</h2><span>{inspection.layers.map(layer => layer.width).join(' → ')} · {id === 'attack75' ? 75 : 97} parameters</span></div>
            <Court name={names[id]} frame={current.frame} explanation={explanation} stage={stage} />
            <div className="attack-decision" aria-live="polite"><output aria-label={`${names[id]} chosen move`}>{actions[explanation.chosen]}</output><span>Paddle target <output aria-label={`${names[id]} paddle target`}>{number(explanation.target)}</output>{explanation.target < 15 || explanation.target > 145 ? ' · beyond reach' : ''}</span></div>
            <div className="attack-scores">{explanation.scores.map((value, index) => <div key={index} className={index === explanation.chosen ? 'is-chosen' : undefined}><span>{actions[index]}</span><output aria-label={`${names[id]} ${actions[index]} score`}>{number(value, 6)}</output></div>)}</div>
            <p className="attack-stage-value">{stageValue(stage, explanation)}</p>
          </article>
        })}</div>
        <div className="attack-court-legend"><span><i className="legend-target" />Reachable paddle target</span><span><i className="legend-estimate" />Landing estimate</span><span><i className="legend-corner" />Intended corner</span></div>
        <p className="attack-comparison-caption">{current.models.attack75.explanation.chosen === current.models.counter97.explanation.chosen ? 'Both choose the same move here.' : 'The same observation produces different moves.'} {frame.vx > 0 ? 'Attack75 continues tracking; Counter97 suppresses the forecast and recovers.' : scenario === 'corner' ? 'Both share the same opponent-aware corner rule.' : 'Counter97’s longer look-ahead changes its landing estimate.'}</p>
      </>}
      <div className="attack-stage-picker" role="group" aria-label="Decision stages">{stages.map((value, i) => <button key={value} aria-pressed={stage === value} onClick={() => setStage(value)}><span>0{i + 1}</span>{value}</button>)}</div>
      <div className="attack-stage-explanation"><span className="section-code">INSIDE {names[selected].toUpperCase()}</span><StageExplanation stage={stage} selected={selected} /></div>
      {sample && <NeuronInspector key={selected} sample={sample} selected={selected} neuron={sample.explanation.neurons.some(value => `${value.layer}:${value.index}` === neuron) ? neuron : '0:0'} onNeuron={setNeuron} />}
    </section>
    <section className="weights-context"><div><h2>Different decisions, with visible causes.</h2><p>Attack75 spends 75 parameters on prediction and corner aiming. Counter97 uses 97 to extend anticipation and recover after a shot. Both are handwritten; neither is best against every opponent. These frozen situations explain individual decisions, rather than playing a match. The values above come from copies of the bundled models with their internal tensors exposed.</p></div><div className="weights-downloads"><a href={assetPath(`models/${selected}.onnx`)} download={`${selected}.onnx`}><Download size={15} />Download {names[selected]}</a><a href={selected === 'attack75' ? attackSourceUrl : counterSourceUrl} download={`${selected}.py`}><Code2 size={15} />Read {names[selected]}’s constructor</a></div></section>
  </div>
}

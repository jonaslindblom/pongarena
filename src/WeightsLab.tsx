import { ArrowRight, Code2, Download, Pause, Play } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import * as ort from 'onnxruntime-web/wasm'
import pb from 'onnx-proto'
import followSourceUrl from '../handwritten/follow.py?url'
import { assetPath } from './assets'
import { createBrainGraph } from './brain'
import type { Controller } from './models'
import { choice, type Scores } from './rules'
import './WeightsLab.css'

const inputNames = ['Ball depth', 'Ball position', 'Depth velocity', 'Sideways velocity', 'Paddle position', 'Opponent position']
const actionNames = ['Left', 'Stay', 'Right']
const number = (value: number) => value.toFixed(3).replace(/^-0\.000$/, '0.000')
const weight = (value: number) => Number.isInteger(value) ? String(value) : value.toPrecision(7)
const signed = (value: number) => value > 0 ? `+${weight(value)}` : weight(value)
const clamp = (value: number, limit: number) => Math.max(-limit, Math.min(limit, value))

interface Parameters {
  input: number[]
  hiddenBias: number
  output: number[]
  outputBias: number[]
}
interface Snapshot {
  input: number[]
  error: number
  hidden: number
  scores: Scores
  chosen: number
}

function inspectModel(bytes: Uint8Array) {
  const inspection = createBrainGraph(bytes)
  const model = pb.onnx.ModelProto.decode(inspection.bytes)
  const graph = model.graph!
  const dense = graph.node!.filter((node) => node.opType === 'Gemm')
  if (inspection.layers.length !== 3 || inspection.layers[1].width !== 1 || inspection.layers[1].activation !== 'Tanh' || dense.length !== 2) {
    throw new Error('This explanation needs Follow’s 6 → 1 → 3 Tanh network.')
  }
  function values(name: string | null | undefined, length: number) {
    const tensor = graph.initializer!.find((entry) => entry.name === name)
    if (!tensor) throw new Error('A Follow weight tensor is missing.')
    const raw = tensor.rawData
    const data = raw?.length ? Array.from({ length: raw.byteLength / 4 }, (_, i) => new DataView(raw.buffer, raw.byteOffset, raw.byteLength).getFloat32(i * 4, true)) : Array.from(tensor.floatData ?? [])
    if (data.length !== length || data.some((value) => !Number.isFinite(value))) throw new Error('Follow’s weight tensor has unexpected values.')
    return data
  }
  const parameters: Parameters = {
    input: values(dense[0].input![1], 6), hiddenBias: values(dense[0].input![2], 1)[0],
    output: values(dense[1].input![1], 3), outputBias: values(dense[1].input![2], 3),
  }
  const linearName = dense[0].output![0]!
  graph.output!.push({ name: linearName, type: { tensorType: { elemType: 1, shape: { dim: [{ dimValue: 1 }, { dimValue: 1 }] } } } })
  return { bytes: pb.onnx.ModelProto.encode(model).finish(), parameters, linearName, hiddenName: inspection.layers[1].name }
}

function Network({ parameters, snapshot }: { parameters: Parameters; snapshot: Snapshot }) {
  return (
    <div className="weights-network-scroll" tabIndex={0} aria-label="Follow’s weights and activations; scroll horizontally on narrow screens">
      <svg className="weights-network" viewBox="0 0 760 310" role="img" aria-label={`Ball position ${number(snapshot.input[1])}, paddle position ${number(snapshot.input[4])}, error ${number(snapshot.error)}, Tanh ${number(snapshot.hidden)}. Follow chooses ${actionNames[snapshot.chosen]}.`}>
        <text x="20" y="22" className="weights-graph-heading">6 INPUTS</text>
        <text x="395" y="22" textAnchor="middle" className="weights-graph-heading">1 SHARED NEURON</text>
        <text x="640" y="22" textAnchor="middle" className="weights-graph-heading">3 ACTION SCORES</text>
        {inputNames.map((name, i) => {
          const y = 60 + i * 42
          const active = parameters.input[i] !== 0
          return (
            <g key={name} className={active ? 'weights-input' : 'weights-input is-ignored'}>
              <path d={`M 197 ${y} C 260 ${y}, 260 152, 315 152`} className={active ? 'weights-edge is-active' : 'weights-edge'} />
              <text x="20" y={y - 4} className="weights-node-name">{name}</text>
              <text x="20" y={y + 12} className="weights-node-value">{number(snapshot.input[i])}</text>
              <circle cx="193" cy={y} r="5" fill={active ? i === 1 ? '#c5f77e' : '#eead78' : '#819486'} />
              <rect x="211" y={y - 13} width="31" height="22" rx="5" className="weights-edge-label" />
              <text x="226" y={y + 2} textAnchor="middle" className="weights-edge-value">{signed(parameters.input[i])}</text>
            </g>
          )
        })}
        {actionNames.map((name, i) => {
          const y = 72 + i * 83
          const selected = snapshot.chosen === i
          return (
            <g key={name}>
              <path d={`M 475 152 C 528 152, 528 ${y + 9}, 584 ${y + 9}`} className={'weights-edge' + (parameters.output[i] ? ' is-active' : '')} />
              <rect x="523" y={y - 5} width="33" height="22" rx="5" className="weights-edge-label" />
              <text x="540" y={y + 10} textAnchor="middle" className="weights-edge-value">×{signed(parameters.output[i])}</text>
              <rect x="584" y={y - 23} width="151" height="75" rx="10" className={'weights-output' + (selected ? ' is-selected' : '')} />
              <text x="599" y={y - 3} className="weights-node-name">{name}{selected ? ' ← chosen' : ''}</text>
              <text x="599" y={y + 23} className="weights-score">{number(snapshot.scores[i])}</text>
              <text x="599" y={y + 43} className="weights-bias">bias {signed(parameters.outputBias[i])}</text>
            </g>
          )
        })}
        <rect x="315" y="95" width="160" height="114" rx="12" className="weights-hidden" />
        <text x="395" y="119" textAnchor="middle" className="weights-node-name">gap = {number(snapshot.error)}</text>
        <text x="395" y="150" textAnchor="middle" className="weights-tanh">tanh(gap)</text>
        <text x="395" y="184" textAnchor="middle" className="weights-score">h = {number(snapshot.hidden)}</text>
        <text x="395" y="234" textAnchor="middle" className="weights-bias">bias {signed(parameters.hiddenBias)}</text>
      </svg>
    </div>
  )
}

export function WeightsLab({ model, onPlay }: { model: Controller | undefined; onPlay: () => void }) {
  const [positions, setPositions] = useState({ ball: 0.5, paddle: -0.3 })
  const [animate, setAnimate] = useState(false)
  const [reducedMotion, setReducedMotion] = useState(false)
  const [parameters, setParameters] = useState<Parameters | null>(null)
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [error, setError] = useState('')
  const latest = useRef({ positions, animate })
  latest.current = { positions, animate }

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)')
    const update = () => { setReducedMotion(media.matches); if (media.matches) setAnimate(false) }
    update()
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])

  useEffect(() => {
    let disposed = false, busy = false, direction = 1, previousTime = performance.now()
    let session: ort.InferenceSession | undefined
    let timer: ReturnType<typeof setInterval> | undefined
    let pendingRun: Promise<unknown> | undefined
    let previous: Snapshot | null = null
    setParameters(null); setSnapshot(null); setError('')
    if (!model) return
    async function open() {
      try {
        if (!model!.modelBytes) throw new Error('Follow’s model bytes are unavailable.')
        const graph = inspectModel(model!.modelBytes)
        session = await ort.InferenceSession.create(graph.bytes, { executionProviders: ['wasm'] })
        if (disposed) { await session.release(); return }
        setParameters(graph.parameters)
        async function sample() {
          if (disposed || busy) return
          const now = performance.now()
          const elapsed = Math.min(0.1, (now - previousTime) / 1000)
          previousTime = now
          let current = latest.current.positions
          if (latest.current.animate && !document.hidden) {
            if (current.ball >= 0.8) direction = -1
            if (current.ball <= -0.8) direction = 1
            current = {
              ball: clamp(current.ball + direction * elapsed * 0.38, 0.95),
              paddle: clamp(current.paddle + ((previous?.chosen ?? 1) - 1) * elapsed * 0.75, 0.8125),
            }
            latest.current.positions = current
            setPositions(current)
          }
          const input = Float32Array.of(0, current.ball, 0, 0, current.paddle, 0)
          if (previous && previous.input[1] === input[1] && previous.input[4] === input[4]) return
          busy = true
          try {
            const run = session!.run({ observation: new ort.Tensor('float32', input, [1, 6]) })
            pendingRun = run
            const result = await run
            if (disposed) return
            const scores = Array.from(result.action_scores.data as Float32Array) as Scores
            const gap = (result[graph.linearName].data as Float32Array)[0]
            const hidden = (result[graph.hiddenName].data as Float32Array)[0]
            if ([gap, hidden, ...scores].some((value) => !Number.isFinite(value))) throw new Error('Follow returned a non-finite activation.')
            previous = { input: Array.from(input), error: gap, hidden, scores, chosen: choice(scores) }
            setSnapshot(previous)
          } catch (reason) {
            if (!disposed) {
              clearInterval(timer)
              setAnimate(false)
              setSnapshot(null)
              setError(reason instanceof Error ? reason.message : 'Could not read Follow’s activations.')
            }
          } finally { busy = false }
        }
        timer = setInterval(() => { void sample() }, 50)
        await sample()
      } catch (reason) {
        if (!disposed) setError(reason instanceof Error ? reason.message : 'Could not open Follow.')
      }
    }
    void open()
    return () => {
      disposed = true
      clearInterval(timer)
      if (session) {
        const currentSession = session
        void (pendingRun ?? Promise.resolve()).catch(() => {}).then(() => currentSession.release())
      }
    }
  }, [model])

  const movePosition = (key: 'ball' | 'paddle', value: number) => {
    setAnimate(false)
    setPositions((current) => ({ ...current, [key]: value }))
  }
  const allValues = parameters ? [...parameters.input, parameters.hiddenBias, ...parameters.output, ...parameters.outputBias] : []
  return (
    <div className="weights-lab">
      <header className="weights-heading">
        <div><span className="section-code">UNDERSTANDING THE WEIGHTS</span><h1>A tiny brain you can read.</h1><p>Start here: move the ball and paddle, then trace how weights turn their positions into a decision. Follow shares one neuron between left and right, with no training.</p></div>
        <button className="import-btn" onClick={onPlay} disabled={!model}><Play size={15} fill="currentColor" />Play against Follow</button>
      </header>
      <section className="panel weights-workbench" aria-label="Interactive Follow model">
        <div className="weights-workbench-heading"><strong>Move the ball. Read the decision.</strong><span>6 → 1 → 3 · {model?.params ?? 13} parameters · {model?.macs ?? 9} MACs</span></div>
        <div className="weights-interactive">
          <div className="weights-position-controls">
            <div className="weights-mini-court" role="img" aria-label={`Ball at ${number(positions.ball)}, paddle at ${number(positions.paddle)}. Positions run left to right.`}>
              <span className="weights-court-label">BALL</span>
              <i className="weights-demo-ball" style={{ left: `${(positions.ball + 1) * 50}%` }} />
              <span className="weights-gap" style={{ left: `${(Math.min(positions.ball, positions.paddle) + 1) * 50}%`, width: `${Math.abs(positions.ball - positions.paddle) * 50}%` }} />
              <i className="weights-demo-paddle" style={{ left: `${(positions.paddle + 1) * 50}%` }} />
              <span className="weights-court-label is-paddle">PADDLE</span>
            </div>
            <label htmlFor="weights-ball">Ball position <output>{number(positions.ball)}</output></label>
            <input id="weights-ball" type="range" min={-0.95} max={0.95} step={0.0125} value={positions.ball} onChange={(event) => movePosition('ball', Number(event.target.value))} />
            <label htmlFor="weights-paddle">Paddle position <output>{number(positions.paddle)}</output></label>
            <input id="weights-paddle" type="range" min={-0.8125} max={0.8125} step={0.0125} value={positions.paddle} onChange={(event) => movePosition('paddle', Number(event.target.value))} />
            <div className="weights-scale"><span>−1 left</span><span>0 center</span><span>+1 right</span></div>
            <button className="secondary-btn weights-animate" onClick={() => setAnimate(!animate)} aria-pressed={animate} disabled={reducedMotion || !snapshot} title={reducedMotion ? 'Animation is off because you prefer reduced motion. Use the sliders to explore.' : undefined}>{animate ? <Pause size={14} /> : <Play size={14} />}{animate ? 'Pause' : 'Animate'}</button>
            <p className="weights-demo-caption">{reducedMotion ? 'Use the sliders to explore at your own pace.' : 'Animate lets the model move the paddle.'} This is a position demo, not a match.</p>
          </div>
          <div className="weights-live-network">
            {snapshot && <p className="weights-scroll-hint">Swipe to follow the signal →</p>}
            {error ? <p className="weights-status" role="alert">{error}</p> : parameters && snapshot ? <Network parameters={parameters} snapshot={snapshot} /> : <p className="weights-status" role="status">Loading Follow’s weights…</p>}
            <p className="weights-network-caption">{snapshot && <strong>Decision: {actionNames[snapshot.chosen]}. </strong>}Live ONNX values. Lines show the stored weights; the highest action score wins. Four zero weights ignore four inputs.</p>
          </div>
        </div>
        <div className="weights-steps">
          <div><span>01</span><h2>Subtract two positions</h2><p><code>+1 × ball − 1 × paddle</code> gives the signed gap. Positive means the ball is to the right.</p></div>
          <div><span>02</span><h2>Share one signal</h2><p><code>h = tanh(gap)</code> keeps its sign and squeezes it between −1 and +1. Left and right reuse this signal; stay uses a constant bias.</p></div>
          <div><span>03</span><h2>Choose left, stay or right</h2><p>The scores are <code>−h, δ, +h</code>. The small stay bias δ avoids jitter within about 1.5 game units of the ball.</p></div>
        </div>
        {parameters && <details className="weights-parameters">
          <summary>See all {allValues.length} parameters <span>{allValues.filter((value) => value !== 0).length} nonzero · zeros count too</span></summary>
          <div className="weights-tables">
            <table><caption>Inputs → hidden neuron · 7 values</caption><thead><tr><th scope="col">Input</th><th scope="col">Weight</th></tr></thead><tbody>{inputNames.map((name, i) => <tr key={name}><th scope="row">{name}</th><td>{weight(parameters.input[i])}</td></tr>)}<tr><th scope="row">Hidden bias</th><td>{weight(parameters.hiddenBias)}</td></tr></tbody></table>
            <div><table><caption>Hidden neuron → actions · 6 values</caption><thead><tr><th scope="col">Action</th><th scope="col">Weight</th><th scope="col">Bias</th></tr></thead><tbody>{actionNames.map((name, i) => <tr key={name}><th scope="row">{name}</th><td>{weight(parameters.output[i])}</td><td>{weight(parameters.outputBias[i])}</td></tr>)}</tbody></table><p>A direct 6 → 3 layer would need 21 parameters. Sharing one hidden neuron reduces that to 13, the smallest architecture allowed by this arena’s dense model contract.</p><p>The displayed horizontal position is called Y in the simulator. Inputs here use the same centered, normalized coordinates as a match.</p></div>
          </div>
        </details>}
      </section>
      <section className="weights-context" aria-label="Weight programming and source files">
        <div><h2>A small program, written in weights.</h2><p>Follow tracks the ball’s current position. It cannot anticipate bounces or aim a return. <a href="https://rikiyat.github.io/blog/weight-space-programming/" target="_blank" rel="noreferrer">Sparks of Weight-Space Programming <ArrowRight size={13} /></a> explores LLMs reading and writing weights directly. Here, the open question is practical: how far can handwritten models go against independently trained ones?</p></div>
        <div className="weights-downloads"><a href={assetPath('models/follow.onnx')} download="follow.onnx"><Download size={15} />Download model</a><a href={followSourceUrl} download="follow.py"><Code2 size={15} />Read the constructor</a></div>
      </section>
    </div>
  )
}

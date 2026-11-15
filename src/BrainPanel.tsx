import { Brain, ChevronDown } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import * as ort from 'onnxruntime-web/wasm'
import { activationLevels, createBrainGraph, type BrainLayer } from './brain'
import type { Controller } from './models'
import { choice, type Frame, type Pair, type Ruleset, type Side } from './rules'
import type { Match } from './useMatch'
import { playerNumber, topSide } from './courtPlayers'
import { actionLabels, observationDescriptions, observationLabels, observationValueHelp } from './observationLabels'
import './BrainPanel.css'

interface Snapshot {
  layers: BrainLayer[]
  values: number[][]
  tick: number
  chosen: number
}
const number = (value: number) => value !== 0 && Math.abs(value) < 0.001 ? value.toExponential(1) : value.toFixed(3)

function Network({ layers, snapshot }: { layers: BrainLayer[]; snapshot: Snapshot | null }) {
  const x = (index: number) => 180 + index * 560 / (layers.length - 1)
  return (
    <div className="brain-graph-scroll" tabIndex={0} aria-label="Neural activation graph; scroll horizontally on narrow screens">
      <svg className="brain-graph" viewBox="0 0 836 280" role="img" aria-label={snapshot ? `Real neuron activations at tick ${snapshot.tick}; selected move ${actionLabels[snapshot.chosen]}` : 'Neural network layout; awaiting decision'}>
        {layers.slice(1).map((_, i) => (
          <path key={i} d={`M ${x(i) + 42} 134 C ${x(i) + 62} 134, ${x(i + 1) - 62} 134, ${x(i + 1) - 42} 134`}
            className="brain-flow" />
        ))}
        {layers.map((layer, index) => {
          const data = snapshot?.values[index] ?? Array<number>(layer.width).fill(0)
          const { peak, intensities } = activationLevels(data)
          const visible = Math.min(256, layer.width)
          const columns = layer.kind === 'hidden' ? Math.ceil(Math.sqrt(visible / 2)) : 1
          const rows = Math.ceil(visible / columns)
          const small = layer.kind === 'hidden' && layer.width <= 32
          const spacing = small ? 17 : 10
          return (
            <g key={layer.name}>
              <text x={x(index)} y="25" textAnchor="middle" className="brain-layer-name">{layer.kind === 'input' ? 'INPUT' : layer.kind === 'output' ? 'ACTION' : `HIDDEN ${index}`}</text>
              <text x={x(index)} y="43" textAnchor="middle" className="brain-layer-size">{layer.width}{layer.activation ? ` · ${layer.activation === 'Relu' ? 'ReLU' : 'Tanh'}` : ''}</text>
              {data.slice(0, visible).map((value, neuron) => {
                const cx = x(index) + (neuron % columns - (columns - 1) / 2) * spacing
                const cy = 134 + (Math.floor(neuron / columns) - (rows - 1) / 2) * (layer.kind === 'hidden' ? Math.min(small ? 17 : 9, 156 / Math.max(1, rows - 1)) : layer.kind === 'output' ? 43 : 25)
                const selected = layer.kind === 'output' && neuron === snapshot?.chosen
                const label = layer.kind === 'input' ? observationLabels[neuron] : layer.kind === 'output' ? actionLabels[neuron] : `Neuron ${neuron + 1}`
                return (
                  <g key={neuron}>
                    {selected && <circle cx={cx} cy={cy} r="11" className="brain-selected" />}
                    <circle cx={cx} cy={cy} r={layer.kind === 'hidden' ? small ? 5.3 : 3.1 : 5.5}
                      fill={value === 0 ? '#91a594' : value < 0 ? '#eead78' : '#bce78b'} fillOpacity={value === 0 ? 0.08 : 0.25 + 0.75 * intensities[neuron]}
                      stroke={value === 0 ? '#65816e' : 'none'} strokeWidth={0.6}>
                      <title>{label}: {snapshot ? value.toPrecision(7) : 'Awaiting decision'}{layer.kind === 'input' ? `. ${observationDescriptions[neuron]}` : ''}</title>
                    </circle>
                    {layer.kind === 'input' && <text x={cx - 12} y={cy + 3} textAnchor="end" className="brain-node-label">{label}</text>}
                    {layer.kind === 'output' && <text x={cx + 18} y={cy + 4} className={'brain-node-label' + (selected ? ' chosen' : '')}>{label} {snapshot ? number(value) : '—'}</text>}
                  </g>
                )
              })}
              {layer.kind === 'hidden' && <text x={x(index)} y="235" textAnchor="middle" className="brain-active">{snapshot ? `${data.filter((value) => value !== 0).length} / ${layer.width} active` : 'Awaiting decision'}</text>}
              <text x={x(index)} y="255" textAnchor="middle" className="brain-range">max |value| {snapshot ? number(peak) : '—'}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function BrainView({ model, side, frame, previous, start, rules }: {
  model: Controller; side: Side; frame: Frame; previous?: Frame; start: Frame; rules: Ruleset
}) {
  const latest = useRef({ frame, previous, start })
  latest.current = { frame, previous, start }
  const [layers, setLayers] = useState<BrainLayer[]>([])
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    let disposed = false, busy = false
    let session: ort.InferenceSession | undefined
    let timer: ReturnType<typeof setInterval> | undefined
    let pendingRun: Promise<unknown> | undefined
    let last: Frame | undefined
    async function open() {
      try {
        if (!model.modelBytes) throw new Error('Original model bytes are unavailable for inspection.')
        const graph = createBrainGraph(model.modelBytes)
        setLayers(graph.layers)
        session = await ort.InferenceSession.create(graph.bytes, { executionProviders: ['wasm'] })
        if (disposed) { await session.release(); return }
        setLoading(false)
        async function sample() {
          if (disposed || busy) return
          const current = latest.current
          const expected = current.frame.scores[side]
          if (!expected || !current.previous) {
            last = undefined
            setSnapshot(null)
            return
          }
          if (last === current.frame) return
          busy = true
          last = current.frame
          try {
            const input = rules.observation(current.previous, side)
            const run = session!.run({ observation: new ort.Tensor('float32', input, [1, input.length]) })
            pendingRun = run
            const result = await run
            if (disposed || latest.current.start !== current.start || latest.current.frame.tick < current.frame.tick) return
            const values = graph.layers.map((layer) => layer.kind === 'input' ? Array.from(input) : Array.from(result[layer.name].data as Float32Array))
            if (values.some((layer) => layer.some((value) => !Number.isFinite(value)))) throw new Error('The activation sample contains non-finite values.')
            if (choice(values[values.length - 1] as [number, number, number]) !== choice(expected)) throw new Error('The activation sample could not reproduce this decision.')
            setSnapshot({ layers: graph.layers, values, tick: current.frame.tick, chosen: choice(expected) })
            setError('')
          } catch (reason) {
            if (!disposed) { setSnapshot(null); setError(reason instanceof Error ? reason.message : 'Could not read model activations.') }
          } finally { busy = false }
        }
        timer = setInterval(() => { void sample() }, 250)
        await sample()
      } catch (reason) {
        if (!disposed) { setLoading(false); setError(reason instanceof Error ? reason.message : 'Could not open model inspection.') }
      }
    }
    void open()
    return () => {
      disposed = true
      clearInterval(timer)
      // Let an in-flight WASM run finish before releasing its session.
      if (session) {
        const currentSession = session
        void (pendingRun ?? Promise.resolve()).catch(() => {}).then(() => currentSession.release())
      }
    }
  }, [model, rules, side])
  return (
    <div className="brain-view">
      <div className="brain-source"><span>{model.name}</span><span>{snapshot ? `DECISION TICK ${snapshot.tick}` : loading ? 'OPENING MODEL' : 'AWAITING DECISION'}</span></div>
      {error ? <p className="brain-message" role="status">{error}</p> : layers.length > 0 ? <Network layers={layers} snapshot={snapshot} /> : <p className="brain-message">Reading the network…</p>}
      <div className="brain-legend"><span><i /> Positive</span><span><i /> Negative</span><span>Dark = zero</span></div>
      <p className="brain-caption">{observationValueHelp}</p>
      <p className="brain-caption">Real model values · sampled up to 4 times/s. Brightness is scaled within each layer and emphasizes small values; hover a node for its value. Active counts show nonzero values; zero is normal for ReLU. Lines show layer flow, not connection strengths.{layers.some((layer) => layer.width > 256) ? ' First 256 neurons per layer shown.' : ''}</p>
    </div>
  )
}

export function BrainPanel({ m, controllers, bottom }: { m: Match; controllers: Pair<Controller | undefined>; bottom: Side }) {
  const [open, setOpen] = useState(true)
  const [side, setSide] = useState<Side>(topSide(bottom))
  useEffect(() => {
    setSide(topSide(bottom))
  }, [bottom])
  const model = controllers[side]
  return (
    <section className={'panel brain-panel' + (open ? ' is-open' : '')} aria-label="Model brain">
      <div className="brain-heading">
        <button className="brain-toggle" onClick={() => setOpen(!open)} aria-expanded={open} aria-controls="brain-content" aria-label={open ? 'Collapse brain' : 'Expand brain'}><Brain size={18} />Brain<ChevronDown size={15} /></button>
        {open ? <div className="segmented" aria-label="Brain paddle">{([topSide(bottom), bottom] as const).map((player) => <button key={player} onClick={() => setSide(player)} aria-pressed={side === player} aria-label={'Show brain for player ' + playerNumber(player, bottom)}>P{playerNumber(player, bottom)} · {player === bottom ? 'Bottom' : 'Top'}</button>)}</div> : <span className="brain-hint">Live neuron activations</span>}
      </div>
      {open && <div id="brain-content">{model?.type === 'weights' ? <BrainView key={`${side}:${model.id}:${m.seed}`} model={model} side={side} frame={m.frame} previous={m.history[m.tick - 1]} start={m.history[0]} rules={m.rules} /> : <p className="brain-message">{model ? `${model.name} has no neural network. Choose a weight model for this paddle, or inspect the other player.` : 'Waiting for a model to load.'}</p>}</div>}
    </section>
  )
}

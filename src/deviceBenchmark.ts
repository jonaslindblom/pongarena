import { advance, choice, endedMatch, endedRally, LoopDetector, observation, serve } from './game'
import type { Controller } from './models'
import type { Scores } from './rules'
import { scriptScores } from './scripted'
import type { MatchTickTiming } from './useMatch'

export const DEVICE_TEST = {
  version: 'rally-device-v3',
  seed: 902,
  inputCount: 512,
  warmup: 128,
  batchSize: 2048,
  batches: 7,
  animationMs: 15000,
  modelIds: ['tracker', 'robust-strong-v4', 'prism-small-v4'],
  matchIds: ['prism-small-v4', 'quill-plus-v4'],
} as const

export const DEVICE_TEST_METHODS = {
  inputs: '512 consecutive legal Classic v4 positions, seed 902, alternating observing side; Attacker vs Defender generates the positions.',
  inference: 'Each model is warmed with 128 calls. Seven rounds of 2048 calls are timed per model, rotating model order each round. Times include the ordinary inference wrapper plus benchmark action selection, checksums and abort checks, and exclude between-batch UI yields. Percentiles describe batch-average decision time, not individual-call latency. Browser clock resolution can limit precision; raw batch durations and zero-duration counts are retained.',
  animation: '15 seconds of ordinary Arena playback at 1×, Prism Small vs Quill Plus, seed 902. Frame intervals measure requestAnimationFrame callback cadence, not physical display FPS or touch-to-display latency. Tick lateness is relative to the scheduler deadline; its normal stall-debt limit still applies. Counts separate decisions from intentional 650 ms rally transitions; no adjusted tick-rate estimate is made.',
  checksum: 'FNV-1a 32-bit checksum of ordered selected actions, with action indices encoded as bytes 1–3; a comparison aid, not a cryptographic model identity.',
} as const

export interface SampleSummary {
  count: number
  min: number
  median: number
  p95: number
  max: number
  mean: number
}

export function summarizeSamples(values: readonly number[]): SampleSummary | null {
  if (!values.length) return null
  if (values.some((value) => !Number.isFinite(value) || value < 0))
    throw new Error('Measurement samples must be finite and non-negative.')
  const sorted = [...values].sort((a, b) => a - b)
  const percentile = (fraction: number) => sorted[Math.max(0, Math.ceil(sorted.length * fraction) - 1)]
  return {
    count: sorted.length, min: sorted[0], median: percentile(0.5), p95: percentile(0.95),
    max: sorted[sorted.length - 1], mean: sorted.reduce((sum, value) => sum + value, 0) / sorted.length,
  }
}

export function createBenchmarkInputs() {
  const inputs: Float32Array[] = []
  let seed = DEVICE_TEST.seed
  let frame = serve(seed)
  let loops = new LoopDetector()
  for (let i = 0; i < DEVICE_TEST.inputCount; i++) {
    if (endedMatch(frame)) {
      frame = serve(++seed)
      loops = new LoopDetector()
    } else if (endedRally(frame)) frame = serve(seed, frame)
    const left = observation(frame, 0), right = observation(frame, 1)
    inputs.push(i % 2 === 0 ? left : right)
    frame = advance(frame, [scriptScores('attack', left), scriptScores('defend', right)], loops)
  }
  return inputs
}

export interface InferenceMeasurement {
  id: string
  name: string
  sha256: string
  params: number
  macs: number
  batchAverageDecisionUs: SampleSummary
  batchAveragesUs: number[]
  batchDurationsMs: number[]
  zeroDurationBatches: number
  actionChecksum: string
}

export interface InferenceProgress {
  completed: number
  total: number
  modelName: string
}

type InferModel = (model: Controller, input: Float32Array) => Promise<Scores>

function yieldToUI(signal: AbortSignal) {
  signal.throwIfAborted()
  return new Promise<void>((resolve, reject) => {
    const abort = () => { clearTimeout(timer); reject(signal.reason) }
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', abort)
      resolve()
    }, 0)
    signal.addEventListener('abort', abort, { once: true })
  })
}

export async function benchmarkInference(
  models: Controller[], signal: AbortSignal, onProgress: ((progress: InferenceProgress) => void) | undefined,
  infer: InferModel,
): Promise<InferenceMeasurement[]> {
  signal.throwIfAborted()
  const selected = DEVICE_TEST.modelIds.map((id) => {
    const model = models.find((entry) => entry.id === id)
    if (!model || model.type !== 'weights' || model.inputSize !== 6 || model.outputSize !== 3 || !model.sha256)
      throw new Error('Device test needs the bundled ' + id + ' model.')
    return model
  })
  const inputs = createBenchmarkInputs()
  const records = selected.map((model) => ({ model, batches: [] as number[], checksum: 2166136261 }))
  for (const { model } of records) {
    for (let i = 0; i < DEVICE_TEST.warmup; i++) {
      signal.throwIfAborted()
      choice(await infer(model, inputs[i % inputs.length]) as [number, number, number])
    }
    await yieldToUI(signal)
  }
  let completed = 0
  for (let round = 0; round < DEVICE_TEST.batches; round++) {
    for (let offset = 0; offset < records.length; offset++) {
      const record = records[(round + offset) % records.length]
      signal.throwIfAborted()
      const started = performance.now()
      for (let i = 0; i < DEVICE_TEST.batchSize; i++) {
        signal.throwIfAborted()
        const action = choice(await infer(record.model, inputs[i % inputs.length]) as [number, number, number])
        record.checksum = Math.imul(record.checksum ^ (action + 1), 16777619) >>> 0
      }
      record.batches.push(performance.now() - started)
      signal.throwIfAborted()
      onProgress?.({ completed: ++completed, total: records.length * DEVICE_TEST.batches, modelName: record.model.name })
      await yieldToUI(signal)
    }
  }
  return records.map(({ model, batches, checksum }) => ({
    id: model.id, name: model.name, sha256: model.sha256!, params: model.params, macs: model.macs,
    batchAverageDecisionUs: summarizeSamples(batches.map((ms) => ms * 1000 / DEVICE_TEST.batchSize))!,
    batchAveragesUs: batches.map((ms) => ms * 1000 / DEVICE_TEST.batchSize),
    batchDurationsMs: batches, zeroDurationBatches: batches.filter((ms) => ms === 0).length,
    actionChecksum: 'fnv1a32:' + checksum.toString(16).padStart(8, '0'),
  }))
}

export interface AnimationMeasurement {
  durationMs: number
  rafCallbacks: number
  frameIntervalsMs: SampleSummary | null
  finalCallbackGapMs: number
  gapsOver50Ms: number
  ticks: number
  decisionTicks: number
  rallyTransitions: number
  decisionDeadlineLatenessMs: SampleSummary | null
  rallyTransitionLatenessMs: SampleSummary | null
}

export function summarizeAnimation(
  frameTimes: readonly number[], ticks: readonly MatchTickTiming[], startedAt: number, endedAt: number,
): AnimationMeasurement {
  const intervals = frameTimes.slice(1).map((at, i) => at - frameTimes[i])
  const measured = ticks.filter((tick) => tick.completedAt >= startedAt && tick.completedAt <= endedAt)
  const decisions = measured.filter((tick) => !tick.previousEndedRally)
  const transitions = measured.filter((tick) => tick.previousEndedRally)
  const lateness = (tick: MatchTickTiming) => Math.max(0, tick.completedAt - tick.scheduledAt)
  return {
    durationMs: endedAt - startedAt, rafCallbacks: frameTimes.length,
    frameIntervalsMs: summarizeSamples(intervals), gapsOver50Ms: intervals.filter((interval) => interval > 50).length,
    finalCallbackGapMs: frameTimes.length ? Math.max(0, endedAt - frameTimes[frameTimes.length - 1]) : endedAt - startedAt,
    ticks: measured.length, decisionTicks: decisions.length, rallyTransitions: transitions.length,
    decisionDeadlineLatenessMs: summarizeSamples(decisions.map(lateness)),
    rallyTransitionLatenessMs: summarizeSamples(transitions.map(lateness)),
  }
}

export function runAnimationSample({ start, stop, signal, ticks }: {
  start: () => void
  stop: () => void
  signal: AbortSignal
  ticks: MatchTickTiming[]
}): Promise<AnimationMeasurement> {
  signal.throwIfAborted()
  return new Promise((resolve, reject) => {
    let raf = 0, timer: ReturnType<typeof setTimeout> | undefined
    let startedAt: number | undefined, finished = false
    const frames: number[] = []
    const finish = (error?: unknown) => {
      if (finished) return
      finished = true
      cancelAnimationFrame(raf)
      clearTimeout(timer)
      signal.removeEventListener('abort', abort)
      const endedAt = performance.now()
      try {
        if (startedAt !== undefined) stop()
        if (error !== undefined) throw error
        if (startedAt === undefined || frames.length < 2 || !ticks.some((tick) => tick.completedAt >= startedAt!))
          throw new Error('The animated test did not record game ticks. Please run it again.')
        resolve(summarizeAnimation(frames, ticks, startedAt, endedAt))
      } catch (failure) { reject(failure) }
    }
    const abort = () => finish(signal.reason ?? new DOMException('Device test cancelled.', 'AbortError'))
    const frame = (at: number) => {
      if (finished) return
      frames.push(at)
      if (startedAt === undefined) {
        startedAt = performance.now()
        try { start() } catch (error) { finish(error); return }
        if (finished) return
        timer = setTimeout(() => finish(), DEVICE_TEST.animationMs)
      }
      raf = requestAnimationFrame(frame)
    }
    signal.addEventListener('abort', abort, { once: true })
    raf = requestAnimationFrame(frame)
  })
}

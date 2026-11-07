import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import {
  DEVICE_TEST, benchmarkInference, createBenchmarkInputs, runAnimationSample, summarizeAnimation, summarizeSamples,
} from '../src/deviceBenchmark'
import type { Controller } from '../src/models'
import { validateArenaModel } from '../src/validate'

test('device measurements cover every shipped model and a playable match', () => {
  assert.equal(DEVICE_TEST.version, 'pong-device')
  assert.deepEqual(DEVICE_TEST.modelIds, ['quill', 'small', 'large', 'follow'])
  assert.deepEqual(DEVICE_TEST.matchIds, ['large', 'quill'])
  for (const id of new Set([...DEVICE_TEST.modelIds, ...DEVICE_TEST.matchIds])) {
    const stats = validateArenaModel(readFileSync(new URL(`../public/models/${id}.onnx`, import.meta.url)))
    assert.deepEqual([stats.inputSize, stats.outputSize], [6, 3])
    assert.ok(stats.architecture.widths, `${id} must use the active dense-network contract`)
  }
})

test('benchmark percentiles describe measured batch averages without inventing single-call precision', () => {
  assert.equal(summarizeSamples([]), null)
  assert.deepEqual(summarizeSamples([8, 1, 4, 2, 3]), { count: 5, min: 1, median: 3, p95: 8, max: 8, mean: 3.6 })
  assert.equal(summarizeSamples(Array.from({ length: 100 }, (_, i) => i + 1))!.p95, 95)
  assert.throws(() => summarizeSamples([1, NaN]), /finite/)
  assert.throws(() => summarizeSamples([-1]), /non-negative/)
})

test('the workload repeats the same legal float32 observations across runs', () => {
  const inputs = createBenchmarkInputs()
  assert.equal(inputs.length, DEVICE_TEST.inputCount)
  assert.deepEqual(inputs, createBenchmarkInputs())
  assert.ok(inputs.every((input) => input instanceof Float32Array && input.length === 6 && input.every(Number.isFinite)))
  assert.ok(inputs.some((input) => input[2] < 0) && inputs.some((input) => input[2] > 0), 'Both incoming and outgoing balls are measured')
  assert.ok(inputs.some((input) => Math.round(Math.abs(input[2]) * 6) === 13), 'The workload reaches the fast-ball phase')
})

test('animation reports intentional rally transitions separately from ordinary decision deadlines', () => {
  const report = summarizeAnimation([0, 16, 33, 100], [
    { tick: 0, completedAt: -1, scheduledAt: -4, previousEndedRally: false },
    { tick: 1, completedAt: 36, scheduledAt: 33, previousEndedRally: false },
    { tick: 2, completedAt: 70, scheduledAt: 66, previousEndedRally: false },
    { tick: 3, completedAt: 724, scheduledAt: 720, previousEndedRally: true },
    { tick: 4, completedAt: 800, scheduledAt: 790, previousEndedRally: false },
  ], 0, 750)
  assert.equal(report.ticks, 3)
  assert.equal(report.decisionTicks, 2)
  assert.equal(report.rallyTransitions, 1)
  assert.equal(report.decisionDeadlineLatenessMs!.mean, 3.5)
  assert.equal(report.rallyTransitionLatenessMs!.median, 4, 'The scheduled 650 ms pause is not a late decision')
  assert.equal(report.gapsOver50Ms, 1)
  assert.equal(report.frameIntervalsMs!.median, 17)
  assert.equal(report.finalCallbackGapMs, 650, 'The final quiet period remains visible even if the sample ends before another callback')
})

const models: Controller[] = DEVICE_TEST.modelIds.map((id) => ({
  type: 'weights', id, name: id, kind: 'test', description: '', params: 1, macs: 1,
  inputSize: 6, outputSize: 3, sha256: id.padEnd(64, '0'),
}))

test('inference has a fixed per-model workload and checksums selected actions, not elapsed time', async () => {
  const counts = new Map<string, number>()
  const progress: number[] = []
  const measured = await benchmarkInference(models, new AbortController().signal, (event) => progress.push(event.completed),
    async (model, input) => {
      counts.set(model.id, (counts.get(model.id) ?? 0) + 1)
      return input[2] < 0 ? [1, 0, 0] : [0, 0, 1]
    })
  assert.deepEqual(measured.map((model) => model.id), models.map((model) => model.id), 'Every shipped model gets measured')
  for (const model of measured) {
    assert.equal(counts.get(model.id), DEVICE_TEST.warmup + DEVICE_TEST.batches * DEVICE_TEST.batchSize)
    assert.equal(model.batchAverageDecisionUs.count, DEVICE_TEST.batches)
    assert.equal(model.batchDurationsMs.length, DEVICE_TEST.batches)
    assert.equal(model.zeroDurationBatches, model.batchDurationsMs.filter((ms) => ms === 0).length)
    assert.equal(model.sha256, models.find((entry) => entry.id === model.id)!.sha256)
    assert.match(model.actionChecksum, /^fnv1a32:[a-f0-9]{8}$/)
  }
  assert.equal(new Set(measured.map((model) => model.actionChecksum)).size, 1, 'The same decisions produce the same checksum despite timing differences')
  assert.deepEqual(progress, Array.from({ length: models.length * DEVICE_TEST.batches }, (_, i) => i + 1))
})

test('cancelling inference stops work instead of producing a partial successful report', async () => {
  const abort = new AbortController()
  let calls = 0
  await assert.rejects(benchmarkInference(models, abort.signal, undefined, async () => {
    if (++calls === 10) abort.abort(new Error('Stopped for test'))
    return [0, 1, 0]
  }), /Stopped for test/)
  assert.equal(calls, 10)
  assert.throws(() => runAnimationSample({ start: () => assert.fail(), stop: () => assert.fail(), signal: abort.signal, ticks: [] }), /Stopped for test/)
})

test('animation cancellation and failed starts stop playback and leave no scheduled work', async () => {
  const originalRAF = globalThis.requestAnimationFrame
  const originalCancelRAF = globalThis.cancelAnimationFrame
  const originalTimeout = globalThis.setTimeout
  const originalClearTimeout = globalThis.clearTimeout
  let next = 0
  const frames = new Map<number, FrameRequestCallback>()
  const timers = new Set<number>()
  globalThis.requestAnimationFrame = (callback) => { const id = ++next; frames.set(id, callback); return id }
  globalThis.cancelAnimationFrame = (id) => { frames.delete(id) }
  globalThis.setTimeout = ((_: () => void) => { const id = ++next; timers.add(id); return id }) as typeof setTimeout
  globalThis.clearTimeout = ((id: number) => { timers.delete(id) }) as typeof clearTimeout
  try {
    for (const mode of ['abort while running', 'abort during start', 'failed start'] as const) {
      const abort = new AbortController()
      let starts = 0, stops = 0
      const result = runAnimationSample({
        start: () => {
          starts++
          if (mode === 'abort during start') abort.abort(new Error(mode))
          if (mode === 'failed start') throw new Error(mode)
        },
        stop: () => { stops++ }, signal: abort.signal, ticks: [],
      })
      const [id, callback] = [...frames][0]
      frames.delete(id)
      callback(performance.now())
      if (mode === 'abort while running') {
        assert.equal(timers.size, 1)
        assert.equal(frames.size, 1)
        abort.abort(new Error(mode))
      }
      await assert.rejects(result, new RegExp(mode))
      assert.equal(starts, 1)
      assert.equal(stops, 1)
      assert.equal(frames.size, 0, mode + ' must cancel the animation callback')
      assert.equal(timers.size, 0, mode + ' must clear the finish timeout')
    }
  } finally {
    globalThis.requestAnimationFrame = originalRAF
    globalThis.cancelAnimationFrame = originalCancelRAF
    globalThis.setTimeout = originalTimeout
    globalThis.clearTimeout = originalClearTimeout
  }
})

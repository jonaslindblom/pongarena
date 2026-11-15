import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import pb from 'onnx-proto'
import * as ort from 'onnxruntime-web/wasm'
import { choice, observation, serve, type Frame, type Side, type Scores } from '../src/game'
import { validateArenaModel } from '../src/validate'

const bytes = readFileSync(new URL('../public/models/follow.onnx', import.meta.url))

test('Follow counts all stored values even though only five drive its tracking policy', () => {
  const stats = validateArenaModel(bytes)
  assert.deepEqual([stats.params, stats.macs, stats.nodes], [13, 9, 3])
  assert.deepEqual(stats.architecture.widths, [6, 1, 3])
  const model = pb.onnx.ModelProto.decode(bytes)
  let nonzero = 0
  for (const tensor of model.graph!.initializer!) {
    const raw = tensor.rawData!
    const values = new DataView(raw.buffer, raw.byteOffset, raw.byteLength)
    for (let i = 0; i < raw.length; i += 4) if (values.getFloat32(i, true) !== 0) nonzero++
  }
  assert.equal(nonzero, 5, 'Unused inputs and zero biases still occupy the counted dense matrices')
})

test('Follow tracks the current ball position on both sides through the actual WASM network', async () => {
  ort.env.wasm.numThreads = 1
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
  const scores = async (changes: Partial<Frame>, side: Side = 0) => {
    const input = observation({ ...serve(1), ...changes }, side)
    const result = await session.run({ observation: new ort.Tensor('float32', input, [1, 6]) })
    return Array.from(result.action_scores.data as Float32Array) as Scores
  }
  try {
    for (const side of [0, 1] as const) {
      for (const [offset, action] of [[-10, 0], [-1.6, 0], [-1.4, 1], [0, 1], [1.4, 1], [1.6, 2], [10, 2]]) {
        assert.equal(choice(await scores({ y: 80 + offset, paddles: [80, 80] }, side)), action,
          'Move toward the ball, but hold inside the 1.5-unit deadband to avoid overshooting')
      }
      const paddles: [number, number] = side === 0 ? [110, 20] : [20, 110]
      assert.equal(choice(await scores({ y: 100, paddles }, side)), 0,
        'Compare against our own paddle rather than the opponent or court center')
    }
    const baseline = await scores({ x: 180, y: 100, vx: -5, vy: -4, paddles: [80, 20] })
    assert.deepEqual(await scores({ x: 16, y: 100, vx: 13, vy: 9, paddles: [80, 140] }), baseline,
      'Distance, velocity and opponent position are unused: Follow tracks rather than predicts')
    assert.equal(baseline[1], Math.fround(Math.tanh(1.5 / 80)),
      'The constant stay score sets the documented deadband after Tanh')
    assert.ok(Math.abs(baseline[2] - Math.tanh(20 / 80)) < 1e-6,
      'The displayed hidden value is the bounded current position error')
    assert.equal(baseline[0], -baseline[2])
  } finally {
    await session.release()
  }
})

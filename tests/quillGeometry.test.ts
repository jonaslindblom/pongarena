import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import * as ort from 'onnxruntime-web/wasm'
import { choice, observation, serve, type Frame, type Scores } from '../src/game'
import { validateArenaModel } from '../src/validate'
import { smallestDivision } from '../src/competition'

const bytes = readFileSync(new URL('../public/models/quill_geometry.onnx', import.meta.url))

test('Quill Geometry fits the Large contract with all geometric operations in dense weights', () => {
  const stats = validateArenaModel(bytes)
  assert.deepEqual([stats.params, stats.macs, stats.nodes], [41430, 40182, 9])
  assert.deepEqual(stats.architecture.widths, [6, 43, 729, 6, 467, 3])
  assert.equal(smallestDivision(stats), 'large')
})

test('Geometry anticipates slow flights and recovers using the opponent landing through WASM', async () => {
  ort.env.wasm.numThreads = 1
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
  const action = async (changes: Partial<Frame>) => {
    const input = observation({ ...serve(1), ...changes }, 0)
    const result = await session.run({ observation: new ort.Tensor('float32', input, [1, 6]) })
    return choice(Array.from(result.action_scores.data as Float32Array) as Scores)
  }
  try {
    assert.equal(await action({ x: 180, y: 80, vx: -5, vy: 1, paddles: [80, 80] }), 2,
      'Account for the slow serve flight instead of tracking current ball height')
    assert.equal(await action({ x: 340, y: 4, vx: 13, vy: 0, paddles: [80, 80] }), 2,
      'A near-wall opponent contact constrains its possible returns: recover below center')
    assert.equal(await action({ x: 340, y: 156, vx: 13, vy: 0, paddles: [80, 80] }), 0,
      'Mirror recovery when the opponent contact is near the opposite wall')
    assert.equal(await action({ x: 340, y: 80, vx: 13, vy: 0, paddles: [80, 80] }), 1,
      'Stay centered when the opponent retains a symmetric return range')
    assert.equal(await action({ x: 20, y: 4, vx: -13, vy: 0, paddles: [80, 80] }), 0,
      'Disable opponent recovery and intercept while the ball approaches')
  } finally {
    await session.release()
  }
})

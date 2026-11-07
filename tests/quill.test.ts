import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import * as ort from 'onnxruntime-web/wasm'
import { choice, observation, serve, type Frame, type Scores } from '../src/game'
import { validateArenaModel } from '../src/validate'
import { smallestDivision } from '../src/competition'

const bytes = readFileSync(new URL('../public/models/quill.onnx', import.meta.url))

test('Quill expresses its policy in a Small dense network with 163 parameters', () => {
  const stats = validateArenaModel(bytes)
  assert.deepEqual([stats.params, stats.macs, stats.nodes], [163, 141, 7])
  assert.deepEqual(stats.architecture.widths, [6, 12, 4, 3, 3])
  assert.equal(smallestDivision(stats), 'small')
})

test('Quill predicts, aims and recovers through the actual WASM network', async () => {
  ort.env.wasm.numThreads = 1
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
  const action = async (changes: Partial<Frame>) => {
    const input = observation({ ...serve(1), ...changes }, 0)
    const result = await session.run({ observation: new ort.Tensor('float32', input, [1, 6]) })
    return choice(Array.from(result.action_scores.data as Float32Array) as Scores)
  }
  try {
    assert.equal(await action({ x: 80, y: 80, vx: -13, vy: 6, paddles: [80, 80] }), 2,
      'Anticipate the landing rather than follow the current ball height')
    assert.equal(await action({ x: 180, y: 140, vx: -13, vy: 6, paddles: [100, 80] }), 0,
      'A predicted wall bounce must reverse the intercept direction')
    const contact = { x: 16, y: 80, vx: -13, vy: 0 }
    assert.equal(await action({ ...contact, paddles: [80, 60] }), 0,
      'Offset contact to aim away from an opponent above center')
    assert.equal(await action({ ...contact, paddles: [80, 100] }), 2,
      'Reverse shot placement when the opponent crosses the center')
    assert.equal(await action({ ...contact, paddles: [85, 100] }), 1,
      'A small movement deadband avoids overshooting an already reached target')
    assert.equal(await action({ x: 180, y: 140, vx: 13, vy: 9, paddles: [140, 100] }), 0,
      'Recover while the ball travels away')
    for (const other of [15, 60, 79, 80, 100, 145]) {
      assert.equal(await action({ x: 180, y: 140, vx: 13, vy: 9, paddles: [80, other] }), 1,
        'At maximum speed, center recovery must not depend on the opponent side')
    }
    assert.equal(await action({ x: 180, y: 80, vx: 5, vy: 0, paddles: [80, 100] }), 2,
      'Slower returns leave time for an opponent-dependent recovery position')
  } finally {
    await session.release()
  }
})

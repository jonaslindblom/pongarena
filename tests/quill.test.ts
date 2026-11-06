import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import * as ort from 'onnxruntime-web/wasm'
import { modelFile } from './modelFiles'
import { choice, observation, serve, type Frame, type Scores } from '../src/game'
import { validateArenaModel } from '../src/validate'

const bytes = readFileSync(modelFile('quill-nano-v4'))

test('Quill preserves the exact handwritten budget and the evaluated model identity', () => {
  const stats = validateArenaModel(bytes)
  assert.deepEqual([stats.params, stats.macs, stats.nodes], [163, 141, 7])
  assert.deepEqual(stats.architecture.widths, [6, 12, 4, 3, 3])
  assert.equal(createHash('sha256').update(bytes).digest('hex'),
    '5442d7fb451b6f6e3abc520d2943f774cac119586164ffd6da0b1a95b228cb17',
    'Changing the published weights requires fresh performance evidence')
})

test('Quill anticipates returns, places shots and recovers using the actual WASM network', async () => {
  ort.env.wasm.numThreads = 1
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
  const action = async (changes: Partial<Frame>) => {
    const input = observation({ ...serve(1), ...changes }, 0)
    const output = await session.run({ observation: new ort.Tensor('float32', input, [1, 6]) })
    return choice(Array.from(output.action_scores.data) as Scores)
  }
  try {
    assert.equal(await action({ x: 80, y: 80, vx: -13, vy: 6, paddles: [80, 80] }), 2,
      'Move toward the future landing even when currently aligned with the ball')
    assert.equal(await action({ x: 180, y: 140, vx: -13, vy: 6, paddles: [100, 80] }), 0,
      'A wall bounce reverses the intercept direction relative to tracking current height')
    const contact = { x: 16, y: 80, vx: -13, vy: 0 }
    assert.equal(await action({ ...contact, paddles: [80, 60] }), 0,
      'Offset the paddle to send the ball away from an opponent above center')
    assert.equal(await action({ ...contact, paddles: [80, 100] }), 2,
      'Reverse shot placement when the opponent moves below center')
    assert.equal(await action({ x: 180, y: 140, vx: 13, vy: 9, paddles: [140, 100] }), 0,
      'Recover toward center while the ball travels away')
    assert.equal(await action({ ...contact, paddles: [85, 100] }), 1,
      'The movement deadband prevents overshooting an already reached contact target')
  } finally {
    await session.release()
  }
})

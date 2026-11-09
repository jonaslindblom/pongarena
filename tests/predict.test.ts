import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import * as ort from 'onnxruntime-web/wasm'
import { choice, observation, serve, type Frame, type Scores } from '../src/game'
import { sha256, smallestDivision } from '../src/competition'
import { validateArenaModel } from '../src/validate'

const bytes = readFileSync(new URL('../public/models/predict.onnx', import.meta.url))

test('Predict preserves the frozen Small model with 75 parameters and 63 MACs', async () => {
  const stats = validateArenaModel(bytes)
  assert.deepEqual([stats.params, stats.macs, stats.nodes], [75, 63, 5])
  assert.deepEqual(stats.architecture.widths, [6, 6, 3, 3])
  assert.equal(smallestDivision(stats), 'small')
  assert.equal(await sha256(bytes), '0d405fe22a0a5f568c1d23d76356893e12aae1ef1c9a7f26150e8f35520b3cad')
})

test('Predict anticipates and reflects through its actual WASM network', async () => {
  ort.env.wasm.numThreads = 1
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
  const scores = async (changes: Partial<Frame>): Promise<Scores> => {
    const input = observation({ ...serve(1), ...changes }, 0)
    const result = await session.run({ observation: new ort.Tensor('float32', input, [1, 6]) })
    return Array.from(result.action_scores.data as Float32Array) as Scores
  }
  const action = async (changes: Partial<Frame>) => choice(await scores(changes))
  try {
    assert.equal(await action({ x: 80, y: 80, vx: -13, vy: 6, paddles: [80, 80] }), 2,
      'Anticipate the incoming ball instead of staying at its current height')
    assert.equal(await action({ x: 180, y: 140, vx: -13, vy: 6, paddles: [100, 80] }), 0,
      'A predicted wall bounce must reverse the intercept direction')
    assert.equal(await action({ x: 16, y: 80, vx: -13, vy: 0, paddles: [81, 100] }), 1,
      'The movement deadband avoids overshooting an almost reached target')
    const contact = { x: 16, y: 80, vx: -13, vy: 0 }
    assert.deepEqual(await scores({ ...contact, paddles: [80, 60] }),
      await scores({ ...contact, paddles: [80, 100] }),
      'This compact candidate deliberately has no opponent-dependent shot placement')
    const outgoing = { x: 180, y: 140, vx: 13, vy: 0, paddles: [80, 100] }
    assert.equal(await action(outgoing), 2,
      'Known limitation: follow the target while departing rather than recover to center')
    assert.deepEqual(await scores(outgoing), await scores({ ...outgoing, vx: -13 }),
      'The forecast deliberately ignores horizontal direction and speed')
  } finally {
    await session.release()
  }
})

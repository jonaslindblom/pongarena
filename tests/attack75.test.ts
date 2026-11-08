import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import * as ort from 'onnxruntime-web/wasm'
import { advance, choice, endedRally, observation, serve, type Frame, type Pair, type Scores, type Side } from '../src/game'
import { sha256, smallestDivision } from '../src/competition'
import { validateArenaModel } from '../src/validate'

const bytes = readFileSync(new URL('../public/models/attack75.onnx', import.meta.url))

test('Attack remains a separate Small model with Predict’s 75-parameter layout', async () => {
  const stats = validateArenaModel(bytes)
  assert.deepEqual([stats.params, stats.macs, stats.nodes], [75, 63, 5])
  assert.deepEqual(stats.architecture.widths, [6, 6, 3, 3])
  assert.equal(smallestDivision(stats), 'small')
  assert.equal(await sha256(bytes), '646d1a1365e3e307bd4431fc73930c5efa19de16b7cc22bbb9200608c2b720f8')
  assert.equal(await sha256(readFileSync(new URL('../public/models/predict75.onnx', import.meta.url))),
    '0d405fe22a0a5f568c1d23d76356893e12aae1ef1c9a7f26150e8f35520b3cad',
    'Adding an aggressive fork must preserve the original controller')
})

test('Attack anticipates bounces and changes shot placement with the opponent in WASM', async () => {
  ort.env.wasm.numThreads = 1
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
  const action = async (changes: Partial<Frame>) => {
    const result = await session.run({ observation: new ort.Tensor('float32',
      observation({ ...serve(1), ...changes }, 0), [1, 6]) })
    return choice(Array.from(result.action_scores.data as Float32Array) as Scores)
  }
  try {
    assert.equal(await action({ x: 80, y: 80, vx: -13, vy: 6, paddles: [80, 80] }), 2,
      'Start intercepting before the incoming ball reaches our height')
    assert.equal(await action({ x: 180, y: 140, vx: -13, vy: 6, paddles: [120, 80] }), 0,
      'Reflect the landing forecast instead of chasing the ball toward the wall')
    assert.equal(await action({ x: 180, y: 20, vx: -13, vy: -6, paddles: [40, 80] }), 2,
      'The opposite wall must receive the same defensive treatment')
    const contact = { x: 16, y: 80, vx: -13, vy: 0 }
    assert.equal(await action({ ...contact, paddles: [80, 60] }), 0,
      'Move above the landing point to send the return below an opponent above center')
    assert.equal(await action({ ...contact, paddles: [80, 100] }), 2,
      'Reverse the intended corner when the opponent changes sides')
    assert.equal(await action({ ...contact, paddles: [85, 100] }), 1,
      'Stop at the offset target rather than jittering across it')
  } finally {
    await session.release()
  }
})

test('Attack’s aiming produces angled corner returns through real collision physics on both sides', async () => {
  ort.env.wasm.numThreads = 1
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
  try {
    for (const side of [0, 1] as const) {
      for (const [height, opponent, corner] of [[80, 40, 'high'], [80, 120, 'low'],
        [40, 80, 'high'], [120, 80, 'low']] as const) {
        const other = (1 - side) as Side
        const paddles: Pair<number> = [height, height]
        paddles[other] = opponent
        let frame: Frame = { ...serve(1), x: side === 0 ? 80 : 280, y: height,
          vx: side === 0 ? -13 : 13, vy: 0, paddles }
        let returnVelocity: number | undefined
        for (let tick = 0; tick < 60 && !endedRally(frame); tick++) {
          const scores: Pair<Scores> = [[0, 1, 0], [0, 1, 0]]
          if (returnVelocity === undefined) {
            const result = await session.run({ observation: new ort.Tensor('float32', observation(frame, side), [1, 6]) })
            scores[side] = Array.from(result.action_scores.data as Float32Array) as Scores
          }
          frame = advance(frame, scores)
          if (frame.event === 'return' && frame.actor === side) returnVelocity = frame.vy
        }
        assert(returnVelocity !== undefined && Math.abs(returnVelocity) >= 2,
          'Moving off center must create a real angled return, not merely different action scores')
        assert.equal(frame.points[side], 1, 'The shot should pass the stationary opponent')
        assert(corner === 'high' ? frame.y >= 132 : frame.y <= 28,
          'Choose the open outer 24 units of court, including when a centered opponent requires a cross-court tie-break')
      }
    }
  } finally {
    await session.release()
  }
})

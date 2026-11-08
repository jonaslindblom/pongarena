import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import * as ort from 'onnxruntime-web/wasm'
import { advance, choice, endedRally, observation, serve, type Frame, type Pair, type Scores, type Side } from '../src/game'
import { sha256, smallestDivision } from '../src/competition'
import { validateArenaModel } from '../src/validate'

const bytes = readFileSync(new URL('../public/models/counter97.onnx', import.meta.url))

test('Counter adds recovery within 100 parameters while preserving the existing handwritten models', async () => {
  const stats = validateArenaModel(bytes)
  assert.deepEqual([stats.params, stats.macs, stats.nodes], [97, 81, 7])
  assert.deepEqual(stats.architecture.widths, [6, 6, 4, 3, 3])
  assert.equal(smallestDivision(stats), 'small')
  assert.equal(await sha256(bytes), '5f7a480b98c982cf403f2f8849fac1e5e02503ff12e6b0b5d1e377856534e35c')
  for (const [id, hash] of [
    ['predict75', '0d405fe22a0a5f568c1d23d76356893e12aae1ef1c9a7f26150e8f35520b3cad'],
    ['attack75', '646d1a1365e3e307bd4431fc73930c5efa19de16b7cc22bbb9200608c2b720f8'],
  ]) {
    assert.equal(await sha256(readFileSync(new URL(`../public/models/${id}.onnx`, import.meta.url))), hash,
      'The stronger model must remain a separate addition, preserving both earlier controllers')
  }
})

test('Counter anticipates incoming shots, reflects walls and selects the open corner in WASM', async () => {
  ort.env.wasm.numThreads = 1
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
  const action = async (changes: Partial<Frame>) => {
    const result = await session.run({ observation: new ort.Tensor('float32',
      observation({ ...serve(1), ...changes }, 0), [1, 6]) })
    return choice(Array.from(result.action_scores.data as Float32Array) as Scores)
  }
  try {
    assert.equal(await action({ x: 80, y: 80, vx: -13, vy: 6, paddles: [80, 80] }), 2,
      'Cover the incoming landing before the ball reaches our paddle height')
    assert.equal(await action({ x: 200, y: 80, vx: -6, vy: 2, paddles: [110, 80] }), 2,
      'The longer forecast must cover a distant cross-court return instead of retreating toward the ball')
    assert.equal(await action({ x: 180, y: 140, vx: -13, vy: 6, paddles: [120, 80] }), 0,
      'A wall reflection must reverse the interception direction')
    assert.equal(await action({ x: 180, y: 20, vx: -13, vy: -6, paddles: [40, 80] }), 2,
      'Apply the same defensive reflection at the opposite wall')
    const contact = { x: 16, y: 80, vx: -13, vy: 0 }
    assert.equal(await action({ ...contact, paddles: [80, 60] }), 0,
      'Offset the paddle to send the return away from an opponent above center')
    assert.equal(await action({ ...contact, paddles: [80, 100] }), 2,
      'Reverse the intended corner when the opponent crosses center')
    assert.equal(await action({ ...contact, paddles: [85, 100] }), 1,
      'Keep the reached shot offset steady rather than overshooting it')
  } finally {
    await session.release()
  }
})

test('Counter recovers toward center on both sides without chasing an outgoing ball', async () => {
  ort.env.wasm.numThreads = 1
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
  const scores = async (frame: Frame, side: Side): Promise<Scores> => {
    const result = await session.run({ observation: new ort.Tensor('float32', observation(frame, side), [1, 6]) })
    return Array.from(result.action_scores.data as Float32Array) as Scores
  }
  try {
    for (const side of [0, 1] as const) {
      for (const own of [15, 80, 145]) {
        const referencePaddles: Pair<number> = [80, 80]
        referencePaddles[side] = own
        const reference = await scores({ ...serve(1), vx: side === 0 ? 13 : -13,
          vy: 0, paddles: referencePaddles }, side)
        for (const x of [15, 345]) for (const y of [4, 156]) {
          for (const vy of [-9, 9]) for (const opponent of [15, 80, 145]) {
            const paddles: Pair<number> = [opponent, opponent]
            paddles[side] = own
            const frame = { ...serve(1), x, y, vx: side === 0 ? 13 : -13, vy, paddles }
            const result = await scores(frame, side)
            assert.equal(choice(result), own < 80 ? 2 : own > 80 ? 0 : 1,
              'Fast departure must bring either edge paddle back and leave a centered paddle still')
            assert(result.every((score, i) => Math.abs(score - reference[i]) < 1e-5),
              'At maximum speed, neither the ball trajectory nor opponent position may pull recovery off center')
          }
        }
      }
      for (const speed of [5, 6, 7, 8, 9, 10, 11, 12]) {
        for (const y of [4, 156]) for (const vy of [-9, 9]) for (const opponent of [15, 80, 145]) {
          const paddles: Pair<number> = [opponent, opponent]
          paddles[side] = 80
          const result = await scores({ ...serve(1), y, vy, vx: side === 0 ? speed : -speed, paddles }, side)
          // Movement logits are signed target error in normalized court units.
          // A small opponent-dependent recovery offset is permitted at slow speeds.
          const recoveryOffset = 80 * (result[2] - result[0]) / 2
          assert(Math.abs(recoveryOffset) <= 3 + 1e-3,
            'Slow departure must keep the target within three units of center, preserving room to defend either corner')
        }
      }
    }
  } finally {
    await session.release()
  }
})

test('Counter still creates real angled corner returns after adding recovery', async () => {
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
          'The recovery gate must not suppress opponent-aware aiming on approach')
        assert.equal(frame.points[side], 1, 'The angled return should pass the stationary opponent')
        assert(corner === 'high' ? frame.y >= 132 : frame.y <= 28,
          'Preserve placement in the open outer 24 units, including a cross-court choice against a centered opponent')
      }
    }
  } finally {
    await session.release()
  }
})

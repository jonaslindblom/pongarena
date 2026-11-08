import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import * as ort from 'onnxruntime-web/wasm'
import { choice, observation, serve, type Frame, type Pair, type Scores, type Side } from '../src/game'
import { smallestDivision } from '../src/competition'
import { validateArenaModel } from '../src/validate'

const speedBytes = readFileSync(new URL('../public/models/quill_speed.onnx', import.meta.url))
const originalBytes = readFileSync(new URL('../public/models/quill.onnx', import.meta.url))

test('Quill Speed adds a bounded flight correction within the Small budget', () => {
  const stats = validateArenaModel(speedBytes)
  assert.deepEqual([stats.params, stats.macs, stats.nodes], [229, 201, 7])
  assert.deepEqual(stats.architecture.widths, [6, 18, 4, 3, 3])
  assert.equal(smallestDivision(stats), 'small')
})

async function sessions() {
  ort.env.wasm.numThreads = 1
  return Promise.all([originalBytes, speedBytes].map(bytes =>
    ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })))
}

async function scores(session: ort.InferenceSession, frame: Frame, side: Side): Promise<Scores> {
  const result = await session.run({ observation: new ort.Tensor('float32',
    observation(frame, side), [1, 6]) })
  const output = Array.from(result.action_scores.data as Float32Array) as Scores
  assert(output.every(Number.isFinite))
  return output
}

test('Quill Speed anticipates slow cross-court landings in either direction and on either side', async () => {
  const [original, speed] = await sessions()
  try {
    for (const side of [0, 1] as const) {
      for (const [verticalVelocity, own, expected] of [[2, 120, 2], [-2, 40, 0]] as const) {
        const paddles: Pair<number> = [80, 80]
        paddles[side] = own
        const frame: Frame = { ...serve(1), x: side === 0 ? 200 : 160,
          y: 80, vx: side === 0 ? -6 : 6, vy: verticalVelocity, paddles }
        // The contact plane is 185 units away: ceil(185/6)=31 ticks.
        // With no wall crossing, the actual landing is 142 or 18.
        const landing = 80 + 31 * verticalVelocity
        assert(expected === 2 ? landing > own : landing < own)
        assert.equal(choice(await scores(speed, frame, side)), expected,
          'The slower-flight correction should move toward the geometric landing')
        assert.equal(choice(await scores(original, frame, side)), 2 - expected,
          'This probe captures the original fixed-speed forecast moving away from that landing')
      }
    }
  } finally {
    await Promise.all([original.release(), speed.release()])
  }
})

test('The speed correction vanishes at maximum speed, zero vertical velocity and contact', async () => {
  const [original, speed] = await sessions()
  const probes: Partial<Frame>[] = []
  for (const x of [15, 80, 180, 345]) {
    for (const y of [4, 80, 156]) {
      for (const vy of [-9, 0, 9]) {
        probes.push({ x, y, vx: -13, vy, paddles: [110, 60] })
      }
    }
  }
  for (const vx of [-5, -6, -9, -12]) {
    probes.push({ x: 200, y: 80, vx, vy: 0, paddles: [110, 60] })
    probes.push({ x: 15, y: 80, vx, vy: 3, paddles: [110, 60] })
  }
  try {
    for (const changes of probes) {
      const frame = { ...serve(1), ...changes }
      const baseline = await scores(original, frame, 0)
      const corrected = await scores(speed, frame, 0)
      assert(corrected.every((score, index) => Math.abs(score - baseline[index]) < 1e-5),
        `A zero correction must retain Quill's scores for ${JSON.stringify(changes)}`)
      assert.equal(choice(corrected), choice(baseline))
    }
  } finally {
    await Promise.all([original.release(), speed.release()])
  }
})

test('Quill Speed preserves centered recovery while the ball departs at maximum speed', async () => {
  const [original, speed] = await sessions()
  try {
    for (const side of [0, 1] as const) {
      for (const own of [15, 80, 145]) {
        for (const opponent of [15, 80, 145]) {
          const paddles: Pair<number> = [opponent, opponent]
          paddles[side] = own
          const frame: Frame = { ...serve(1), x: 180, y: 156,
            vx: side === 0 ? 13 : -13, vy: 9, paddles }
          const baseline = await scores(original, frame, side)
          const corrected = await scores(speed, frame, side)
          assert.equal(choice(corrected), own < 80 ? 2 : own > 80 ? 0 : 1)
          assert(corrected.every((score, index) => Math.abs(score - baseline[index]) < 1e-5))
        }
      }
    }
  } finally {
    await Promise.all([original.release(), speed.release()])
  }
})

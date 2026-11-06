import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import pb from 'onnx-proto'
import * as ort from 'onnxruntime-web/wasm'
import { modelFile } from './modelFiles'
import { choice, observation, serve, type Frame, type Scores } from '../src/game'
import { validateArenaModel } from '../src/validate'

const original = readFileSync(modelFile('quill-nano-v4'))
const bytes = readFileSync(new URL('../public/models/quill-plus-v4.onnx', import.meta.url))

test('Quill Plus changes only four weights while preserving the exact 163-parameter contract', () => {
  const stats = validateArenaModel(bytes)
  assert.deepEqual([stats.params, stats.macs, stats.nodes], [163, 141, 7])
  assert.deepEqual(stats.architecture.widths, [6, 12, 4, 3, 3])
  assert.equal(createHash('sha256').update(bytes).digest('hex'),
    'd2a4f54c292b1925b8d99c82e86801f987f22566a4e377233230b3b1c5ca5273',
    'Performance claims belong to the frozen evaluated file')
  const before = pb.onnx.ModelProto.decode(original).graph!.initializer!
  const after = pb.onnx.ModelProto.decode(bytes).graph!.initializer!
  const changed: string[] = []
  for (const tensor of after) {
    const prior = before.find(t => t.name === tensor.name)!
    assert.deepEqual(tensor.dims, prior.dims)
    const values = (raw: Uint8Array) => new DataView(raw.buffer, raw.byteOffset, raw.byteLength)
    const a = values(prior.rawData!), b = values(tensor.rawData!)
    for (let i = 0; i < a.byteLength / 4; i++)
      if (a.getFloat32(i * 4, true) !== b.getFloat32(i * 4, true)) changed.push(`${tensor.name}:${i}`)
  }
  assert.deepEqual(changed, ['w1:3', 'w1:7', 'w1:22', 'w1:46'],
    'Only the paired square tails and outgoing recovery gates may differ')
})

test('Quill Plus removes the cap-speed recovery offset without changing incoming contact aiming', async () => {
  ort.env.wasm.numThreads = 1
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
  const action = async (changes: Partial<Frame>) => {
    const input = observation({ ...serve(1), ...changes }, 0)
    const result = await session.run({ observation: new ort.Tensor('float32', input, [1, 6]) })
    return choice(Array.from(result.action_scores.data as Float32Array) as Scores)
  }
  try {
    for (const other of [15, 60, 79, 80, 100, 145]) {
      assert.equal(await action({ x: 180, y: 140, vx: 13, vy: 9, paddles: [80, other] }), 1,
        'At maximum speed, center recovery must be independent of the opponent side')
    }
    const contact = { x: 16, y: 80, vx: -13, vy: 0 }
    assert.equal(await action({ ...contact, paddles: [80, 60] }), 0,
      'Recovery changes must preserve incoming opposite-end shot placement')
    assert.equal(await action({ ...contact, paddles: [80, 100] }), 2)
    assert.equal(await action({ x: 180, y: 80, vx: 5, vy: 0, paddles: [80, 100] }), 2,
      'At lower speeds, recovery retains part of the original opponent-dependent offset')
  } finally {
    await session.release()
  }
})

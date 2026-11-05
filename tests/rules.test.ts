import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import pb from 'onnx-proto'
import { BURST, CLASSIC, type Pair, type Scores } from '../src/rules'
import * as classic from '../src/game'
import { validateModel } from '../src/validate'
import { smallestDivision } from '../src/competition'

const still3: Pair<Scores> = [[0, 1, 0], [0, 1, 0]]
const still5: Pair<Scores> = [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]
const burstUp: Pair<Scores> = [[0, 0, 0, 1, 0], [0, 1, 0, 0, 0]]

test('each rules wrapper exposes the correct ports and preserves the frozen classic behavior', () => {
  assert.deepEqual([CLASSIC.inputSize, CLASSIC.outputSize, BURST.inputSize, BURST.outputSize], [6, 3, 8, 5])
  assert.notEqual(CLASSIC.contractId, BURST.contractId)
  const raw = classic.serve(24)
  const wrapped = CLASSIC.serve(24)
  assert.deepEqual(wrapped, raw)
  assert.deepEqual(CLASSIC.advance(wrapped, still3), classic.advance(raw, still3 as Pair<classic.Scores>))
  assert.deepEqual(CLASSIC.observation(wrapped, 0), classic.observation(raw, 0))
  assert.equal(wrapped.energy, undefined)
  const preview = BURST.serve(24)
  assert.deepEqual({ ...preview, energy: undefined }, { ...wrapped, energy: undefined })
  assert.deepEqual(preview.energy, [96, 96])
})

test('burst observations expose both resources in the observing player perspective', () => {
  const a = { ...BURST.serve(24), x: 30, vx: -7.5,
    paddles: [32, 116] as Pair<number>, energy: [16, 88] as Pair<number> }
  const b = { ...a, x: 330, vx: 7.5, paddles: [116, 32] as Pair<number>, energy: [88, 16] as Pair<number> }
  assert.deepEqual(BURST.observation(a, 0), BURST.observation(b, 1))
  assert.equal(BURST.observation(a, 0)[6], Math.fround(16 / 96))
  assert.equal(BURST.observation(a, 0)[7], Math.fround(88 / 96))
})

test('burst buys legal movement with an observable cost; empty requests cannot recharge themselves', () => {
  const start = BURST.serve(24)
  const boosted = BURST.advance(start, burstUp)
  assert.equal(boosted.paddles[0], 76)
  assert.deepEqual(boosted.energy, [88, 96])
  const recovering = BURST.advance(boosted, still5)
  assert.equal(recovering.energy![0], 89)
  const exhausted = BURST.advance({ ...start, energy: [7, 96] }, burstUp)
  assert.equal(exhausted.paddles[0], 78)
  assert.equal(exhausted.energy![0], 7)
  const atWall = BURST.advance({ ...start, paddles: [15, 80] }, burstUp)
  assert.equal(atWall.paddles[0], 15)
  assert.equal(atWall.energy![0], 88, 'Requested burst is spent even when the wall clamps motion')
  assert.deepEqual(BURST.serve(24, { ...boosted, event: 'point' }).energy, [96, 96])
})

test('half-step burst pacing preserves gentle openings without changing classic return speeds', () => {
  const burstFrame = { ...BURST.serve(24), x: 20, y: 80, vx: -5, vy: 0 }
  assert.equal(BURST.advance(burstFrame, still5).vx, 5.5)
  const classicFrame = { ...CLASSIC.serve(24), x: 20, y: 80, vx: -5, vy: 0 }
  assert.equal(CLASSIC.advance(classicFrame, still3).vx, 6)
  assert.equal(BURST.advance({ ...burstFrame, vx: -13 }, still5).vx, 13)
})

test('a repeated physical position cannot draw a burst rally while either resource differs', () => {
  const detector = BURST.detector()
  const incoming = { ...BURST.serve(24), x: 20, y: 80, vx: -13, vy: 0 }
  assert.equal(BURST.advance(incoming, still5, detector).event, 'return')
  assert.equal(BURST.advance({ ...incoming, energy: [80, 96] }, still5, detector).event, 'return')
  assert.equal(BURST.advance({ ...incoming, energy: [96, 80] }, still5, detector).event, 'return')
  assert.equal(BURST.advance(incoming, still5, detector).drawReason, 'loop')
})

test('size-study downloads are the evaluated plain eight-input/five-output weights in their stated classes', () => {
  const manifest = JSON.parse(readFileSync(new URL('../training/results/burst-study/models/release/manifest.json', import.meta.url), 'utf8'))
  for (const [name, widths, params, macs, division] of [
    ['micro', [8, 16, 5], 229, 208, 'nano'],
    ['tiny', [8, 16, 16, 5], 501, 464, 'nano'],
    ['small', [8, 32, 32, 5], 1509, 1440, 'small'],
    ['medium', [8, 64, 64, 5], 5061, 4928, 'standard'],
    ['large', [8, 128, 128, 5], 18309, 18048, 'standard'],
  ] as const) {
    const bytes = readFileSync(new URL(`../public/models/burst-${name}.onnx`, import.meta.url))
    assert.equal(createHash('sha256').update(bytes).digest('hex'),
      manifest.files[`training/results/burst-study/models/release/${name}/model.onnx`])
    const model = pb.onnx.ModelProto.decode(bytes)
    assert.equal(model.metadataProps.find((p) => p.key === 'rules')?.value, BURST.id)
    const stats = validateModel(bytes)
    assert.deepEqual([stats.inputSize, stats.outputSize, stats.params, stats.macs], [8, 5, params, macs])
    assert.deepEqual(stats.architecture.widths, widths)
    assert.equal(smallestDivision(stats), division)
    assert.deepEqual(stats.architecture.operations.map((node) => node.op),
      name === 'micro' ? ['Gemm', 'Tanh', 'Gemm'] : ['Gemm', 'Tanh', 'Gemm', 'Tanh', 'Gemm'])
  }
})

test('a model cannot mix the classic observation port with burst actions or vice versa', () => {
  const bytes = readFileSync(new URL('../public/models/burst-micro.onnx', import.meta.url))
  const model = pb.onnx.ModelProto.decode(bytes)
  model.graph!.input![0].type!.tensorType!.shape!.dim![1].dimValue = 6
  assert.throws(() => validateModel(pb.onnx.ModelProto.encode(model).finish()), /action_scores/)
  const classicBytes = readFileSync(new URL('../public/models/tracker.onnx', import.meta.url))
  const legacy = pb.onnx.ModelProto.decode(classicBytes)
  legacy.graph!.input![0].type!.tensorType!.shape!.dim![1].dimValue = 8
  assert.throws(() => validateModel(pb.onnx.ModelProto.encode(legacy).finish()), /action_scores/)
})

test('the focused follow-up remains a separately identified Standard model with its evaluated weights', () => {
  const bytes = readFileSync(new URL('../public/models/burst-focus.onnx', import.meta.url))
  const frozen = JSON.parse(readFileSync(new URL('../training/results/burst-study/models/followup-release/manifest.json', import.meta.url), 'utf8'))
  assert.equal(createHash('sha256').update(bytes).digest('hex'), frozen.onnx_sha256)
  const stats = validateModel(bytes)
  assert.deepEqual([stats.inputSize, stats.outputSize, stats.params, stats.macs], [8, 5, 18309, 18048])
  assert.deepEqual(stats.architecture.widths, [8, 128, 128, 5])
  assert.equal(smallestDivision(stats), 'standard')
  assert.deepEqual(stats.architecture.operations.map((node) => node.op), ['Gemm', 'Tanh', 'Gemm', 'Tanh', 'Gemm'])
})

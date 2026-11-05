import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import pb from 'onnx-proto'
import {
  advance,
  choice,
  endedMatch,
  LoopDetector,
  observation,
  RALLIES,
  RALLY_TICK_LIMIT,
  serve,
  type Pair,
  type Scores,
} from '../src/game'
import { DenseModelError, validateArenaModel, validateModel } from '../src/validate'
import { smallestDivision } from '../src/competition'
import { advance as advanceV1, serve as serveV1 } from '../training/v1/game'
import { advance as advanceV2 } from '../training/v2/game'
import { advance as advanceV3, LoopDetector as LoopDetectorV3 } from '../training/v3/game'

const still: Pair<Scores> = [
  [0, 1, 0],
  [0, 1, 0],
]
test('a policy can play either side using the same observation', () => {
  const a = { ...serve(24), x: 24, vx: -5, paddles: [28, 62] as Pair<number> }
  const b = { ...a, x: 360 - a.x, vx: -a.vx, paddles: [62, 28] as Pair<number> }
  assert.deepEqual(observation(a, 0), observation(b, 1))
})
test('opening placement changes direction without an immediate maximum-speed return', () => {
  const before = { ...serve(24), x: 20, y: 80, vy: 0, vx: -5 }
  assert.equal(advance(before, still).vx, 6)
  const edge = advance({ ...before, y: 95 }, still)
  assert.equal(edge.vx, 6)
  assert.equal(edge.vy, 4)
  assert.equal(before.points[0] + before.points[1], 0)
})
test('successful returns build pressure to a cap; a center hit cannot reset it', () => {
  let f = { ...serve(24), x: 20, y: 80, vy: 0, vx: -5 }
  for (let returns = 1; returns <= 12; returns++) {
    f = advance({ ...f, x: 20, vx: -Math.abs(f.vx), y: 80, vy: 0 }, still)
    assert.equal(f.vx, Math.min(13, 5 + returns))
    assert.equal(f.vy, 0)
  }
  const next = serve(24, { ...f, event: 'point' })
  assert.equal(Math.abs(next.vx), 5)
})
test('the next shot depends only on observed state, with the same aiming law on both sides', () => {
  for (const side of [0, 1] as const) for (let speed = 5; speed <= 13; speed++) {
    for (let offset = -15; offset <= 15; offset++) {
      const before = { ...serve(24), x: side === 0 ? 20 : 340, y: 80 + offset,
        vx: (side === 0 ? -1 : 1) * speed, vy: 0 }
      const next = advance(before, still)
      assert.equal(Math.abs(next.vx), Math.min(13, speed + 1))
      assert.equal(next.vy, Math.round(9 * Math.min(13, speed + 1) * offset / 195))
      const counted = advance({ ...before, returns: 100, rallyTick: 700 }, still)
      assert.deepEqual([next.x, next.y, next.vx, next.vy], [counted.x, counted.y, counted.vx, counted.vy])
    }
  }
})
test('v2 softens edge speed while preserving aiming, paddle motion and seeded serves', () => {
  for (const seed of [0, 24, 902, 4294967295]) assert.deepEqual(serve(seed), serveV1(seed))
  for (const side of [0, 1] as const) {
    for (let offset = -15; offset <= 15; offset++) {
      const before = {
        ...serve(24),
        x: side === 0 ? 20 : 340,
        y: 80 + offset,
        vx: side === 0 ? -5 : 5,
        vy: 0,
      }
      const old = advanceV1(before, still)
      const next = advanceV2(before, still)
      assert.deepEqual({ ...next, vx: old.vx }, old)
      assert.ok(Math.abs(next.vx) <= 13)
    }
  }
  const edge = { ...serve(24), x: 20, y: 94, vx: -5, vy: 0 }
  assert.equal(advanceV2(edge, still).vx, 12)
  assert.equal(advanceV1(edge, still).vx, 14)
})
test('five seeded rallies reproduce and a missed ball awards exactly one point', () => {
  let f = serve(902)
  assert.deepEqual(f, serve(902))
  assert.notDeepEqual([f.y, f.vy], [serve(24).y, serve(24).vy])
  for (let rally = 1; rally <= 5; rally++) {
    f = advance({ ...f, x: 17, y: 30, vx: -5, vy: 0 }, still)
    assert.equal(f.event, 'point')
    assert.equal(f.points[1], rally)
    assert.equal(endedMatch(f), rally === 5)
    if (rally < 5) f = serve(902, f)
  }
})
test('both paddles face rising and falling serves across five distinct height bands', () => {
  const openingSides = new Set<number>()
  for (let i = 0; i < 100; i++) {
    const seed = Math.imul(i, 2654435761) >>> 0
    const rallies = [serve(seed)]
    while (rallies.length < RALLIES) rallies.push(serve(seed, rallies[rallies.length - 1]))
    const bands = rallies.map((f) => Math.floor((f.y - 22) / 24))
    assert.deepEqual(bands.sort(), [0, 1, 2, 3, 4])
    openingSides.add(rallies[0].vx)
    for (const vx of [-5, 5]) {
      const received = rallies.filter((f) => f.vx === vx)
      assert.ok(received.some((f) => f.vy < 0) && received.some((f) => f.vy > 0))
    }
    for (const [index, f] of rallies.entries()) {
      const prior = index ? { ...rallies[index - 1], tick: 712, points: [2, 1] as Pair<number> } : undefined
      const again = serve(seed, prior)
      assert.deepEqual([again.y, again.vx, again.vy], [f.y, f.vx, f.vy])
    }
  }
  assert.equal(openingSides.size, 2)
})
test('a rally timeout is a draw and a last-tick point takes precedence', () => {
  const f = { ...serve(24), rallyTick: RALLY_TICK_LIMIT - 1 }
  assert.notEqual(advance({ ...f, rallyTick: 799 }, still).event, 'draw')
  assert.equal(advance(f, still).drawReason, 'timeout')
  assert.equal(advance({ ...f, x: 17, y: 30, vx: -5, vy: 0 }, still).event, 'point')
})
test('v3 keeps v2 physics and observes a loop without moving either player differently', () => {
  const start = { ...serve(24), x: 20, y: 80, vx: -5, vy: 0 }
  let model = start, human = start, old = start
  const loops = new LoopDetectorV3()
  for (let tick = 1; tick <= 133; tick++) {
    model = advanceV3(model, still, loops)
    human = advanceV3(human, still)
    old = advanceV2(old, still)
    const { drawReason, ...physical } = human
    assert.equal(drawReason, undefined)
    assert.deepEqual(physical, old)
    if (tick < 133) assert.deepEqual(model, human)
  }
  assert.equal(model.event, 'draw')
  assert.equal(model.drawReason, 'loop')
  assert.equal(model.actor, undefined)
  assert.deepEqual(model.points, [0, 0])
  assert.equal(human.event, 'return')
  while (human.rallyTick < RALLY_TICK_LIMIT) human = advanceV3(human, still)
  assert.equal(human.drawReason, 'timeout')
})
test('only the complete post-contact state proves a deterministic loop', () => {
  const contact = advance({ ...serve(24), x: 20, y: 80, vx: -5, vy: 0 }, still)
  const loops = new LoopDetector()
  assert.equal(loops.observe({ ...contact, event: null }), false)
  assert.equal(loops.observe(contact), false)
  for (const key of ['x', 'y', 'vx', 'vy'] as const)
    assert.equal(loops.observe({ ...contact, [key]: contact[key] + 1 }), false)
  assert.equal(loops.observe({ ...contact, paddles: [82, 80] }), false)
  assert.equal(loops.observe({ ...contact, paddles: [80, 82] }), false)
  assert.equal(loops.observe({ ...contact, tick: 99, rallyTick: 99, returns: 5, points: [1, 1] }), true)
  assert.equal(loops.observe({ ...contact, rally: 2 }), false)
  loops.reset()
  assert.equal(loops.observe({ ...contact, rally: 2 }), false)
})
test('a proven loop at the deadline is labelled as a loop; points still win', () => {
  const before = { ...serve(24), x: 20, y: 80, vx: -5, vy: 0 }
  const loops = new LoopDetector()
  loops.observe(advance(before, still))
  assert.equal(advance({ ...before, rallyTick: RALLY_TICK_LIMIT - 1 }, still, loops).drawReason, 'loop')
  const point = advance({ ...before, y: 30, rallyTick: RALLY_TICK_LIMIT - 1 }, still, loops)
  assert.equal(point.event, 'point')
  assert.equal(point.drawReason, undefined)
  assert.equal(serve(24, point).drawReason, undefined)
})
test('ties go to the first action and unusable output fails explicitly', () => {
  assert.equal(choice([2, 2, -1]), 0)
  assert.throws(() => choice([0, NaN, 1]))
})
const starter = readFileSync(new URL('../public/models/tracker.onnx', import.meta.url))
test('the starter fits the documented contract and compute budget', () => {
  const stats = validateModel(starter)
  assert.deepEqual([stats.params, stats.macs, stats.nodes], [163, 144, 3])
  assert.deepEqual(stats.architecture.widths, [6, 16, 3])
  assert.deepEqual(
    stats.architecture.operations.map((n) => n.op),
    ['Gemm', 'Relu', 'Gemm'],
  )
})
test('the archive inspector checks graph and file limits without granting active eligibility', () => {
  const model = pb.onnx.ModelProto.decode(starter)
  model.graph!.node![1].opType = 'Tanh'
  const encode = () => pb.onnx.ModelProto.encode(model).finish()
  assert.equal(validateModel(encode()).nodes, 3)
  model.graph!.node![1].opType = 'Loop'
  assert.throws(() => validateModel(encode()), /Unsupported operator/)
  assert.throws(() => validateModel(new Uint8Array(262145)), /256 KiB/)
})
test('the MAC limit counts repeated use of an embedded matrix', () => {
  const m = pb.onnx.ModelProto.decode(starter),
    g = m.graph!
  const weights = (name: string, dims: number[]) => ({
    name,
    dims,
    dataType: 1,
    rawData: new Uint8Array(dims.reduce((a, b) => a * b, 1) * 4),
  })
  g.initializer = [weights('a', [6, 128]), weights('b', [128, 128]), weights('c', [128, 3])]
  const mm = (a: string, b: string, out: string) => ({
    opType: 'MatMul',
    input: [a, b],
    output: [out],
  })
  g.node = [
    mm('observation', 'a', 'x'),
    mm('x', 'b', 'y'),
    mm('y', 'b', 'z'),
    mm('z', 'b', 'v'),
    mm('v', 'b', 'u'),
    mm('u', 'c', 'action_scores'),
  ]
  assert.throws(() => validateModel(pb.onnx.ModelProto.encode(m).finish()), /compute budget/)
})

test('the published v4 training reference stays identical to the live rules', () => {
  assert.equal(
    readFileSync(new URL('../training/v4/game.ts', import.meta.url), 'utf8'),
    readFileSync(new URL('../src/game.ts', import.meta.url), 'utf8'),
  )
})

test('the archived v3 engine stays frozen for earlier models and rule studies', () => {
  const bytes = readFileSync(new URL('../training/v3/game.ts', import.meta.url))
  assert.equal(createHash('sha256').update(bytes).digest('hex'),
    '74e4e0eae1e5ec54bca0b9926a8b0cf9d6a973115371e4df4351599bf188dc6a')
})

test('the archived v1 engine stays frozen for old submissions and training results', () => {
  const bytes = readFileSync(new URL('../training/v1/game.ts', import.meta.url))
  assert.equal(
    createHash('sha256').update(bytes).digest('hex'),
    '1404bea9c6a840cdb917c977cafc65fd830fa023dc599094bf4924aa39f7cb21',
  )
})

test('the archived v2 engine stays frozen for earlier models and balance studies', () => {
  const bytes = readFileSync(new URL('../training/v2/game.ts', import.meta.url))
  assert.equal(createHash('sha256').update(bytes).digest('hex'),
    '70fb842396dcd67b01693763bc98497206dad1ca4668a61535db22311fb0d45d')
})

test('trained exports reveal their actual widths and remain within the upload contract', () => {
  for (const name of ['apprentice-v2', 'selfplay-v2', 'apprentice-v3', 'selfplay-v3']) {
    const bytes = readFileSync(new URL(`../public/models/${name}.onnx`, import.meta.url))
    const graph = pb.onnx.ModelProto.decode(bytes)
    assert.equal(graph.metadataProps.find((p) => p.key === 'rules')?.value, 'pong-' + name.split('-')[1])
    const stats = validateModel(bytes)
    assert.deepEqual(stats.architecture.widths, [6, 64, 64, 3])
    assert.equal(stats.params, 4803)
    assert.equal(stats.macs, 4672)
    assert.deepEqual(
      stats.architecture.operations.map((n) => n.op),
      ['Gemm', 'Tanh', 'Gemm', 'Tanh', 'Gemm'],
    )
  }
})

test('the archive inspector describes a branched model without inventing a dense architecture', () => {
  const model = pb.onnx.ModelProto.decode(starter)
  const nodes = model.graph!.node!
  const hidden = nodes[0].output![0]
  const activated = nodes[1].output![0]
  nodes[2].input![0] = 'residual'
  nodes.splice(2, 0, { opType: 'Add', input: [hidden, activated], output: ['residual'] })
  const stats = validateModel(pb.onnx.ModelProto.encode(model).finish())
  assert.equal(stats.architecture.widths, null)
  assert.deepEqual(stats.architecture.operations[2].inputs, [hidden, activated])
  assert.deepEqual(stats.architecture.operations[2].shape, [1, 16])
})

test('the fully learned release is the evaluated plain MLP, without fixed feature branches', () => {
  const bytes = readFileSync(new URL('../public/models/contender-v3.onnx', import.meta.url))
  const frozen = JSON.parse(readFileSync(new URL('../training/results/fully-learned/candidate/freeze.json', import.meta.url), 'utf8'))
  assert.equal(createHash('sha256').update(bytes).digest('hex'), frozen.onnx_sha256)
  const stats = validateModel(bytes)
  assert.deepEqual(stats.architecture.widths, [6, 64, 64, 3])
  assert.equal(stats.params, frozen.trainable_parameters)
  assert.deepEqual(stats.architecture.operations.map((node) => node.op), ['Gemm', 'Tanh', 'Gemm', 'Tanh', 'Gemm'])
})

test('v4 releases use the evaluated learned weights and expose their larger architecture', () => {
  const manifest = JSON.parse(readFileSync(new URL('../training/results/pace-study/training/manifest.json', import.meta.url), 'utf8'))
  for (const name of ['pacer', 'apprentice']) {
    const bytes = readFileSync(new URL(`../public/models/${name}-v4.onnx`, import.meta.url))
    const expected = manifest.files[`training/results/pace-study/training/candidate/${name}/best.onnx`]
    assert.equal(createHash('sha256').update(bytes).digest('hex'), expected)
    const graph = pb.onnx.ModelProto.decode(bytes)
    assert.equal(graph.metadataProps.find((p) => p.key === 'rules')?.value, 'pong-v4')
    const stats = validateModel(bytes)
    assert.deepEqual(stats.architecture.widths, [6, 128, 128, 3])
    assert.equal(stats.params, 17795)
    assert.deepEqual(stats.architecture.operations.map((node) => node.op), ['Gemm', 'Tanh', 'Gemm', 'Tanh', 'Gemm'])
  }
})

test('Vector Nano publishes its frozen plain dense weights within the Nano budget', () => {
  const bytes = readFileSync(new URL('../public/models/vector-nano-v4.onnx', import.meta.url))
  assert.equal(createHash('sha256').update(bytes).digest('hex'),
    'ad41b0b8b9be1264c7b3daea95aec2341ab14f4802038ef175aea5d4898ac551')
  const stats = validateArenaModel(bytes)
  assert.deepEqual([stats.inputSize, stats.outputSize], [6, 3])
  assert.deepEqual(stats.architecture.widths, [6, 28, 9, 3])
  assert.deepEqual(stats.architecture.operations.map((node) => node.op),
    ['Gemm', 'Relu', 'Gemm', 'Relu', 'Gemm'])
  assert.deepEqual([stats.params, stats.macs, stats.nodes], [487, 447, 5])
  assert.equal(smallestDivision(stats), 'nano')
  const model = pb.onnx.ModelProto.decode(bytes)
  assert.match(model.metadataProps.find((p) => p.key === 'method')!.value!, /no fitting\/training/)
})

test('Prism Nano publishes its frozen plain dense weights within the Nano budget', () => {
  const bytes = readFileSync(new URL('../public/models/prism-nano-v4.onnx', import.meta.url))
  assert.equal(createHash('sha256').update(bytes).digest('hex'),
    '1eedf32a2fff0721aa101d4ba8a925a6fac7734f259daa9180ecb6523f65991a')
  const stats = validateArenaModel(bytes)
  assert.deepEqual([stats.inputSize, stats.outputSize], [6, 3])
  assert.deepEqual(stats.architecture.widths, [6, 29, 9, 3])
  assert.deepEqual(stats.architecture.operations.map((node) => node.op),
    ['Gemm', 'Relu', 'Gemm', 'Relu', 'Gemm'])
  assert.deepEqual([stats.params, stats.macs, stats.nodes], [503, 462, 5])
  assert.equal(smallestDivision(stats), 'nano')
  const model = pb.onnx.ModelProto.decode(bytes)
  assert.match(model.metadataProps.find((p) => p.key === 'method')!.value!, /no training/)
})

test('Prism Small publishes its frozen plain dense weights within the Small budget', () => {
  const bytes = readFileSync(new URL('../public/models/prism-small-v4.onnx', import.meta.url))
  assert.equal(createHash('sha256').update(bytes).digest('hex'),
    '64b6f14a604fd6d7ff72b91a2861b7a061f6568b6edde80d4697c2e2160e31ca')
  const stats = validateArenaModel(bytes)
  assert.deepEqual([stats.inputSize, stats.outputSize], [6, 3])
  assert.deepEqual(stats.architecture.widths, [6, 42, 33, 3])
  assert.deepEqual(stats.architecture.operations.map((node) => node.op),
    ['Gemm', 'Relu', 'Gemm', 'Relu', 'Gemm'])
  assert.deepEqual([stats.params, stats.macs, stats.nodes], [1815, 1737, 5])
  assert.equal(smallestDivision(stats), 'small')
  const model = pb.onnx.ModelProto.decode(bytes)
  assert.match(model.metadataProps.find((p) => p.key === 'method')!.value!, /no training/)
})

test('the retired handcrafted policies remain inspectable archives but cannot enter dense-only play', () => {
  const frozen = JSON.parse(readFileSync(new URL('../training/results/compact-handcrafted-v4/freeze.json', import.meta.url), 'utf8'))
  for (const size of ['nano', 'small']) {
    const bytes = readFileSync(new URL(`../public/models/handcrafted-${size}-v4.onnx`, import.meta.url))
    const expected = frozen.models[size]
    assert.equal(createHash('sha256').update(bytes).digest('hex'), expected.sha256)
    const model = pb.onnx.ModelProto.decode(bytes)
    const metadata = new Map(model.metadataProps.map((p) => [p.key, p.value]))
    assert.equal(metadata.get('rules'), 'pong-v4')
    assert.equal(metadata.get('division'), size)
    assert.match(metadata.get('method')!, /no training/)
    const stats = validateModel(bytes)
    assert.deepEqual([stats.inputSize, stats.outputSize], [6, 3])
    assert.deepEqual([stats.params, stats.macs, stats.nodes],
      [expected.arena.params, expected.arena.macs, expected.arena.nodes])
    assert.equal(smallestDivision(stats), size)
    assert.equal(stats.architecture.widths, null)
    assert.throws(() => validateArenaModel(bytes), DenseModelError,
      'Correct ports and an old Nano/Small label must not admit a physics graph')
  }
})

test('the archived counter retains its frozen bytes but is ineligible under the dense contract', () => {
  const bytes = readFileSync(new URL('../public/models/counter-attacker-v4.onnx', import.meta.url))
  const frozen = JSON.parse(readFileSync(new URL('../training/results/counter-attacker-v4/freeze.json', import.meta.url), 'utf8'))
  const verification = JSON.parse(readFileSync(new URL('../training/results/counter-attacker-v4/verification.json', import.meta.url), 'utf8'))
  assert.equal(createHash('sha256').update(bytes).digest('hex'), frozen.model_sha256)
  const model = pb.onnx.ModelProto.decode(bytes)
  const metadata = new Map(model.metadataProps.map((p) => [p.key, p.value]))
  assert.equal(metadata.get('rules'), 'pong-v4')
  assert.match(metadata.get('method')!, /no training/)
  assert.match(metadata.get('limitation')!, /cap13/)
  const stats = validateModel(bytes)
  assert.deepEqual([stats.inputSize, stats.outputSize], [6, 3])
  assert.deepEqual([stats.params, stats.macs, stats.nodes],
    [verification.arena.params, verification.arena.macs, verification.arena.nodes])
  assert.equal(smallestDivision(stats), 'small')
  assert.equal(stats.architecture.widths, null)
  assert.throws(() => validateArenaModel(bytes), DenseModelError)
})

test('the earlier Strategist hybrid cannot bypass dense-only eligibility with six inputs and three outputs', () => {
  const bytes = readFileSync(new URL('../public/models/strategist-v3.onnx', import.meta.url))
  assert.deepEqual([validateModel(bytes).inputSize, validateModel(bytes).outputSize], [6, 3])
  assert.throws(() => validateArenaModel(bytes), DenseModelError)
})

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
import { validateModel } from '../src/validate'
import { advance as advanceV1, serve as serveV1 } from '../training/v1/game'
import { advance as advanceV2 } from '../training/v2/game'

const still: Pair<Scores> = [
  [0, 1, 0],
  [0, 1, 0],
]
test('a policy can play either side using the same observation', () => {
  const a = { ...serve(24), x: 24, vx: -5, paddles: [28, 62] as Pair<number> }
  const b = { ...a, x: 360 - a.x, vx: -a.vx, paddles: [62, 28] as Pair<number> }
  assert.deepEqual(observation(a, 0), observation(b, 1))
})
test('edge contact gives a defender less time to reach the corner', () => {
  const before = { ...serve(24), x: 20, y: 80, vy: 0, vx: -5 }
  assert.equal(advance(before, still).vx, 5)
  assert.equal(advance({ ...before, y: 95 }, still).vx, 13)
  assert.equal(before.points[0] + before.points[1], 0)
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
      const { drawReason, ...next } = advance(before, still)
      assert.equal(drawReason, undefined)
      assert.deepEqual({ ...next, vx: old.vx }, old)
      assert.ok(Math.abs(next.vx) <= 13)
    }
  }
  const edge = { ...serve(24), x: 20, y: 94, vx: -5, vy: 0 }
  assert.equal(advance(edge, still).vx, 12)
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
  const loops = new LoopDetector()
  for (let tick = 1; tick <= 133; tick++) {
    model = advance(model, still, loops)
    human = advance(human, still)
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
  while (human.rallyTick < RALLY_TICK_LIMIT) human = advance(human, still)
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
test('uploads may change hidden architecture but must respect graph and file limits', () => {
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

test('the published v3 training reference stays identical to the live rules', () => {
  assert.equal(
    readFileSync(new URL('../training/v3/game.ts', import.meta.url), 'utf8'),
    readFileSync(new URL('../src/game.ts', import.meta.url), 'utf8'),
  )
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

test('a branched upload is shown as a graph rather than an invented sequential architecture', () => {
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

import test from 'node:test'
import assert from 'node:assert/strict'
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
      assert.equal(next.vy, Math.round(9 * Math.min(13, speed + 1) * offset / 195) || 0)
      const counted = advance({ ...before, returns: 100, rallyTick: 700 }, still)
      assert.deepEqual([next.x, next.y, next.vx, next.vy], [counted.x, counted.y, counted.vx, counted.vy])
    }
  }
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
test('centered serves vary their angles and give both receivers both directions', () => {
  const openingSides = new Set<number>()
  for (let i = 0; i < 100; i++) {
    const seed = Math.imul(i, 2654435761) >>> 0
    const rallies = [serve(seed)]
    while (rallies.length < RALLIES) rallies.push(serve(seed, rallies[rallies.length - 1]))
    assert.equal(new Set(rallies.map((f) => Math.abs(f.vy))).size, 5)
    for (const [index, frame] of rallies.entries()) {
      assert.deepEqual([frame.x, frame.y, ...frame.paddles], [180, 80, 80, 80])
      const angle = Math.atan2(Math.abs(frame.vy), Math.abs(frame.vx)) * 180 / Math.PI
      assert.ok(angle >= 10 && angle <= 15, `Opening angle ${angle} must stay within 10–15°`)
      if (index) assert.equal(frame.vx, -rallies[index - 1].vx, 'Receivers alternate')
    }
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
test('every opening is catchable after a 600 ms reaction delay, with no wall bounce', () => {
  for (let i = 0; i < 1000; i++) {
    const seed = Math.imul(i, 2654435761) >>> 0
    let prior: ReturnType<typeof serve> | undefined
    for (let rally = 1; rally <= RALLIES; rally++) {
      const start = serve(seed, prior)
      const receiver = start.vx < 0 ? 0 : 1
      const landing = start.y + start.vy * 33
      let frame = start
      for (let tick = 1; tick <= 33; tick++) {
        const scores: Pair<Scores> = [[0, 1, 0], [0, 1, 0]]
        if (tick > 18) { // Eighteen decisions at 30 Hz allow 600 ms to react.
          const error = landing - frame.paddles[receiver]
          scores[receiver] = Math.abs(error) <= 1 ? [0, 1, 0]
            : error < 0 ? [1, 0, 0] : [0, 0, 1]
        }
        frame = advance(frame, scores)
        assert.ok(frame.y >= 4 && frame.y <= 156)
        if (tick < 33) assert.equal(frame.vy, start.vy, 'No opening wall bounce')
      }
      assert.equal(frame.event, 'return', `Seed ${seed}, rally ${rally} must be catchable`)
      assert.equal(frame.actor, receiver)
      prior = frame
    }
  }
})
test('a rally timeout is a draw and a last-tick point takes precedence', () => {
  const f = { ...serve(24), rallyTick: RALLY_TICK_LIMIT - 1 }
  assert.notEqual(advance({ ...f, rallyTick: 799 }, still).event, 'draw')
  assert.equal(advance(f, still).drawReason, 'timeout')
  assert.equal(advance({ ...f, x: 17, y: 30, vx: -5, vy: 0 }, still).event, 'point')
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

test('loop detection ends a deterministic rally without altering preceding movement', () => {
  let detected = { ...serve(24), x: 20, y: 80, vx: -5, vy: 0 }, uninterrupted = detected
  const loops = new LoopDetector()
  while (!detected.drawReason) {
    detected = advance(detected, still, loops)
    uninterrupted = advance(uninterrupted, still)
    if (!detected.drawReason) assert.deepEqual(detected, uninterrupted)
  }
  assert.equal(detected.drawReason, 'loop')
  assert.equal(uninterrupted.event, 'return')
  assert.deepEqual(detected.points, [0, 0])
  assert.ok(detected.rallyTick < RALLY_TICK_LIMIT)
})

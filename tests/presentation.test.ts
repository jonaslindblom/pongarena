import test from 'node:test'
import assert from 'node:assert/strict'
import { advance, serve, type Frame, type Pair, type Scores } from '../src/game'
import { boundedDeadline, frameDuration, interpolate, progress, TICK_MS, type Presentation } from '../src/presentation'

const still: Pair<Scores> = [[0, 1, 0], [0, 1, 0]]

test('rendering fills intervals without changing either authoritative physics frame', () => {
  const from = { ...serve(1), x: 100, y: 80, vx: 5, vy: 3 }
  const to = advance(from, [[1, 0, 0], [0, 0, 1]])
  const snapshot = JSON.stringify([from, to])
  assert.deepEqual(interpolate(from, to, 0), { x: 100, y: 80, paddles: [80, 80] })
  assert.deepEqual(interpolate(from, to, 0.5), { x: 102.5, y: 81.5, paddles: [79, 81] })
  assert.deepEqual(interpolate(from, to, 1), { x: to.x, y: to.y, paddles: to.paddles })
  assert.equal(JSON.stringify([from, to]), snapshot)
})

test('wall animation reaches the wall then reflects instead of cutting through the corner', () => {
  const from = { ...serve(1), x: 100, y: 5, vx: 5, vy: -4 }
  const to = advance(from, still)
  assert.equal(to.y, 7)
  assert.equal(interpolate(from, to, 0.25).y, 4)
  assert.equal(interpolate(from, to, 0.5).y, 5)
  for (let t = 0; t <= 1; t += 0.01) assert.ok(interpolate(from, to, t).y >= 4)
})

test('a return never visually travels behind its paddle contact plane', () => {
  const from = { ...serve(1), x: 20, y: 80, vx: -13, vy: 0 }
  const to = advance(from, still)
  assert.equal(to.event, 'return')
  for (let t = 0; t <= 1; t += 0.01) {
    const point = interpolate(from, to, t)
    assert.ok(point.x >= 15 && point.x <= 20)
  }
  assert.equal(interpolate(from, to, 1).x, 15)
})

test('new serves and nonadjacent frames snap directly to the authoritative state', () => {
  const from = { ...serve(1), x: 15, y: 140, tick: 90 }
  for (const to of [serve(1, from), { ...from, tick: 20, x: 250 }])
    assert.deepEqual(interpolate(from, to, 0.25), { x: to.x, y: to.y, paddles: to.paddles })
  const to = advance(from, still)
  const p: Presentation = { from, to, startedAt: 100, duration: TICK_MS, animate: false, trail: [] }
  assert.equal(progress(p, 100), 1, 'Pausing must show the latest physics state immediately')
  assert.ok(Math.abs(progress({ ...p, animate: true }, 100 + TICK_MS / 2) - 0.5) < 1e-12)
  assert.equal(progress({ ...p, animate: true }, 500), 1, 'Rendering must never predict an uncomputed next state')
})

test('absolute tick deadlines retain cadence and bound stall debt without skipping game states', () => {
  const frame = serve(1)
  assert.equal(frameDuration(frame), 1000 / 30, 'Live play must use the Classic cadence of 30 decisions per second')
  const terminal = { ...frame, event: 'point' } as Frame
  assert.equal(frameDuration(terminal), 650)
  assert.equal(boundedDeadline(100, 105), 100, 'Normal inference latency must not move the next deadline')
  assert.equal(boundedDeadline(100, 1000), 900, 'A long stall leaves at most three ticks of wall-clock debt')
})

import test from 'node:test'
import assert from 'node:assert/strict'
import { PONG, endedMatch, type Frame, type Pair, type Scores } from '../src/rules'
import { interactiveMatchEnded, interactiveServe, interactiveTickLimit } from '../src/interactiveMatch'
import { frameDuration } from '../src/presentation'

const still: Pair<Scores> = [[0, 1, 0], [0, 1, 0]]

test('human games stop at three while league and spectator games play every scheduled rally', () => {
  for (const points of [[3, 0], [1, 3]] as Pair<number>[]) {
    const win: Frame = { ...PONG.serve(24), rally: points[0] + points[1], event: 'point', points }
    assert.equal(interactiveMatchEnded(win, true), true)
    assert.equal(interactiveMatchEnded(win, false), false)
    assert.equal(endedMatch(win), false)
  }
  const draw: Frame = { ...PONG.serve(24), rally: 5, event: 'draw', points: [2, 1] }
  assert.equal(interactiveMatchEnded(draw, true), false, 'draws cannot consume the chance to reach three')
  assert.equal(interactiveMatchEnded(draw, false), true)
  assert.equal(endedMatch(draw), true)
})

test('extra human serves remain finite, retain scores, and replay deterministically after draws', () => {
  function serves() {
    let frame = PONG.serve(24)
    const result: Frame[] = []
    for (let i = 0; i < 16; i++) {
      const prior: Frame = { ...frame, event: 'draw', points: [2, 2] }
      frame = interactiveServe(PONG, 24, prior)
      assert.equal(frame.rally, prior.rally + 1)
      assert.equal(frame.tick, prior.tick + 1)
      assert.deepEqual(frame.points, [2, 2])
      assert.ok(Number.isFinite(frame.vy) && frame.vy !== 0)
      result.push(frame)
    }
    return result
  }
  assert.deepEqual(serves(), serves())
})

test('spectators get 90 seconds without extending the league or human deadline', () => {
  const prior = { ...PONG.serve(24), rallyTick: 1599 }
  assert.equal(PONG.advance(prior, still).drawReason, 'timeout')
  assert.equal(PONG.advance(prior, still, undefined, interactiveTickLimit(true)).drawReason, 'timeout')
  assert.equal(PONG.advance(prior, still, undefined, interactiveTickLimit(false)).event, null)
  const lastTick = { ...prior, rallyTick: 2699 }
  assert.equal(PONG.advance(lastTick, still, undefined, interactiveTickLimit(false)).drawReason, 'timeout')
  assert.equal(PONG.advance({ ...lastTick, x: 17, vx: -5, y: 30, vy: 0 }, still,
    undefined, interactiveTickLimit(false)).event, 'point', 'a last-tick goal must count')
})

test('draw explanations stay visible longer than ordinary point transitions', () => {
  const frame = PONG.serve(24)
  assert.equal(frameDuration({ ...frame, event: 'draw' }), 3000)
  assert.equal(frameDuration({ ...frame, event: 'point' }), 650)
})

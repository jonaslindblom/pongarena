import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { advance, LoopDetector, observation, serve, type Frame, type Pair, type Scores } from '../src/game'

const fixture = JSON.parse(readFileSync(new URL('../training/fixtures/game.json', import.meta.url), 'utf8')) as {
  serves: { seed: number; frames: Frame[] }[]
  traces: { name: string; initial: Frame; actions: Pair<number>[]; states: number[][]; observations: Pair<number[]> }[]
}
const actions: Scores[] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

function state(frame: Frame) {
  return [frame.tick, frame.rally, frame.rallyTick, frame.x, frame.y, frame.vx, frame.vy,
    ...frame.paddles, ...frame.points, frame.returns,
    [null, 'serve', 'return', 'point', 'draw'].indexOf(frame.event),
    frame.actor ?? -1, [undefined, 'loop', 'timeout'].indexOf(frame.drawReason)]
}

test('browser serves match the independent Python training simulator for every rally', () => {
  for (const { seed, frames } of fixture.serves) {
    let prior: Frame | undefined
    for (const expected of frames) {
      const actual = serve(seed, prior)
      assert.deepEqual(actual, expected, `seed ${seed}, rally ${expected.rally}`)
      prior = actual
    }
  }
})

test('browser physics and observations match independent Python edge-case traces', () => {
  for (const trace of fixture.traces) {
    let frame = trace.initial
    const loops = new LoopDetector()
    assert.equal(trace.actions.length, trace.states.length)
    for (const [index, [left, right]] of trace.actions.entries()) {
      frame = advance(frame, [actions[left], actions[right]], loops)
      assert.deepEqual(state(frame), trace.states[index], `${trace.name}, step ${index + 1}`)
    }
    assert.deepEqual([...observation(frame, 0)], trace.observations[0], `${trace.name}, left observation`)
    assert.deepEqual([...observation(frame, 1)], trace.observations[1], `${trace.name}, right observation`)
  }
})

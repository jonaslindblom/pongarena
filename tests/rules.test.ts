import test from 'node:test'
import assert from 'node:assert/strict'
import { PONG } from '../src/rules'
import { advance, observation, serve } from '../src/game'

test('every play mode uses the same six-input, three-action Pong rules', () => {
  assert.deepEqual([PONG.id, PONG.contractId], ['pong', 'dense-mlp'])
  assert.deepEqual([PONG.inputSize, PONG.outputSize, PONG.actions], [6, 3, ['Up', 'Stay', 'Down']])
  const start = PONG.serve(24)
  assert.deepEqual(start, serve(24))
  assert.deepEqual(PONG.observation(start, 1), observation(start, 1))
  assert.deepEqual(PONG.advance(start, [[0, 1, 0], [0, 0, 1]], PONG.detector()),
    advance(start, [[0, 1, 0], [0, 0, 1]]))
})

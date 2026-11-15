import test from 'node:test'
import assert from 'node:assert/strict'
import { newMatchSeed } from '../src/matchSeed'

test('fresh matches draw from the full uint32 seed range', (t) => {
  t.mock.method(crypto, 'getRandomValues', (values: Uint32Array) => {
    values[0] = 0xffffffff
    return values
  })
  assert.equal(newMatchSeed(), 0xffffffff)
  assert.equal(newMatchSeed(902), 0xffffffff)
})

test('even a repeated random draw gives the next match a different valid seed', (t) => {
  let draw = 902
  t.mock.method(crypto, 'getRandomValues', (values: Uint32Array) => {
    values[0] = draw
    return values
  })
  assert.equal(newMatchSeed(902), 903)
  draw = 0xffffffff
  assert.equal(newMatchSeed(draw), 0, 'Collisions wrap within the uint32 seed range')
})

import test from 'node:test'
import assert from 'node:assert/strict'
import { courtTargetAt, heldHumanAction, humanSideAt, humanTargetAction, type HeldDirection } from '../src/humanInput'

test('a burst drag finishes at its target without oscillating or wasting energy', () => {
  assert.equal(humanTargetAction(60, 80, true), 4)
  assert.equal(humanTargetAction(78, 80, true), 2)
  assert.equal(humanTargetAction(80, 80, true), 1)
  assert.equal(humanTargetAction(82, 80, true), 0)
  assert.equal(humanTargetAction(90, 80, true), 3)
  assert.equal(humanTargetAction(60, 80, false), 2)
})

test('a single human can drag anywhere while two humans keep their own court halves', () => {
  assert.equal(humanSideAt(290, 300, [true, false]), 0)
  assert.equal(humanSideAt(10, 300, [false, true]), 1)
  assert.equal(humanSideAt(149, 300, [true, true]), 0)
  assert.equal(humanSideAt(150, 300, [true, true]), 1)
  assert.equal(humanSideAt(90, 300, [false, false]), null)
})

test('a captured drag outside the court remains a reachable paddle target', () => {
  assert.equal(courtTargetAt(-90, 200), 15)
  assert.equal(courtTargetAt(100, 200), 80)
  assert.equal(courtTargetAt(350, 200), 145)
  assert.equal(courtTargetAt(0, 0), 15)
})

test('opposite held buttons cancel, releasing one restores the remaining direction', () => {
  const held = new Map<string, HeldDirection>([
    ['first-finger', { side: 0, action: 0 }],
    ['second-finger', { side: 0, action: 2 }],
    ['other-player', { side: 1, action: 2 }],
  ])
  assert.equal(heldHumanAction(held.values(), 0), 1)
  assert.equal(heldHumanAction(held.values(), 1), 2)
  held.delete('second-finger')
  assert.equal(heldHumanAction(held.values(), 0), 0)
  held.delete('first-finger')
  assert.equal(heldHumanAction(held.values(), 0), null)
  assert.equal(heldHumanAction(held.values(), 1), 2)
  held.clear()
  assert.equal(heldHumanAction(held.values(), 1), null)
})

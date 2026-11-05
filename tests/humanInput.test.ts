import test from 'node:test'
import assert from 'node:assert/strict'
import { courtActionAt, heldHumanAction, humanKeyAction, humanSideAt, type HeldDirection } from '../src/humanInput'

test('screen halves request held directions instead of a target at the thumb position', () => {
  assert.equal(courtActionAt(0, 300), 0)
  assert.equal(courtActionAt(149, 300), 0)
  assert.equal(courtActionAt(150, 300), 2)
  assert.equal(courtActionAt(300, 300), 2)
  assert.equal(courtActionAt(-20, 600), 0)
  assert.equal(courtActionAt(700, 600), 2)
})

test('a solo human can use both thumbs anywhere on the screen', () => {
  for (const height of [10, 590]) {
    assert.equal(humanSideAt(height, 600, [true, false]), 0)
    assert.equal(humanSideAt(height, 600, [false, true]), 1)
    assert.equal(humanSideAt(height, 600, [false, false]), null)
  }
})

test('releasing one of two thumbs on the same half keeps the other thumb moving', () => {
  const held = new Map<number, HeldDirection>([
    [1, { side: 0, action: 0 }],
    [2, { side: 0, action: 0 }],
  ])
  assert.equal(heldHumanAction(held.values(), 0), 0)
  held.delete(1)
  assert.equal(heldHumanAction(held.values(), 0), 0)
  held.delete(2)
  assert.equal(heldHumanAction(held.values(), 0), null)
})

test('two humans have independent top and bottom touch zones', () => {
  assert.equal(humanSideAt(10, 600, [true, true]), 1)
  assert.equal(humanSideAt(299, 600, [true, true]), 1)
  assert.equal(humanSideAt(300, 600, [true, true]), 0)
  assert.equal(humanSideAt(590, 600, [true, true]), 0)
})

test('horizontal keys steer either solo paddle and keep two-human controls independent', () => {
  for (const side of [0, 1] as const) {
    for (const code of ['KeyA', 'ArrowLeft', 'KeyW', 'ArrowUp'])
      assert.equal(humanKeyAction(new Set([code]), side, false), 0)
    for (const code of ['KeyD', 'ArrowRight', 'KeyS', 'ArrowDown'])
      assert.equal(humanKeyAction(new Set([code]), side, false), 2)
    assert.equal(humanKeyAction(new Set(['ArrowLeft', 'ArrowRight']), side, false), 1)
    assert.equal(humanKeyAction(new Set(), side, false), null)
  }
  assert.equal(humanKeyAction(new Set(['KeyA', 'ArrowRight']), 0, true), 0)
  assert.equal(humanKeyAction(new Set(['KeyA', 'ArrowRight']), 1, true), 2)
  assert.equal(humanKeyAction(new Set(['KeyD']), 1, true), null)
  assert.equal(humanKeyAction(new Set(['ArrowLeft']), 0, true), null)
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

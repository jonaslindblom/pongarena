import test from 'node:test'
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { sha256 } from '../src/competition'
import { modelFile } from './modelFiles'

test('model identities use standard SHA-256 on secure origins', async () => {
  assert.equal(await sha256(new Uint8Array()), 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')
  assert.equal(await sha256(new TextEncoder().encode('abc')), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
})

test('phone HTTP fallback preserves the same file and benchmark identities without SubtleCrypto', async () => {
  const bytes = readFileSync(modelFile('pacer-v4'))
  const native = await sha256(bytes)
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'crypto')!
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: { getRandomValues: crypto.getRandomValues.bind(crypto) } })
  try {
    assert.equal(globalThis.crypto.subtle, undefined)
    assert.equal(await sha256(bytes), native)
    assert.equal(await sha256(bytes), '01eeeee5f0c2c65d934c4988ebd24e91c0da7c3f1d88c9a48829dc881262739e')
    assert.equal(await sha256(new TextEncoder().encode('abc')), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
  } finally {
    Object.defineProperty(globalThis, 'crypto', descriptor)
  }
})

test('hashing a typed-array view excludes surrounding bytes and does not mutate the upload', async () => {
  const backing = new Uint8Array([99, 97, 98, 99, 100])
  const view = backing.subarray(1, 4)
  const expected = createHash('sha256').update(view).digest('hex')
  assert.equal(await sha256(view), expected)
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'crypto')!
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: undefined })
  try { assert.equal(await sha256(view), expected) }
  finally { Object.defineProperty(globalThis, 'crypto', descriptor) }
  assert.deepEqual([...backing], [99, 97, 98, 99, 100])
})

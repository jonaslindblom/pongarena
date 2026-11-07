import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { parseSubmissionRoster } from '../scripts/submission-roster'
import { presets } from '../src/roster'

test('submitted roster metadata is parsed without executing the module', () => {
  assert.deepEqual(parseSubmissionRoster(readFileSync('src/roster.ts', 'utf8')), presets)
  const global = globalThis as typeof globalThis & { rosterExecuted?: boolean }
  assert.throws(() => parseSubmissionRoster('globalThis.rosterExecuted = true; export const presets = []'))
  assert.equal(global.rosterExecuted, undefined)
  assert.throws(() => parseSubmissionRoster('export const presets = [{ id: "x", name: (() => { globalThis.rosterExecuted = true; return "X" })() }]'))
  assert.equal(global.rosterExecuted, undefined)
})

test('submitted roster rejects paths, duplicate IDs, spreads and computed metadata', () => {
  for (const source of [
    'export const presets = [{ id: "../escape", name: "Escape" }]',
    'export const presets = [{ id: "x", name: "X" }, { id: "x", name: "Other" }]',
    'export const presets = [...models]',
    'export const presets = [{ ...model, id: "x", name: "X" }]',
    'export const presets = [{ id: "x", name: "X", ["handcrafted"]: true }]',
    'export const presets = [{ id: "x", id: "y", name: "X" }]',
    'export const presets = [{ id: "x", name: "X", handcrafted: 1 }]',
  ]) assert.throws(() => parseSubmissionRoster(source), source)
})

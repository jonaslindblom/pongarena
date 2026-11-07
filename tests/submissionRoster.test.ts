import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { parseSubmissionRoster } from '../scripts/submission-roster'
import { isGitHubUsername } from '../src/submitter'
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
    'export const presets = [{ id: "../escape", name: "Escape", submittedBy: "author" }]',
    'export const presets = [{ id: "x", name: "X", submittedBy: "author" }, { id: "x", name: "Other", submittedBy: "author" }]',
    'export const presets = [...models]',
    'export const presets = [{ ...model, id: "x", name: "X", submittedBy: "author" }]',
    'export const presets = [{ id: "x", name: "X", submittedBy: "author", ["handcrafted"]: true }]',
    'export const presets = [{ id: "x", id: "y", name: "X", submittedBy: "author" }]',
    'export const presets = [{ id: "x", name: "X", submittedBy: "author", handcrafted: 1 }]',
  ]) assert.throws(() => parseSubmissionRoster(source), source)
})

test('submitter attribution accepts GitHub handles and cannot contain an arbitrary link', () => {
  for (const submittedBy of ['jonaslindblom', 'A', 'two-parts', 'a'.repeat(39)]) {
    assert(isGitHubUsername(submittedBy))
    const source = `export const presets = [${JSON.stringify({ id: 'x', name: 'X', submittedBy })}]`
    assert.equal(parseSubmissionRoster(source)[0].submittedBy, submittedBy)
  }
  for (const submittedBy of ['', '-name', 'name-', 'two--hyphens', 'has_underscore', 'a'.repeat(40),
    'https://github.com/name', 'javascript:alert(1)', 'name/other', 'name%2Fother', 'nämé', 'has space', '\nname']) {
    assert.equal(isGitHubUsername(submittedBy), false)
    const source = `export const presets = [${JSON.stringify({ id: 'x', name: 'X', submittedBy })}]`
    assert.throws(() => parseSubmissionRoster(source), /submittedBy/)
  }
  assert.throws(() => parseSubmissionRoster('export const presets = [{ id: "x", name: "X" }]'), /submittedBy/)
})

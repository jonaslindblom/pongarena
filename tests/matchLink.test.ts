import test from 'node:test'
import assert from 'node:assert/strict'
import { readMatchLink, writeMatchLink } from '../src/matchLink'

test('new games default to a human facing Counter', () => {
  assert.deepEqual(readMatchLink(''), { mode: 'play', ids: ['keyboard', 'counter97'] })
  assert.deepEqual(readMatchLink('?model=follow'), { mode: 'play', ids: ['keyboard', 'follow'] })
})

test('copied match links restore both watch players and preserve fullscreen and unrelated parameters', () => {
  const url = writeMatchLink(new URL('https://example.com/pong/?fullscreen=1&source=test'), 'watch', ['counter97', 'follow'])
  assert.deepEqual(readMatchLink(url.search), { mode: 'watch', ids: ['counter97', 'follow'] })
  assert.equal(url.searchParams.get('fullscreen'), '1')
  assert.equal(url.searchParams.get('source'), 'test')
  const play = writeMatchLink(url, 'play', ['keyboard', 'quill'])
  assert.deepEqual(readMatchLink(play.search), { mode: 'play', ids: ['keyboard', 'quill'] })
  assert.equal(play.searchParams.has('opponent'), false)
})

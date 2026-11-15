import test from 'node:test'
import assert from 'node:assert/strict'
import { readMatchLink, writeMatchLink } from '../src/matchLink'

test('unconfigured visits show Quill versus Counter, including fullscreen and campaign links', () => {
  for (const search of ['', '?fullscreen=1', '?source=test'])
    assert.deepEqual(readMatchLink(search), { mode: 'watch', ids: ['quill', 'counter'], demo: true })
})

test('human match links wait for input while watch links autoplay the chosen models', () => {
  assert.deepEqual(readMatchLink('?model=follow'), { mode: 'play', ids: ['keyboard', 'follow'], demo: false })
  assert.deepEqual(readMatchLink('?mode=play'), { mode: 'play', ids: ['keyboard', 'counter'], demo: false })
  assert.deepEqual(readMatchLink('?mode=watch'), { mode: 'watch', ids: ['counter', 'quill'], demo: true })
  assert.equal(readMatchLink('?mode=watch&model=keyboard').demo, false)
  assert.equal(readMatchLink('?mode=watch&opponent=keyboard').demo, false)
})

test('copied match links restore both watch players and preserve fullscreen and unrelated parameters', () => {
  const url = writeMatchLink(new URL('https://example.com/pong/?fullscreen=1&source=test'), 'watch', ['counter', 'follow'])
  assert.deepEqual(readMatchLink(url.search), { mode: 'watch', ids: ['counter', 'follow'], demo: true })
  assert.equal(url.searchParams.get('fullscreen'), '1')
  assert.equal(url.searchParams.get('source'), 'test')
  const play = writeMatchLink(url, 'play', ['keyboard', 'quill'])
  assert.deepEqual(readMatchLink(play.search), { mode: 'play', ids: ['keyboard', 'quill'], demo: false })
  assert.equal(play.searchParams.has('opponent'), false)
})

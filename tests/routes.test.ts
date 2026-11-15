import test from 'node:test'
import assert from 'node:assert/strict'
import { normalizeReadingUrl, pageMetadata, pagePath, readPage } from '../src/routes'

for (const prefix of ['/', '/pongarena/']) {
  const base = new URL('https://example.com' + prefix)
  test(`shared reading URLs retain their destination and query under ${prefix}`, () => {
    for (const [legacy, path, page, lesson] of [
      ['#how-it-thinks', 'how-it-thinks/', 'How it thinks', 'follow'],
      ['#how-it-thinks/counter', 'how-it-thinks/counter/', 'How it thinks', 'counter'],
      ['#findings', 'findings/', 'Findings', 'training'],
      ['#findings/oneshot', 'findings/oneshot/', 'Findings', 'oneshot'],
      ['#how-it-thinks/oneshot', 'findings/oneshot/', 'Findings', 'oneshot'],
    ]) {
      const original = new URL('?source=shared&model=quill' + legacy, base)
      const normalized = normalizeReadingUrl(original, base)
      assert.equal(normalized.href, new URL(path + '?source=shared&model=quill', base).href)
      assert.equal(original.hash, legacy, 'normalization must not mutate the caller’s URL')
      const route = readPage(normalized, base)
      assert.equal(route.page, page)
      assert.equal(route.finding ?? route.explainer, lesson)
      assert.equal(pagePath(route.page, route.finding ?? route.explainer), path)
      assert.equal(readPage(new URL(path.slice(0, -1), base), base).page, page)
    }
  })

  test(`document anchors and other app tabs remain usable under ${prefix}`, () => {
    const reading = new URL('findings/#training-method-title', base)
    assert.equal(normalizeReadingUrl(reading, base).href, reading.href)
    assert.equal(readPage(reading, base).page, 'Findings')
    assert.deepEqual(readPage(new URL('#models/oneshot', base), base), {
      page: 'Model library', explainer: 'follow', catalogId: 'oneshot',
    })
    assert.equal(readPage(new URL('#guide', base), base).page, 'Rules')
    assert.equal(readPage(new URL('?model=quill&fullscreen=1#play', base), base).page, 'Arena')
  })
}

test('reading metadata identifies the selected page and restores the homepage on return', () => {
  const base = new URL('https://example.com/pongarena/')
  const reading = pageMetadata(readPage(new URL('how-it-thinks/counter/?source=test', base), base))
  assert.equal(reading.title, 'Switch Off a Pong Neuron: Inside Counter — PongArena')
  assert.equal(reading.url, 'https://jonaslindblom.github.io/pongarena/how-it-thinks/counter/')
  const home = pageMetadata(readPage(new URL('#play', base), base))
  assert.equal(home.title, 'PongArena — Play Pong Against Tiny Neural Networks')
  assert.equal(home.url, 'https://jonaslindblom.github.io/pongarena/')
})

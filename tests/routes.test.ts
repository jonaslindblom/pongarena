import test from 'node:test'
import assert from 'node:assert/strict'
import { pageMetadata, pagePath, readPage } from '../src/routes'

for (const prefix of ['/', '/pongarena/']) {
  const base = new URL('https://example.com' + prefix)
  test(`every page has a direct path under ${prefix}`, () => {
    for (const [path, page, lesson] of [
      ['', 'Arena', 'follow'],
      ['models/', 'Model library', 'follow'],
      ['league/', 'Rankings', 'follow'],
      ['rules/', 'Rules', 'follow'],
      ['device-test/', 'Device test', 'follow'],
      ['how-it-thinks/', 'How it thinks', 'follow'],
      ['how-it-thinks/counter/', 'How it thinks', 'counter'],
      ['findings/', 'Findings', 'training'],
      ['findings/oneshot/', 'Findings', 'oneshot'],
    ]) {
      const route = readPage(new URL(path + '?source=shared', base), base)
      assert.equal(route.page, page)
      assert.equal(route.finding ?? route.explainer, lesson)
      assert.equal(pagePath(route.page, route.finding ?? route.explainer), path)
      assert.equal(readPage(new URL(path.slice(0, -1), base), base).page, page)
    }
  })

  test(`fragments identify content within a page under ${prefix}`, () => {
    const reading = new URL('findings/#training-method-title', base)
    assert.equal(readPage(reading, base).page, 'Findings')
    assert.deepEqual(readPage(new URL('models/#catalog-model-oneshot', base), base), {
      page: 'Model library', explainer: 'follow', catalogId: 'oneshot',
    })
    assert.equal(readPage(new URL('?model=quill&fullscreen=1', base), base).page, 'Arena')
  })
}

test('metadata identifies the selected page and restores the homepage on return', () => {
  const base = new URL('https://example.com/pongarena/')
  const reading = pageMetadata(readPage(new URL('how-it-thinks/counter/?source=test', base), base))
  assert.equal(reading.title, 'Switch Off a Pong Neuron: Inside Counter — PongArena')
  assert.equal(reading.url, 'https://jonaslindblom.github.io/pongarena/how-it-thinks/counter/')
  const models = pageMetadata(readPage(new URL('models/#catalog-model-oneshot', base), base))
  assert.equal(models.url, 'https://jonaslindblom.github.io/pongarena/models/')
  const home = pageMetadata(readPage(base, base))
  assert.equal(home.title, 'PongArena — Play Pong Against Tiny Neural Networks')
  assert.equal(home.url, 'https://jonaslindblom.github.io/pongarena/')
})

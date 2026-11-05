import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { advance, endedMatch, endedRally, LoopDetector, serve, type Pair, type Scores } from './game.ts'

const fixtures = JSON.parse(readFileSync(new URL('./fixtures.json', import.meta.url), 'utf8'))
const scores = (actions: number[]) => actions.map(a => [0, 1, 2, 3, 4].map(i => Number(i === a))) as Pair<Scores>
const plain = (value: unknown) => JSON.parse(JSON.stringify(value))
for (const example of fixtures.boundaries) {
  assert.deepEqual(plain(advance(example.before, scores(example.actions))), example.after)
}
let frames = 0
for (const match of fixtures.matches) {
  let frame = serve(match.seed)
  const loop = new LoopDetector()
  for (const step of match.steps) {
    assert.equal(endedRally(frame), step.actions === null)
    frame = step.actions === null ? serve(match.seed, frame) : advance(frame, scores(step.actions), loop)
    assert.deepEqual(plain(frame), step.state)
    frames++
  }
  assert.ok(endedMatch(frame))
}
console.log(`PASS: ${fixtures.boundaries.length} boundaries and ${frames} full-match frames agree with the independent Python reference.`)

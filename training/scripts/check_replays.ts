import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { advance, endedMatch, endedRally, serve } from '../../src/game.ts'
import { validateModel } from '../../src/validate.ts'

const file = process.argv[2] ?? 'training/results/v2/onnx-replays.json'
const matches = JSON.parse(readFileSync(file, 'utf8'))
let ticks = 0
for (const match of matches) {
  let frame = serve(match.seed)
  for (const step of match.steps) {
    assert.equal(endedRally(frame), step.scores === null)
    frame = step.scores === null ? serve(match.seed, frame) : advance(frame, step.scores)
    assert.deepEqual(JSON.parse(JSON.stringify(frame)), step.state)
    ticks++
  }
  assert.ok(endedMatch(frame))
}
for (const name of ['apprentice-v2', 'selfplay-v2']) {
  const stats = validateModel(readFileSync(`public/models/${name}.onnx`))
  assert.deepEqual(stats.architecture.widths, [6, 64, 64, 3])
  console.log(name, stats.params, 'parameters;', stats.macs, 'MACs')
}
console.log(
  `PASS: ${matches.length} complete Python/ONNX matches reproduce exactly in the live TypeScript engine (${ticks} frames).`,
)

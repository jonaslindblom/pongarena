import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import * as live from '../../src/game.ts'
import * as v3 from '../v3/game.ts'
import { validateModel } from '../../src/validate.ts'

const archived = process.argv.includes('--v3')
const { advance, LoopDetector, endedMatch, endedRally, serve } = archived ? v3 : live
const file = process.argv.slice(2).find((arg) => arg !== '--v3') ??
  (archived ? 'training/results/v3/onnx-replays.json' : 'public/training/pace-onnx-replays.json')
const matches = JSON.parse(readFileSync(file, 'utf8'))
let ticks = 0
for (const match of matches) {
  let frame = serve(match.seed)
  const loops = new LoopDetector()
  for (const step of match.steps) {
    assert.equal(endedRally(frame), step.scores === null)
    frame = step.scores === null ? serve(match.seed, frame) : advance(frame, step.scores, loops)
    assert.deepEqual(JSON.parse(JSON.stringify(frame)), step.state)
    ticks++
  }
  assert.ok(endedMatch(frame))
}
for (const name of archived ? ['apprentice-v3', 'selfplay-v3'] : ['pacer-v4', 'apprentice-v4']) {
  const stats = validateModel(readFileSync(`public/models/${name}.onnx`))
  assert.deepEqual(stats.architecture.widths, archived ? [6, 64, 64, 3] : [6, 128, 128, 3])
  console.log(name, stats.params, 'parameters;', stats.macs, 'MACs')
}
console.log(
  `PASS: ${matches.length} complete Python/ONNX matches reproduce exactly in the ${archived ? 'archived v3' : 'live v4'} TypeScript engine (${ticks} frames).`,
)

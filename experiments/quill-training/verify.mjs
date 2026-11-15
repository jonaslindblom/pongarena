// Validate the actual browser contract and replay a subset in its WASM runtime.
import assert from 'node:assert/strict'
import { readFile, writeFile } from 'node:fs/promises'
import * as ort from 'onnxruntime-web/wasm'
import { sha256 } from '../../src/competition.ts'
import { competitionMatch } from '../../src/competitionMatch.ts'
import { PONG, choice } from '../../src/rules.ts'
import { validateArenaModel } from '../../src/validate.ts'
import { smokeTestModels } from '../../scripts/validate-models.ts'

const saved = JSON.parse(await readFile(new URL('./results.json', import.meta.url), 'utf8'))
const bytes = await readFile(new URL('../../public/models/small.onnx', import.meta.url))
const stats = validateArenaModel(bytes)
assert.equal(await sha256(bytes), saved.primary_sha256)
assert.deepEqual(stats.architecture.widths, [6, 12, 4, 3, 3])
assert.equal(stats.params, 163)
assert.equal(stats.macs, 141)
await smokeTestModels([{ bytes, stats, entrant: { id: 'quill-student' } }])
ort.env.wasm.numThreads = 1
const sessions = []
async function controller(data) {
  const session = await ort.InferenceSession.create(data, { executionProviders: ['wasm'] })
  sessions.push(session)
  return async input => {
    const output = await session.run({ observation: new ort.Tensor('float32', input, [1, 6]) })
    const scores = Array.from(output.action_scores.data)
    choice(scores)
    return scores
  }
}
const results = {}
try {
  const student = await controller(bytes)
  for (const name of ['quill', 'counter']) {
    const opponentBytes = await readFile(new URL(`../../public/models/${name}.onnx`, import.meta.url))
    assert.equal(await sha256(opponentBytes), saved.opponent_sha256[name])
    const opponent = await controller(opponentBytes)
    const margins = []
    for (const seed of saved.seeds.slice(0, 16)) {
      const a = await competitionMatch([student, opponent], PONG, seed)
      const b = await competitionMatch([opponent, student], PONG, seed)
      margins.push(a.points[0] - a.points[1] + b.points[1] - b.points[0])
    }
    assert.deepEqual(margins, saved.primary_results[name].paired_point_margin.slice(0, 16))
    results[name] = margins
  }
} finally {
  await Promise.all(sessions.map(session => session.release()))
}
const result = { parameters: stats.params, macs: stats.macs, runtime_probes: 217,
  paired_sets_per_opponent: 16, game_count: 64, paired_margins: results }
if (process.argv[2]) await writeFile(process.argv[2], JSON.stringify(result, null, 2) + '\n', { flag: 'wx' })
console.log(JSON.stringify(result))

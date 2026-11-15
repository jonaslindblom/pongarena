import test from 'node:test'
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import * as ort from 'onnxruntime-web/wasm'
import { findingsSource } from '../scripts/generate-findings'
import { findingsData } from '../src/findingsData'
import { competitionMatch } from '../src/competitionMatch'
import { PONG, choice, type Scores } from '../src/rules'
import { validateArenaModel } from '../src/validate'

const root = new URL('../', import.meta.url)
const read = (path: string) => readFile(new URL(path, root), 'utf8')
const saved = JSON.parse(await read('experiments/quill-training/current-comparison.json'))

test('Findings display exactly the frozen evidence for the current model bytes', async () => {
  assert.equal(await read('src/findingsData.ts'), await findingsSource())
  assert.equal(findingsData.rulesId, PONG.id)
  assert.equal(findingsData.contractId, PONG.contractId)
  assert.equal(findingsData.pairedServeSets, 512)
  assert.equal(findingsData.ralliesPerMatchup, 5120)
  assert.equal(findingsData.matchups.length, 20)
  assert.equal(new Set(findingsData.matchups.map(row => row.model + '/' + row.opponent)).size, 20)
  for (const [file, expected] of Object.entries(saved.plan.evaluator_sha256)) {
    const bytes = await readFile(new URL(file, root))
    assert.equal(createHash('sha256').update(bytes).digest('hex'), expected, `${file} no longer matches the frozen evaluator`)
  }
  for (const id of ['quill', 'small', 'oneshot']) {
    const stats = validateArenaModel(await readFile(new URL(`public/models/${id}.onnx`, root)))
    assert.deepEqual(stats.architecture.widths, [6, 12, 4, 3, 3], 'The comparison must hold architecture constant')
    assert.equal(stats.params, 163)
    assert.equal(stats.macs, 141)
  }
})

test('Findings use fresh shared serves and preserve both scoring units in either direction', async () => {
  const earlier = JSON.parse(await read('experiments/quill-training/results.json'))
  const promptSeeds = [
    ...JSON.parse(await read('experiments/one-prompt-weights/seeds.json')),
    ...JSON.parse(await read('experiments/one-prompt-weights/follow-up/seeds.json')),
  ]
  const excluded = new Set([...earlier.seeds, ...promptSeeds])
  assert.equal(new Set(saved.plan.seeds).size, 512)
  assert(saved.plan.seeds.every((seed: number) => Number.isInteger(seed) && seed >= 2_800_000_000 && seed < 2 ** 32 && !excluded.has(seed)))
  for (const row of findingsData.matchups) {
    const inverse = findingsData.matchups.find(other => other.model === row.opponent && other.opponent === row.model)!
    assert.equal(row.rallyScore + inverse.rallyScore, 1)
    assert.equal(row.leagueScore + inverse.leagueScore, 1)
    assert.equal(row.wins + row.draws + row.losses, 5120)
    assert.equal(row.contestWins + row.contestDraws + row.contestLosses, 512)
    assert(Math.abs(row.rallyScore - (row.wins + row.draws / 2) / 5120) < Number.EPSILON)
    assert.equal(row.leagueScore, (row.contestWins + row.contestDraws / 2) / 512)
    assert.equal(row.rallyScoreCi95[0] + inverse.rallyScoreCi95[1], 1)
    assert.equal(row.leagueScoreCi95[0] + inverse.leagueScoreCi95[1], 1)
  }
})

test('the browser runtime reproduces native Findings margins across every retained matchup', async () => {
  ort.env.wasm.numThreads = 1
  const sessions: ort.InferenceSession[] = []
  const decisions = new Map<string, (input: Float32Array) => Promise<Scores>>()
  try {
    for (const { id } of findingsData.models) {
      const bytes = await readFile(new URL(`public/models/${id}.onnx`, root))
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
      sessions.push(session)
      decisions.set(id, async input => {
        const output = await session.run({ observation: new ort.Tensor('float32', input, [1, PONG.inputSize]) })
        const scores = Array.from(output.action_scores.data as Float32Array) as Scores
        choice(scores)
        return scores
      })
    }
    for (const row of saved.matchups) {
      const model = decisions.get(row.model)!, opponent = decisions.get(row.opponent)!
      const margins = []
      for (const seed of saved.plan.seeds.slice(0, 8)) {
        const a = await competitionMatch([model, opponent], PONG, seed)
        const b = await competitionMatch([opponent, model], PONG, seed)
        margins.push(a.points[0] - a.points[1] + b.points[1] - b.points[0])
      }
      assert.deepEqual(margins, row.paired_point_margin.slice(0, 8), `${row.model} vs ${row.opponent}`)
    }
  } finally {
    await Promise.all(sessions.map(session => session.release()))
  }
})

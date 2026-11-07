import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { generateLeague, leagueMarkdown } from '../scripts/generate-league'
import { restoreCompetition, sha256 } from '../src/competition'
import { competitionMatch } from '../src/competitionMatch'
import { LEAGUE_SEEDS } from '../src/league'
import { presets } from '../src/roster'
import { PONG } from '../src/rules'

const saved = JSON.parse(readFileSync(new URL('../public/league.json', import.meta.url), 'utf8'))

test('published results rank the complete shipped roster, exact model bytes and fixed public seeds', async () => {
  const run = await restoreCompetition(saved)
  assert.equal(run.spec.division, 'all')
  assert.equal(run.spec.rulesId, PONG.id)
  assert.equal(run.spec.contractId, PONG.contractId)
  assert.deepEqual(run.spec.seeds, [...LEAGUE_SEEDS].sort((a, b) => a - b))
  assert.deepEqual(run.spec.entrants.map((entry) => entry.id).sort(), presets.map((entry) => entry.id).sort())
  for (const entry of run.spec.entrants) {
    const expected = await sha256(readFileSync(new URL(`../public/models/${entry.id}.onnx`, import.meta.url)))
    assert.equal(entry.sha256, expected, `${entry.id}: the initial page must rank the model actually shipped`)
  }
  assert.equal(readFileSync(new URL('../public/league.md', import.meta.url), 'utf8'), leagueMarkdown(run))
})

test('CI executes every model through the shared simulator and reproduces a published serve set', async () => {
  const seed = LEAGUE_SEEDS[0]
  const generated = await generateLeague([seed])
  assert.equal(generated.status, 'complete')
  assert.deepEqual(generated.contests, saved.contests.filter((contest: { seed: number }) => contest.seed === seed))
})

test('a cancelled inference cannot advance the shared simulation or call the other model', async () => {
  const abort = new AbortController()
  let secondCalls = 0
  await assert.rejects(competitionMatch([
    async () => { abort.abort(); return [0, 1, 0] },
    async () => { secondCalls++; return [0, 1, 0] },
  ], PONG, 24, abort.signal), { name: 'AbortError' })
  assert.equal(secondCalls, 0)
})

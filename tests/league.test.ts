import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { generateLeague, leagueMarkdown, leagueProgressLogger } from '../scripts/generate-league'
import { restoreCompetition } from '../src/competition'
import { competitionMatch } from '../src/competitionMatch'
import { LEAGUE_SEEDS } from '../src/league'
import { readLeagueInputs } from '../scripts/evaluation-plan'
import { PONG } from '../src/rules'

const saved = JSON.parse(readFileSync(new URL('../public/league.json', import.meta.url), 'utf8'))

test('generated development results rank every discovered network on the fixed public seeds', async () => {
  const run = await restoreCompetition(saved)
  assert.equal(run.spec.division, 'all')
  assert.equal(run.spec.rulesId, PONG.id)
  assert.equal(run.spec.contractId, PONG.contractId)
  assert.deepEqual(run.spec.seeds, [...LEAGUE_SEEDS].sort((a, b) => a - b))
  const inputs = await readLeagueInputs()
  assert.deepEqual(run.spec.entrants, [...inputs.entrants].sort((a, b) => a.sha256.localeCompare(b.sha256)))
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


test('progress reports real game counts periodically and always reports terminal status', () => {
  let clock = 0
  const messages: string[] = []
  const report = leagueProgressLogger(message => messages.push(message), () => clock)
  report({ ...saved, status: 'running', completedMatches: 0 })
  clock = 29_999
  report({ ...saved, status: 'running', completedMatches: 10 })
  assert.equal(messages.length, 1, 'per-game callbacks must not flood CI logs')
  clock = 30_000
  report({ ...saved, status: 'running', completedMatches: 20 })
  assert.match(messages[1], /20\/.*30\.0s elapsed/)
  clock = 30_001
  report(saved)
  assert.match(messages[2], /League complete:.*100\.0%/)
  report({ ...saved, status: 'failed', completedMatches: 20 })
  assert.match(messages[3], /League failed: 20\//)
})

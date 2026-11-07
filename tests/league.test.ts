import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { generateLeague, leagueMarkdown } from '../scripts/generate-league'
import { restoreCompetition, sha256 } from '../src/competition'
import { competitionMatch } from '../src/competitionMatch'
import { LEAGUE_SEEDS } from '../src/league'
import { presets, scripts, scriptIdentity } from '../src/roster'
import { CLASSIC, type Pair } from '../src/rules'
import { scriptScores, type ScriptKind } from '../src/scripted'

const saved = JSON.parse(readFileSync(new URL('../public/league.json', import.meta.url), 'utf8'))

test('Windows line endings preserve existing LF script identities across browser and CI', async () => {
  const source = readFileSync(new URL('../src/scripted.ts', import.meta.url), 'utf8').replace(/\r\n/g, '\n')
  for (const { script: policy } of scripts) {
    const original = await sha256(new TextEncoder().encode(JSON.stringify({ policy, source })))
    assert.equal(await scriptIdentity(policy, source), original, 'Existing LF identities must not change')
    assert.equal(await scriptIdentity(policy, source.replace(/\n/g, '\r\n')), original,
      'Git checkout line endings must not create a different controller or public benchmark')
  }
})

test('published results identify the complete shipped roster, exact controller bytes and fixed public seeds', async () => {
  const run = await restoreCompetition(saved)
  assert.equal(run.spec.division, 'all')
  assert.equal(run.spec.rulesId, CLASSIC.id)
  assert.equal(run.spec.contractId, CLASSIC.contractId)
  assert.deepEqual(run.spec.seeds, [...LEAGUE_SEEDS].sort((a, b) => a - b))
  assert.deepEqual(run.spec.entrants.map((entry) => entry.id).sort(), [...presets, ...scripts].map((entry) => entry.id).sort())
  const source = readFileSync(new URL('../src/scripted.ts', import.meta.url), 'utf8')
  for (const entry of run.spec.entrants) {
    const script = scripts.find((item) => item.id === entry.id)
    const expected = script ? await scriptIdentity(script.script, source)
      : await sha256(readFileSync(new URL(`../public/models/${entry.id}.onnx`, import.meta.url)))
    assert.equal(entry.sha256, expected, `${entry.id}: the initial page must rank the controller actually shipped`)
  }
  assert.equal(readFileSync(new URL('../public/league.md', import.meta.url), 'utf8'), leagueMarkdown(run))
})

test('CI executes every model and script through the shared simulator and reproduces a published serve set', async () => {
  const seed = LEAGUE_SEEDS[0]
  const generated = await generateLeague([seed])
  assert.equal(generated.status, 'complete')
  assert.deepEqual(generated.contests, saved.contests.filter((contest: { seed: number }) => contest.seed === seed))
})

test('the shared browser and CI match loop agrees with the independent Python script fixtures', async () => {
  const fixture = JSON.parse(readFileSync(new URL('./fixtures/scripted-v4.json', import.meta.url), 'utf8')) as {
    matches: { seed: number; kinds: Pair<ScriptKind>; endings: number[][] }[]
  }
  for (const match of fixture.matches) {
    const players = match.kinds.map((kind) => async (input: Float32Array) => scriptScores(kind, input))
    const result = await competitionMatch(players as Pair<typeof players[number]>, CLASSIC, match.seed)
    assert.deepEqual(result, { points: match.endings.at(-1)!.slice(9, 11), rallies: 5 },
      `${match.kinds.join(' vs ')} seed ${match.seed}: finish the exact five-rally game`)
  }
})

test('a cancelled inference cannot advance the shared simulation or call the other controller', async () => {
  const abort = new AbortController()
  let secondCalls = 0
  await assert.rejects(competitionMatch([
    async () => { abort.abort(); return [0, 1, 0] },
    async () => { secondCalls++; return [0, 1, 0] },
  ], CLASSIC, 24, abort.signal), { name: 'AbortError' })
  assert.equal(secondCalls, 0)
})

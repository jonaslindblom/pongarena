import test from 'node:test'
import assert from 'node:assert/strict'
import {
  benchmarkIdentity, fitsDivision, restoreCompetition, runCompetition, smallestDivision, standings,
  type CompetitionEntrant, type CompetitionSpec,
} from '../src/competition'

const entrant = (digit: string, extra: Partial<CompetitionEntrant> = {}): CompetitionEntrant => ({
  id: 'model-' + digit, name: 'Model ' + digit, sha256: digit.repeat(64), params: 229, macs: 208, ...extra,
})
const a = entrant('a'), b = entrant('b'), c = entrant('c')
const spec: CompetitionSpec = {
  rulesId: 'pong-v4', contractId: 'float32-observation-6-action-scores-3',
  division: 'nano', entrants: [a, b], seeds: [24, 902],
}

test('classes charge every embedded scalar and repeated matrix use; smaller models can compete up', () => {
  assert.equal(smallestDivision({ params: 512, macs: 2048 }), 'nano')
  assert.equal(smallestDivision({ params: 513, macs: 100 }), 'small')
  assert.equal(smallestDivision({ params: 100, macs: 2049 }), 'small')
  assert.equal(smallestDivision({ params: 17795, macs: 17536 }), 'standard')
  assert.equal(smallestDivision({ params: 30000, macs: 50000 }), 'open')
  assert.equal(smallestDivision({ params: 100, macs: 65537 }), null)
  assert.equal(fitsDivision(a, 'standard'), true)
})

test('benchmark identity fixes rules, ports, class, file roster and seeds, not names or UI order', async () => {
  const expected = await benchmarkIdentity(spec)
  assert.equal(await benchmarkIdentity({ ...spec, seeds: [902, 24],
    entrants: [{ ...b, name: 'Renamed', id: 'other-id' }, a] }), expected)
  for (const change of [{ rulesId: 'pong-v5' }, { contractId: 'observation-8-action-scores-5' },
    { division: 'small' as const }, { seeds: [24, 903] }, { entrants: [a, c] }])
    assert.notEqual(await benchmarkIdentity({ ...spec, ...change }), expected)
})

test('each common serve is played both ways and a side advantage cannot win a paired contest', async () => {
  const calls: { left: string; right: string; seed: number }[] = []
  const run = await runCompetition({ ...spec, playMatch: async ({ left, right, seed }) => {
    calls.push({ left: left.id, right: right.id, seed })
    return { points: [5, 0], rallies: 5 }
  } })
  assert.deepEqual(calls, [
    { left: a.id, right: b.id, seed: 24 }, { left: b.id, right: a.id, seed: 24 },
    { left: a.id, right: b.id, seed: 902 }, { left: b.id, right: a.id, seed: 902 },
  ])
  assert.equal(run.status, 'complete')
  assert.equal(run.completedMatches, 4)
  for (const row of standings(run)) {
    assert.deepEqual([row.wins, row.draws, row.losses, row.leaguePoints, row.rank], [0, 2, 0, 1, 1])
    assert.equal(row.rallyPointsFor, 10)
  }
})

test('non-transitive opponents can share rank; names and rally margin do not invent a winner', async () => {
  const beats = new Map([[a.id, b.id], [b.id, c.id], [c.id, a.id]])
  const run = await runCompetition({ ...spec, entrants: [c, b, a], seeds: [24],
    playMatch: async ({ left, right }) => ({
      points: beats.get(left.id) === right.id ? [3, 1] : [1, 3], rallies: 5,
    }) })
  assert.equal(run.totalMatches, 6)
  for (const row of standings(run)) {
    assert.deepEqual([row.played, row.wins, row.draws, row.losses, row.rank], [2, 1, 0, 1, 1])
    assert.equal(row.rallyDraws, 4)
  }
})

test('cancelling after one side prevents a biased half-pair from contributing to any ranking', async () => {
  const abort = new AbortController()
  let calls = 0
  const run = await runCompetition({ ...spec, signal: abort.signal,
    playMatch: async () => { calls++; return { points: [5, 0], rallies: 5 } },
    onProgress: (progress) => { if (progress.completedMatches === 1) abort.abort() },
  })
  assert.equal(calls, 1)
  assert.equal(run.status, 'cancelled')
  assert.equal(run.completedMatches, 1)
  assert.equal(run.contests.length, 0)
  assert.ok(standings(run).every((row) => row.played === 0 && row.rank === null))
})

test('a failed match remains an explicit incomplete run and never awards a technical win', async () => {
  let calls = 0
  const run = await runCompetition({ ...spec, playMatch: async () => {
    if (++calls === 2) throw new Error('Inference failed')
    return { points: [4, 1], rallies: 5 }
  } })
  assert.equal(run.status, 'failed')
  assert.equal(run.error, 'Inference failed')
  assert.equal(run.contests.length, 0)
  assert.ok(standings(run).every((row) => row.rank === null && row.leaguePoints === 0))
})

test('duplicate files, repeated seeds and over-budget entries cannot inflate league points', async () => {
  const playMatch = async () => { throw new Error('Should not start') }
  await assert.rejects(runCompetition({ ...spec, entrants: [a, { ...a, id: 'copy' }], playMatch }), /only once/)
  await assert.rejects(runCompetition({ ...spec, seeds: [24, 24], playMatch }), /distinct/)
  await assert.rejects(runCompetition({ ...spec, entrants: [a, { ...b, params: 1000 }], playMatch }), /budget/)
})

test('a truncated or impossible game result cannot be counted as a finished ranked match', async () => {
  for (const result of [{ points: [3, 1] as [number, number], rallies: 4 },
    { points: [5, 1] as [number, number], rallies: 5 }]) {
    const run = await runCompetition({ ...spec, playMatch: async () => result })
    assert.equal(run.status, 'failed')
    assert.equal(run.completedMatches, 0)
    assert.equal(run.contests.length, 0)
  }
})

test('restoring local results checks the whole roster and schedule instead of trusting saved ranks', async () => {
  const run = await runCompetition({ ...spec, playMatch: async () => ({ points: [2, 2], rallies: 5 }) })
  const restored = await restoreCompetition(JSON.parse(JSON.stringify(run)))
  assert.deepEqual(standings(restored), standings(run))
  await assert.rejects(restoreCompetition({ ...run, status: 'cancelled' }), /complete/)
  await assert.rejects(restoreCompetition({ ...run, contests: run.contests.slice(1) }), /incomplete/)
  await assert.rejects(restoreCompetition({ ...run, spec: { ...run.spec, seeds: [1, 2] } }), /identity/)
  await assert.rejects(restoreCompetition({ ...run, contests: [run.contests[0], run.contests[0]] }), /repeated/)
})

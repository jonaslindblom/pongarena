import test from 'node:test'
import assert from 'node:assert/strict'
import { benchmarkIdentity, COMPETITION_PROTOCOL, standings, type CompetitionRun } from '../src/competition'
import { leagueScoreIntervals } from '../src/leagueUncertainty'

async function league(count: number, models: number, score: (seed: number, a: number, b: number) => number): Promise<CompetitionRun> {
  const spec = { rulesId: 'test', contractId: 'test', division: 'all' as const,
    seeds: Array.from({ length: count }, (_, index) => index),
    entrants: Array.from({ length: models }, (_, index) => ({ id: String(index), name: 'Model ' + index,
      sha256: (index + 1).toString(16).repeat(64), params: 1, macs: 1 })) }
  const contests: CompetitionRun['contests'] = []
  for (let a = 0; a < models; a++) for (let b = a + 1; b < models; b++) for (const seed of spec.seeds) {
    const point = score(seed, a, b)
    contests.push({ a: spec.entrants[a].sha256, b: spec.entrants[b].sha256, seed,
      matches: point === 1 ? [{ points: [1, 0], rallies: 5 }, { points: [0, 1], rallies: 5 }]
        : point === 0 ? [{ points: [0, 1], rallies: 5 }, { points: [1, 0], rallies: 5 }]
          : [{ points: [1, 0], rallies: 5 }, { points: [1, 0], rallies: 5 }] })
  }
  return { protocol: COMPETITION_PROTOCOL, benchmark: await benchmarkIdentity(spec), spec,
    contests, status: 'complete', completedMatches: contests.length * 2, totalMatches: contests.length * 2 }
}

test('interval rates match actual standings and retain both sides as one observation', async () => {
  const run = await league(128, 2, () => .5)
  const intervals = leagueScoreIntervals(run)
  for (const row of standings(run)) {
    const score = intervals.get(row.entrant.sha256)!
    assert.equal(score.rate, row.leaguePoints / row.played)
    assert.equal(score.serveSets, 128, '256 games still provide 128 paired observations')
    assert.equal(score.interval, null)
    assert.equal(score.reason, 'no-variation', 'Opposite game winners always produce a tied paired contest')
  }
})

test('shared-seed resampling preserves correlation between different opponents', async () => {
  const correlated = await league(128, 3, (seed, a) => a === 0 ? seed % 2 : .5)
  const cancelling = await league(128, 3, (_, a, b) => a === 0 ? b === 1 ? 1 : 0 : .5)
  const hash = correlated.spec.entrants[0].sha256
  const first = leagueScoreIntervals(correlated).get(hash)!
  const second = leagueScoreIntervals(cancelling).get(hash)!
  assert.equal(first.rate, .5)
  assert.equal(second.rate, .5)
  assert.ok(first.interval && first.interval[0] < .5 && first.interval[1] > .5)
  assert.equal(second.interval, null, 'Every seed has the same score across opponents despite different contest winners')
})

test('512 unique serve sets narrow sampling uncertainty compared with 128', async () => {
  const small = await league(128, 2, seed => seed % 2)
  const large = await league(512, 2, seed => seed % 2)
  const hash = small.spec.entrants[0].sha256
  const first = leagueScoreIntervals(small).get(hash)!.interval!
  const second = leagueScoreIntervals(large).get(hash)!.interval!
  assert.ok(second[1] - second[0] < (first[1] - first[0]) * .65)
})

test('resampling is reproducible, independent of display ordering, and does not mutate results', async () => {
  const run = await league(32, 3, (seed, a, b) => (seed + a + b) % 3 / 2)
  const before = JSON.stringify(run)
  const expected = leagueScoreIntervals(run)
  assert.deepEqual(leagueScoreIntervals(run), expected)
  assert.deepEqual(leagueScoreIntervals({ ...run, contests: [...run.contests].reverse(),
    spec: { ...run.spec, seeds: [...run.spec.seeds].reverse(), entrants: [...run.spec.entrants].reverse()
      .map(entrant => ({ ...entrant, name: 'Renamed' })) } }), expected)
  for (const score of expected.values()) {
    assert.ok(score.interval)
    assert.ok(score.interval![0] >= 0 && score.interval![1] <= 1)
    assert.ok(score.interval![0] <= score.rate && score.interval![1] >= score.rate)
  }
  assert.equal(JSON.stringify(run), before)
})

test('unfinished or uninformative samples never claim a precise confidence interval', async () => {
  const run = await league(16, 2, () => 1)
  assert.equal(leagueScoreIntervals({ ...run, status: 'running' }).size, 0)
  assert.equal(leagueScoreIntervals({ ...run, status: 'failed' }).size, 0)
  assert.equal(leagueScoreIntervals(run).values().next().value!.interval, null)
  const single = await league(1, 2, () => 1)
  assert.equal(leagueScoreIntervals(single).values().next().value!.reason, 'insufficient-sets')
})

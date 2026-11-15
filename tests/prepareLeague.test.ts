import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { benchmarkIdentity, type CompetitionRun } from '../src/competition'
import { createEvaluationPlan, readLeagueInputs } from '../scripts/evaluation-plan'
import { leagueMarkdown } from '../scripts/generate-league'
import { prepareLeague } from '../scripts/prepare-league'

async function publishedBuild(action: (directory: string) => Promise<void>) {
  const directory = await mkdtemp(join(tmpdir(), 'pongarena-build-league-'))
  const previous = process.env.PONGARENA_PUBLISHED_BUILD
  process.env.PONGARENA_PUBLISHED_BUILD = '1'
  try { await action(directory) }
  finally {
    if (previous === undefined) delete process.env.PONGARENA_PUBLISHED_BUILD
    else process.env.PONGARENA_PUBLISHED_BUILD = previous
    await rm(directory, { recursive: true, force: true })
  }
}

async function bundle(directory: string, count = 512) {
  const plan = createEvaluationPlan(await readLeagueInputs(), count)
  const spec = { rulesId: plan.rulesId, contractId: plan.contractId, division: 'all' as const,
    seeds: plan.seeds, entrants: [...plan.entrants].sort((a, b) => a.sha256.localeCompare(b.sha256)) }
  const contests: CompetitionRun['contests'] = []
  for (let i = 0; i < spec.entrants.length; i++) for (let j = i + 1; j < spec.entrants.length; j++)
    for (const seed of plan.seeds) contests.push({ a: spec.entrants[i].sha256, b: spec.entrants[j].sha256,
      seed, matches: [{ points: [0, 0], rallies: 5 }, { points: [0, 0], rallies: 5 }] })
  const run: CompetitionRun = { protocol: plan.protocol, benchmark: await benchmarkIdentity(spec), spec,
    status: 'complete', contests, completedMatches: contests.length * 2, totalMatches: contests.length * 2 }
  const files = new Map([
    ['evaluation-plan.json', JSON.stringify(plan)], ['league.json', JSON.stringify(run)],
    ['league.md', leagueMarkdown(run, plan)],
  ])
  for (const [name, contents] of files) await writeFile(join(directory, name), contents)
  return { files, plan, run }
}

test('a Pages build preserves the promoted 512-set results and report byte for byte', async () => {
  await publishedBuild(async directory => {
    const { files } = await bundle(directory)
    await prepareLeague(directory)
    for (const [name, contents] of files)
      assert.equal(await readFile(join(directory, name), 'utf8'), contents,
        'building the site must not regenerate or resample accepted standings')
  })
})

test('a Pages build refuses missing or development-only results instead of publishing fallback standings', async () => {
  await publishedBuild(async directory => {
    await assert.rejects(prepareLeague(directory), /Published builds require.*refusing to replace/)
    assert.deepEqual(await readdir(directory), [], 'failed production preparation must not create development results')
    const { files } = await bundle(directory, 16)
    await assert.rejects(prepareLeague(directory), /Published builds require.*512/)
    for (const [name, contents] of files) assert.equal(await readFile(join(directory, name), 'utf8'), contents)
  })
})

test('a Pages build rejects changed evaluator identity, stale labels and an altered report without rewriting results', async () => {
  await publishedBuild(async directory => {
    const { files, plan, run } = await bundle(directory)
    const changes = [
      ['evaluation-plan.json', JSON.stringify({ ...plan, evaluatorSha256: '0'.repeat(64) }), /does not match/],
      ['league.json', JSON.stringify({ ...run, spec: { ...run.spec, entrants: run.spec.entrants.map(entrant =>
        ({ ...entrant, name: 'Outdated ' + entrant.name })) } }), /labels do not match/],
      ['league.md', 'Unverified replacement standings', /table differs/],
    ] as const
    for (const [name, contents, error] of changes) {
      await writeFile(join(directory, name), contents)
      await assert.rejects(prepareLeague(directory), error)
      assert.equal(await readFile(join(directory, name), 'utf8'), contents,
        'failure must surface rather than silently repairing the publication with local results')
      await writeFile(join(directory, name), files.get(name)!)
    }
  })
})

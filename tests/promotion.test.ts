import test from 'node:test'
import assert from 'node:assert/strict'
import { benchmarkIdentity, type CompetitionRun, type CompetitionSpec } from '../src/competition'
import { createEvaluationPlan, readLeagueInputs, type EvaluationPlan } from '../scripts/evaluation-plan'
import { promoteEvaluation, verifyEvaluationBundle } from '../scripts/promote-evaluation'
import { leagueMarkdown } from '../scripts/generate-league'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

async function completeDraws(plan: EvaluationPlan) {
  const spec: CompetitionSpec = { rulesId: plan.rulesId, contractId: plan.contractId,
    division: 'all', seeds: plan.seeds, entrants: [...plan.entrants].sort((a, b) => a.sha256.localeCompare(b.sha256)) }
  const contests: CompetitionRun['contests'] = []
  for (let i = 0; i < spec.entrants.length; i++) for (let j = i + 1; j < spec.entrants.length; j++)
    for (const seed of plan.seeds) contests.push({ a: spec.entrants[i].sha256, b: spec.entrants[j].sha256,
      seed, matches: [{ points: [0, 0], rallies: 5 }, { points: [0, 0], rallies: 5 }] })
  const run: CompetitionRun = { protocol: plan.protocol, benchmark: await benchmarkIdentity(spec), spec,
    status: 'complete', contests, completedMatches: contests.length * 2, totalMatches: contests.length * 2 }
  return run
}

test('publication accepts only a complete league bound to the frozen models and full PR serve set', async () => {
  const inputs = await readLeagueInputs()
  const plan = createEvaluationPlan(inputs)
  const run = await completeDraws(plan)
  assert.equal((await verifyEvaluationBundle(plan, run, inputs)).run.benchmark, run.benchmark)
  await assert.rejects(verifyEvaluationBundle(createEvaluationPlan(inputs, 128), run, inputs), /512/)
  await assert.rejects(verifyEvaluationBundle(plan, { ...run, status: 'failed' }, inputs), /complete/)
  await assert.rejects(verifyEvaluationBundle(plan, { ...run, contests: run.contests.slice(1) }, inputs), /incomplete/)
  const otherPlan = createEvaluationPlan(inputs)
  await assert.rejects(verifyEvaluationBundle(otherPlan, run, inputs), /recorded serves/)
  await assert.rejects(verifyEvaluationBundle({ ...plan, evaluatorSha256: '0'.repeat(64) }, run, inputs), /does not match/)
})

test('publishing reused results updates model labels without changing evaluated serves or scores', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'pongarena-promote-'))
  try {
    const inputs = await readLeagueInputs()
    const rename = (entrant: typeof inputs.entrants[number]) =>
      ({ ...entrant, id: 'old-' + entrant.id, name: 'Former ' + entrant.name })
    const plan = createEvaluationPlan({ ...inputs, entrants: inputs.entrants.map(rename),
      models: inputs.models.map(model => ({ ...model, entrant: rename(model.entrant) })) })
    const evaluated = await completeDraws(plan)
    for (const [name, contents] of [
      ['evaluation-plan.json', JSON.stringify(plan)], ['league.json', JSON.stringify(evaluated)],
      ['league.md', leagueMarkdown(evaluated, plan)],
    ]) await writeFile(join(directory, name), contents)
    const output = join(directory, 'published')
    await promoteEvaluation(directory, output)
    const published: CompetitionRun = JSON.parse(await readFile(join(output, 'league.json'), 'utf8'))
    assert.deepEqual(published.spec.entrants, [...inputs.entrants].sort((a, b) => a.sha256.localeCompare(b.sha256)))
    assert.equal(published.benchmark, evaluated.benchmark)
    assert.deepEqual(published.spec.seeds, evaluated.spec.seeds)
    assert.deepEqual(published.contests, evaluated.contests)
    assert.equal(await readFile(join(output, 'league.md'), 'utf8'), leagueMarkdown(published, plan))
  } finally { await rm(directory, { recursive: true, force: true }) }
})

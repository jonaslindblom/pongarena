import test from 'node:test'
import assert from 'node:assert/strict'
import { benchmarkIdentity, type CompetitionRun, type CompetitionSpec } from '../src/competition'
import { createEvaluationPlan, readLeagueInputs } from '../scripts/evaluation-plan'
import { verifyEvaluationBundle } from '../scripts/promote-evaluation'

test('publication accepts only a complete league bound to the frozen models and full PR serve set', async () => {
  const inputs = await readLeagueInputs()
  const plan = createEvaluationPlan(inputs)
  const spec: CompetitionSpec = { rulesId: plan.rulesId, contractId: plan.contractId,
    division: 'all', seeds: plan.seeds, entrants: [...inputs.entrants].sort((a, b) => a.sha256.localeCompare(b.sha256)) }
  const contests: CompetitionRun['contests'] = []
  for (let i = 0; i < spec.entrants.length; i++) for (let j = i + 1; j < spec.entrants.length; j++)
    for (const seed of plan.seeds) contests.push({ a: spec.entrants[i].sha256, b: spec.entrants[j].sha256,
      seed, matches: [{ points: [0, 0], rallies: 5 }, { points: [0, 0], rallies: 5 }] })
  const run: CompetitionRun = { protocol: plan.protocol, benchmark: await benchmarkIdentity(spec), spec,
    status: 'complete', contests, completedMatches: contests.length * 2, totalMatches: contests.length * 2 }
  assert.equal((await verifyEvaluationBundle(plan, run, inputs)).run.benchmark, run.benchmark)
  await assert.rejects(verifyEvaluationBundle(createEvaluationPlan(inputs, 128), run, inputs), /512/)
  await assert.rejects(verifyEvaluationBundle(plan, { ...run, status: 'failed' }, inputs), /complete/)
  await assert.rejects(verifyEvaluationBundle(plan, { ...run, contests: contests.slice(1) }, inputs), /incomplete/)
  const otherPlan = createEvaluationPlan(inputs)
  await assert.rejects(verifyEvaluationBundle(otherPlan, run, inputs), /recorded serves/)
  await assert.rejects(verifyEvaluationBundle({ ...plan, evaluatorSha256: '0'.repeat(64) }, run, inputs), /does not match/)
})

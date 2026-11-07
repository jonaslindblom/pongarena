import test from 'node:test'
import assert from 'node:assert/strict'
import { createEvaluationPlan, evaluationInputIdentity, readLeagueInputs,
  validateEvaluationPlan, type LeagueInputs } from '../scripts/evaluation-plan'
import { generateLeague, leagueMarkdown } from '../scripts/generate-league'
import { benchmarkIdentity } from '../src/competition'

test('evaluation plans draw 512 distinct shared seeds after identifying the exact submitted models', async () => {
  const inputs = await readLeagueInputs()
  const first = createEvaluationPlan(inputs)
  const second = createEvaluationPlan(inputs)
  assert.equal(first.seeds.length, 512)
  assert.equal(new Set(first.seeds).size, 512)
  assert(first.seeds.every(seed => Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff))
  assert.deepEqual(first.seeds, [...first.seeds].sort((a, b) => a - b))
  assert.notDeepEqual(first.seeds, second.seeds)
  assert.equal(first.identity, inputs.identity)
  assert.equal(first.evaluatorSha256, inputs.evaluatorSha256)
  assert.deepEqual(validateEvaluationPlan(JSON.parse(JSON.stringify(first)), inputs), first)
  assert.deepEqual(first.entrants.map(entrant => entrant.sha256).sort(), inputs.entrants.map(entrant => entrant.sha256).sort())
})

test('model bytes and evaluator code determine evaluation identity; filenames, names and roster order do not', async () => {
  const { entrants, evaluatorSha256 } = await readLeagueInputs()
  const original = evaluationInputIdentity(entrants, evaluatorSha256)
  assert.equal(evaluationInputIdentity([...entrants].reverse().map(entrant =>
    ({ ...entrant, id: 'renamed-' + entrant.id, name: 'Renamed' })), evaluatorSha256), original)
  const changed = entrants.map((entrant, index) => index ? entrant : { ...entrant, sha256: 'a'.repeat(64) })
  assert.notEqual(evaluationInputIdentity(changed, evaluatorSha256), original)
  assert.notEqual(evaluationInputIdentity(entrants, '0'.repeat(64)), original)
})

test('renaming a submitted file or display name reuses the original plan and its exact seeds', async () => {
  const inputs = await readLeagueInputs()
  const plan = createEvaluationPlan(inputs)
  const rename = (entrant: typeof inputs.entrants[number]) => ({ ...entrant, id: 'renamed-' + entrant.id, name: 'Renamed' })
  const renamed: LeagueInputs = { ...inputs, entrants: inputs.entrants.map(rename),
    models: inputs.models.map(model => ({ ...model, entrant: rename(model.entrant) })) }
  renamed.identity = evaluationInputIdentity(renamed.entrants, renamed.evaluatorSha256)
  assert.equal(renamed.identity, inputs.identity)
  assert.deepEqual(validateEvaluationPlan(plan, renamed).seeds, plan.seeds)
})

test('a replay refuses stale model submissions, altered serves, invalid schedules and evaluator changes', async () => {
  const inputs = await readLeagueInputs()
  const plan = createEvaluationPlan(inputs, 3)
  const copy = () => JSON.parse(JSON.stringify(plan))
  const tampered = copy()
  tampered.seeds[0] = tampered.seeds[0] ? tampered.seeds[0] - 1 : 1
  tampered.seeds.sort((a: number, b: number) => a - b)
  assert.throws(() => validateEvaluationPlan(tampered, inputs), /integrity/)
  for (const change of [
    { seeds: [] }, { seeds: [1, 1] }, { seeds: [2, 1] }, { seeds: [-1] }, { seeds: [0x100000000] },
    { seeds: [1.5] }, { seeds: new Array(513).fill(1).map((_, index) => index) },
    { schema: 'unknown' }, { identity: '0'.repeat(64) }, { createdAt: 'invalid' },
    { entrants: plan.entrants.slice(1) },
  ]) assert.throws(() => validateEvaluationPlan({ ...copy(), ...change }, inputs))
  const alteredEvaluator = { ...inputs, evaluatorSha256: '0'.repeat(64) }
  alteredEvaluator.identity = evaluationInputIdentity(alteredEvaluator.entrants, alteredEvaluator.evaluatorSha256)
  assert.throws(() => validateEvaluationPlan(plan, alteredEvaluator), /does not match/)
  const alteredModel = copy()
  alteredModel.entrants[0].params++
  assert.throws(() => validateEvaluationPlan(alteredModel, inputs), /roster/)
})

test('model bytes cannot change between freezing and drawing an evaluation plan', async () => {
  const inputs = await readLeagueInputs()
  const corrupted: LeagueInputs = { ...inputs,
    models: inputs.models.map(model => ({ ...model, bytes: Uint8Array.from(model.bytes) })) }
  corrupted.models[0].bytes[0] ^= 1
  assert.throws(() => createEvaluationPlan(corrupted), /frozen model inputs changed/)
  assert.throws(() => validateEvaluationPlan(createEvaluationPlan(inputs), corrupted), /frozen model inputs changed/)
})

test('evaluation schedules and competition identity accept up to 512 paired seed sets', async () => {
  const inputs = await readLeagueInputs()
  for (const count of [0, -1, 1.5, 513, NaN]) assert.throws(() => createEvaluationPlan(inputs, count), /1–512/)
  const plan = createEvaluationPlan(inputs, 512)
  assert.equal(validateEvaluationPlan(plan, inputs).seeds.length, 512)
  const spec = { rulesId: plan.rulesId, contractId: plan.contractId,
    division: 'all' as const, entrants: plan.entrants, seeds: plan.seeds }
  assert.match(await benchmarkIdentity(spec), /^[a-f0-9]{64}$/)
  await assert.rejects(benchmarkIdentity({ ...spec, seeds: [...plan.seeds, 0xffffffff] }), /1–512/)
})

test('a recorded PR plan replays the same full roster and paired results in WASM', async () => {
  const inputs = await readLeagueInputs()
  const plan = createEvaluationPlan(inputs, 1)
  const first = await generateLeague(plan.seeds, inputs)
  const replay = await generateLeague(validateEvaluationPlan(plan, inputs).seeds, inputs)
  assert.deepEqual(replay, first)
  assert.deepEqual(first.spec.seeds, plan.seeds)
  assert.equal(first.spec.entrants.length, inputs.entrants.length)
  assert.equal(first.totalMatches, inputs.entrants.length * (inputs.entrants.length - 1))
  assert(first.contests.every(contest => contest.seed === plan.seeds[0] && contest.matches.length === 2))
  assert.match(leagueMarkdown(first, plan), new RegExp(plan.sha256))
  assert.match(leagueMarkdown(first, plan), /pr-evaluation/)
  assert.equal(leagueMarkdown(first, plan).split('Submitted weights').length - 1, inputs.entrants.length)
  assert.doesNotMatch(leagueMarkdown(first, plan), /Handwritten weights|Trained weights/)
})

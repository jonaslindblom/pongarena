import { createHash, randomInt } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { COMPETITION_PROTOCOL, type CompetitionEntrant } from '../src/competition'
import { PONG } from '../src/rules'
import { validateModelCollection, type ValidatedModel } from './validate-models'

export const EVALUATION_PLAN_SCHEMA = 'pongarena-evaluation-plan-v1'
export const DEFAULT_EVALUATION_SEEDS = 512
export const MAX_EVALUATION_SEEDS = 512

// Runtime and contract changes invalidate previously sampled evaluation plans.
// Display names and documentation do not change the seed-set identity.
export const EVALUATOR_FILES = [
  'package-lock.json', 'scripts/evaluation-plan.ts', 'scripts/generate-league.ts',
  'scripts/validate-models.ts', 'src/competition.ts', 'src/competitionMatch.ts',
  'scripts/submission-roster.ts', 'src/submitter.ts', 'src/game.ts', 'src/rules.ts', 'src/validate.ts',
  'src/leagueUncertainty.ts',
] as const
const root = new URL('../', import.meta.url)
const digest = (value: string | Uint8Array) => createHash('sha256').update(value).digest('hex')

export interface LeagueInputs {
  models: ValidatedModel[]
  entrants: CompetitionEntrant[]
  evaluatorSha256: string
  identity: string
  options: LeagueInputOptions
}

export interface LeagueInputOptions {
  modelsDirectory?: string | URL
  roster?: readonly { id: string; name: string; submittedBy: string }[]
}

export interface EvaluationPlan {
  schema: typeof EVALUATION_PLAN_SCHEMA
  identity: string
  evaluatorSha256: string
  createdAt: string
  rulesId: string
  contractId: string
  protocol: typeof COMPETITION_PROTOCOL
  seeds: number[]
  entrants: CompetitionEntrant[]
  sha256: string
}

const entrantKeys = (entrants: readonly CompetitionEntrant[]) => entrants.map(({ id, sha256, params, macs }) =>
  ({ id, sha256, params, macs })).sort((a, b) => a.id.localeCompare(b.id))
const entrantResources = (entrants: readonly CompetitionEntrant[]) => entrants.map(({ sha256, params, macs }) =>
  ({ sha256, params, macs })).sort((a, b) => a.sha256.localeCompare(b.sha256))

export function evaluationInputIdentity(entrants: readonly CompetitionEntrant[], evaluatorSha256: string) {
  return digest(JSON.stringify({ schema: EVALUATION_PLAN_SCHEMA, evaluatorSha256,
    rulesId: PONG.id, contractId: PONG.contractId, protocol: COMPETITION_PROTOCOL,
    entrants: entrantResources(entrants) }))
}

export async function readLeagueInputs(options: LeagueInputOptions = {}): Promise<LeagueInputs> {
  // Validate and snapshot every submitted model before any evaluation seed is drawn.
  const models = await validateModelCollection(options.modelsDirectory, options.roster)
  if (models.length < 2) throw new Error('A league requires at least two distinct submitted models.')
  const sources = await Promise.all(EVALUATOR_FILES.map(async path =>
    [path, digest(await readFile(new URL(path, root)))]))
  const evaluatorSha256 = digest(JSON.stringify(sources))
  const entrants = models.map(model => ({ ...model.entrant }))
  return { models, entrants, evaluatorSha256, identity: evaluationInputIdentity(entrants, evaluatorSha256),
    options: { ...options, roster: options.roster?.map(entrant => ({ ...entrant })) } }
}

function checkFrozenInputs(inputs: LeagueInputs) {
  if (inputs.identity !== evaluationInputIdentity(inputs.entrants, inputs.evaluatorSha256) ||
    JSON.stringify(entrantKeys(inputs.models.map(model => model.entrant))) !== JSON.stringify(entrantKeys(inputs.entrants)) ||
    inputs.models.some(model => digest(model.bytes) !== model.entrant.sha256))
    throw new Error('The frozen model inputs changed. Freeze a new submission before evaluation.')
}

export function evaluationPlanIdentity(plan: Omit<EvaluationPlan, 'sha256'>) {
  return digest(JSON.stringify({ schema: plan.schema, identity: plan.identity,
    evaluatorSha256: plan.evaluatorSha256, createdAt: plan.createdAt,
    rulesId: plan.rulesId, contractId: plan.contractId, protocol: plan.protocol,
    seeds: plan.seeds, entrants: plan.entrants.map(({ id, name, sha256, params, macs }) =>
      ({ id, name, sha256, params, macs })) }))
}

export function createEvaluationPlan(inputs: LeagueInputs, count = DEFAULT_EVALUATION_SEEDS): EvaluationPlan {
  checkFrozenInputs(inputs)
  if (!Number.isInteger(count) || count < 1 || count > MAX_EVALUATION_SEEDS)
    throw new Error(`Choose 1–${MAX_EVALUATION_SEEDS} evaluation seed sets.`)
  const unique = new Set<number>()
  while (unique.size < count) unique.add(randomInt(0, 0x100000000))
  const plan: Omit<EvaluationPlan, 'sha256'> = {
    schema: EVALUATION_PLAN_SCHEMA, identity: inputs.identity, evaluatorSha256: inputs.evaluatorSha256,
    createdAt: new Date().toISOString(), rulesId: PONG.id, contractId: PONG.contractId,
    protocol: COMPETITION_PROTOCOL, seeds: [...unique].sort((a, b) => a - b),
    entrants: inputs.entrants.map(entrant => ({ ...entrant })).sort((a, b) => a.id.localeCompare(b.id)),
  }
  return { ...plan, sha256: evaluationPlanIdentity(plan) }
}

export function validateEvaluationPlan(value: unknown, inputs: LeagueInputs): EvaluationPlan {
  checkFrozenInputs(inputs)
  if (!value || typeof value !== 'object') throw new Error('Expected a saved evaluation plan.')
  const plan = value as EvaluationPlan
  if (plan.schema !== EVALUATION_PLAN_SCHEMA || plan.identity !== inputs.identity ||
    plan.evaluatorSha256 !== inputs.evaluatorSha256 || plan.rulesId !== PONG.id ||
    plan.contractId !== PONG.contractId || plan.protocol !== COMPETITION_PROTOCOL)
    throw new Error('Evaluation plan does not match these model bytes and evaluator. Freeze a new plan.')
  if (typeof plan.createdAt !== 'string' || !Number.isFinite(Date.parse(plan.createdAt)) ||
    new Date(plan.createdAt).toISOString() !== plan.createdAt)
    throw new Error('Evaluation plan has an invalid creation timestamp.')
  if (!Array.isArray(plan.seeds) || !plan.seeds.length || plan.seeds.length > MAX_EVALUATION_SEEDS ||
    plan.seeds.some(seed => !Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) ||
    new Set(plan.seeds).size !== plan.seeds.length ||
    JSON.stringify(plan.seeds) !== JSON.stringify([...plan.seeds].sort((a, b) => a - b)))
    throw new Error('Evaluation plan needs sorted, distinct uint32 seeds within the evaluation limit.')
  if (!Array.isArray(plan.entrants) || plan.entrants.some(entrant => !entrant ||
    typeof entrant.id !== 'string' || !entrant.id || typeof entrant.name !== 'string' || !entrant.name.trim() ||
    !/^[a-f0-9]{64}$/.test(entrant.sha256) || !Number.isSafeInteger(entrant.params) || !Number.isSafeInteger(entrant.macs)) ||
    new Set(plan.entrants.map(entrant => entrant.id)).size !== plan.entrants.length ||
    new Set(plan.entrants.map(entrant => entrant.sha256)).size !== plan.entrants.length ||
    JSON.stringify(entrantResources(plan.entrants)) !== JSON.stringify(entrantResources(inputs.entrants)))
    throw new Error('Evaluation plan roster differs from the validated submitted models.')
  if (typeof plan.sha256 !== 'string' || plan.sha256 !== evaluationPlanIdentity(plan))
    throw new Error('Evaluation plan integrity check failed.')
  return JSON.parse(JSON.stringify(plan)) as EvaluationPlan
}

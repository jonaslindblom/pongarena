import { copyFile, mkdir, readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'
import { restoreCompetition, type CompetitionEntrant } from '../src/competition'
import { DEFAULT_EVALUATION_SEEDS, readLeagueInputs, validateEvaluationPlan, type LeagueInputs } from './evaluation-plan'
import { leagueMarkdown } from './generate-league'

const resources = (entrants: readonly CompetitionEntrant[]) => entrants
  .map(({ sha256, params, macs }) => ({ sha256, params, macs })).sort((a, b) => a.sha256.localeCompare(b.sha256))

/** Artifact origin is checked by CI before this verifies its contents. */
export async function verifyEvaluationBundle(planValue: unknown, runValue: unknown, inputs: LeagueInputs) {
  const plan = validateEvaluationPlan(planValue, inputs)
  if (plan.seeds.length !== DEFAULT_EVALUATION_SEEDS)
    throw new Error(`Published PR evaluations require ${DEFAULT_EVALUATION_SEEDS} paired serve sets.`)
  const run = await restoreCompetition(runValue)
  if (run.spec.rulesId !== plan.rulesId || run.spec.contractId !== plan.contractId ||
    run.spec.division !== 'all' || run.protocol !== plan.protocol ||
    JSON.stringify(run.spec.seeds) !== JSON.stringify(plan.seeds) ||
    JSON.stringify(resources(run.spec.entrants)) !== JSON.stringify(resources(inputs.entrants)))
    throw new Error('The verified league must evaluate these exact models, rules and recorded serves.')
  return { plan, run }
}

export async function promoteEvaluation(source: string, output: string, expectedPR?: number) {
  const inputs = await readLeagueInputs()
  const load = async (name: string) => JSON.parse(await readFile(resolve(source, name), 'utf8')) as unknown
  const { plan, run } = await verifyEvaluationBundle(await load('evaluation-plan.json'), await load('league.json'), inputs)
  if (expectedPR !== undefined) {
    const provenance = await load('provenance.json') as {
      schema?: string; pullRequest?: number; identity?: string; repository?: string; headSha?: string
    }
    if (provenance.schema !== 'pongarena-pr-evaluation-v1' || provenance.pullRequest !== expectedPR ||
      provenance.identity !== inputs.identity || provenance.repository !== process.env.GITHUB_REPOSITORY ||
      !/^[a-f0-9]{40}$/.test(provenance.headSha ?? ''))
      throw new Error('The results provenance does not match the merged PR and current model identities.')
  }
  const markdown = await readFile(resolve(source, 'league.md'), 'utf8')
  if (markdown !== leagueMarkdown(run, plan)) throw new Error('The league table differs from its verified results.')
  await mkdir(output, { recursive: true })
  for (const name of ['league.json', 'league.md', 'evaluation-plan.json'])
    await copyFile(resolve(source, name), resolve(output, name))
  if (expectedPR !== undefined) await copyFile(resolve(source, 'provenance.json'), resolve(output, 'provenance.json'))
  console.log(`Promoted verified benchmark ${run.benchmark} with ${run.spec.seeds.length} shared serve sets.`)
}

async function main() {
  const { values } = parseArgs({ options: {
    source: { type: 'string' }, output: { type: 'string', default: 'public' }, pr: { type: 'string' },
  } })
  if (!values.source) throw new Error('Provide --source with the trusted evaluation bundle directory.')
  const pr = values.pr === undefined ? undefined : Number(values.pr)
  if (pr !== undefined && (!Number.isSafeInteger(pr) || pr < 1)) throw new Error('Expected a positive PR number.')
  await promoteEvaluation(resolve(values.source), resolve(values.output), pr)
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href)
  main().catch(error => { console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1 })

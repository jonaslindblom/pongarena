import { readFile, mkdir, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'
import * as ort from 'onnxruntime-web/wasm'
import { runCompetition, smallestDivision, standings, type CompetitionRun } from '../src/competition'
import { competitionMatch } from '../src/competitionMatch'
import { PONG, choice, type Pair, type Scores } from '../src/rules'
import { presets } from '../src/roster'
import { LEAGUE_JSON, LEAGUE_MARKDOWN, LEAGUE_SEEDS, LEAGUE_SEED_SET } from '../src/league'
import { createEvaluationPlan, readLeagueInputs, validateEvaluationPlan,
  type EvaluationPlan, type LeagueInputs } from './evaluation-plan'
import { readSubmissionRoster } from './submission-roster'

const root = new URL('../', import.meta.url)
type Decision = (input: Float32Array) => Promise<Scores>

export async function generateLeague(seeds: readonly number[] = LEAGUE_SEEDS, frozen?: LeagueInputs): Promise<CompetitionRun> {
  ort.env.wasm.numThreads = 1
  const inputs = frozen ?? await readLeagueInputs()
  const entrants = inputs.entrants
  const decisions = new Map<string, Decision>()
  const sessions: ort.InferenceSession[] = []
  try {
    for (const { bytes, entrant } of inputs.models) {
      const hash = entrant.sha256
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
      sessions.push(session)
      decisions.set(hash, async (input) => {
        const output = await session.run({ observation: new ort.Tensor('float32', input, [1, PONG.inputSize]) })
        const scores = Array.from(output.action_scores.data as Float32Array) as Scores
        choice(scores)
        return scores
      })
    }
    const run = await runCompetition({
      rulesId: PONG.id, contractId: PONG.contractId, division: 'all', seeds: [...seeds], entrants,
      playMatch: ({ left, right, seed, signal }) => competitionMatch(
        [decisions.get(left.sha256)!, decisions.get(right.sha256)!] as Pair<Decision>, PONG, seed, signal),
    })
    if (run.status !== 'complete') throw new Error(run.error ?? 'The published league did not complete.')
    if ((await readLeagueInputs(inputs.options)).identity !== inputs.identity)
      throw new Error('Model bytes or evaluator changed during the league. Discard this run.')
    return run
  } finally {
    await Promise.all(sessions.map((session) => session.release()))
  }
}

const cell = (text: string) => text.replaceAll('|', '\\|').replace(/[\r\n]/g, ' ')

export function leagueMarkdown(run: CompetitionRun, plan?: EvaluationPlan) {
  const rows = standings(run).map((row) => {
    const entrant = row.entrant
    const preset = presets.find((p) => p.id === entrant.id)
    // PR evaluation runs in the trusted base checkout. A proposed model's
    // metadata may only become a compiled preset after merge, so keep this
    // report independent of the base checkout's display-only roster metadata.
    const kind = plan ? 'Submitted weights' : preset?.handcrafted ? 'Handwritten weights' : 'Trained weights'
    const size = smallestDivision(entrant) === 'small' ? 'Small' : 'Large'
    return `| ${row.rank} | ${cell(entrant.name)} | ${kind} | ${size} | ${row.leaguePoints} / ${row.played} | ${row.wins} / ${row.draws} / ${row.losses} | ${entrant.params} |`
  })
  return [
    '# PongArena league', '',
    `All ${run.spec.entrants.length} bundled models, ${run.spec.seeds.length} paired serve sets per matchup, ${run.totalMatches} games.`, '',
    '| Rank | Model | Kind | Size | Points | W / D / L | Parameters |',
    '| ---: | --- | --- | --- | ---: | --- | ---: |', ...rows, '',
    'Each contest combines two five-rally games with swapped sides on the same seed. A contest win earns one point; a tie earns half. Equal points share rank.', '',
    `Rules: \`${run.spec.rulesId}\` · Contract: \`${run.spec.contractId}\` · Protocol: \`${run.protocol}\` · Seeds: \`${plan ? 'pr-evaluation' : LEAGUE_SEED_SET}\`.`, '',
    `Benchmark: \`${run.benchmark}\`.`, '',
    plan
      ? `Evaluated frozen model bytes on a shared random serve set sampled after validation, using ONNX Runtime Web CPU/WASM with one thread. Evaluation plan: \`${plan.sha256}\`. Retries replay this recorded plan.`
      : 'Generated from the repository’s model bytes by `npm run league`, using ONNX Runtime Web CPU/WASM with one thread. CI reruns the league and checks both committed files. Review the passing workflow on the relevant commit before treating a proposed PR table as verified.', '',
    plan
      ? 'The published seeds and exact model hashes support replay. After using these results to tune a model, evaluate the changed submission on a new serve set. Compare scores only within the same benchmark.'
      : 'The public seeds are a repeatable development benchmark, not a private training holdout or a universal strength rating. Compare scores only within the same benchmark.', '',
    '[Full results, model hashes and serve seeds](league.json)', '',
  ].join('\n')
}

async function main() {
  const { values } = parseArgs({ options: {
    check: { type: 'boolean' }, output: { type: 'string' }, identity: { type: 'boolean' },
    'create-plan': { type: 'string' }, plan: { type: 'string' }, count: { type: 'string' },
    'models-dir': { type: 'string' }, roster: { type: 'string' },
  } })
  if (values.identity && Object.keys(values).some(key => !['identity', 'models-dir', 'roster'].includes(key)))
    throw new Error('--identity may only be combined with submission paths.')
  if (values['create-plan'] && (values.plan || values.check || values.output))
    throw new Error('--create-plan cannot be combined with --plan, --check or --output.')
  if (values.count && !values['create-plan']) throw new Error('--count requires --create-plan.')
  const inputOptions = {
    modelsDirectory: values['models-dir'] ? resolve(values['models-dir']) : undefined,
    roster: values.roster ? await readSubmissionRoster(resolve(values.roster)) : undefined,
  }
  if (values.identity || values['create-plan']) {
    const inputs = await readLeagueInputs(inputOptions)
    if (values.identity) { console.log(inputs.identity); return }
    const plan = createEvaluationPlan(inputs, values.count ? Number(values.count) : undefined)
    const path = resolve(values['create-plan']!)
    await mkdir(dirname(path), { recursive: true })
    await writeFile(path, JSON.stringify(plan, null, 2) + '\n', { flag: 'wx' })
    console.log(`Created ${plan.seeds.length} fresh shared serve sets; plan ${plan.sha256}`)
    return
  }
  const started = performance.now()
  const inputs = await readLeagueInputs(inputOptions)
  const plan = values.plan ? validateEvaluationPlan(JSON.parse(await readFile(resolve(values.plan), 'utf8')), inputs) : undefined
  const run = await generateLeague(plan?.seeds ?? LEAGUE_SEEDS, inputs)
  const artifacts = new Map([[LEAGUE_JSON, JSON.stringify(run, null, 2) + '\n'], [LEAGUE_MARKDOWN, leagueMarkdown(run, plan)]])
  if (plan) artifacts.set('evaluation-plan.json', JSON.stringify(plan, null, 2) + '\n')
  if (!values.check || values.output) {
    const directory = values.output ? resolve(values.output) : new URL('public/', root)
    await mkdir(directory, { recursive: true })
    for (const [name, contents] of artifacts) await writeFile(typeof directory === 'string' ? resolve(directory, name) : new URL(name, directory), contents)
  }
  if (values.check) {
    for (const [name, contents] of artifacts) {
      const saved = await readFile(new URL('public/' + name, root), 'utf8').catch(() => '')
      if (saved !== contents) throw new Error(`public/${name} is stale. Run npm run league and commit both league files with the model change.`)
    }
  }
  console.log(`${values.check ? 'Verified' : 'Generated'} ${run.totalMatches} league games in ${((performance.now() - started) / 1000).toFixed(1)}s; benchmark ${run.benchmark}`)
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error) => { console.error(error instanceof Error ? error.message : error); process.exitCode = 1 })
}

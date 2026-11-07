import { readFile, readdir, mkdir, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'
import * as ort from 'onnxruntime-web/wasm'
import { runCompetition, sha256, smallestDivision, standings, type CompetitionEntrant, type CompetitionRun } from '../src/competition'
import { competitionMatch } from '../src/competitionMatch'
import { CLASSIC, choice, type Pair, type Scores } from '../src/rules'
import { presets, scripts, scriptIdentity } from '../src/roster'
import { scriptScores } from '../src/scripted'
import { validateArenaModel } from '../src/validate'
import { LEAGUE_JSON, LEAGUE_MARKDOWN, LEAGUE_SEEDS, LEAGUE_SEED_SET } from '../src/league'

const root = new URL('../', import.meta.url)
type Decision = (input: Float32Array) => Promise<Scores>

export async function generateLeague(seeds: readonly number[] = LEAGUE_SEEDS): Promise<CompetitionRun> {
  ort.env.wasm.numThreads = 1
  const files = (await readdir(new URL('public/models/', root))).filter((file) => file.endsWith('.onnx')).sort()
  if (JSON.stringify(files) !== JSON.stringify(presets.map((p) => p.id + '.onnx').sort()))
    throw new Error('The active roster and public/models files differ. Update src/roster.ts with the model submission.')
  const entrants: CompetitionEntrant[] = []
  const decisions = new Map<string, Decision>()
  const sessions: ort.InferenceSession[] = []
  try {
    const source = await readFile(new URL('src/scripted.ts', root), 'utf8')
    for (const script of scripts) {
      const hash = await scriptIdentity(script.script, source)
      entrants.push({ id: script.id, name: script.name, sha256: hash, type: 'script', params: null, macs: null })
      decisions.set(hash, async (input) => scriptScores(script.script, input))
    }
    for (const preset of presets) {
      const bytes = await readFile(new URL('public/models/' + preset.id + '.onnx', root))
      const stats = validateArenaModel(bytes)
      const hash = await sha256(bytes)
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
      sessions.push(session)
      entrants.push({ id: preset.id, name: preset.name, sha256: hash, type: 'weights', params: stats.params, macs: stats.macs })
      decisions.set(hash, async (input) => {
        const output = await session.run({ observation: new ort.Tensor('float32', input, [1, CLASSIC.inputSize]) })
        const scores = Array.from(output.action_scores.data as Float32Array) as Scores
        choice(scores)
        return scores
      })
    }
    const run = await runCompetition({
      rulesId: CLASSIC.id, contractId: CLASSIC.contractId, division: 'all', seeds: [...seeds], entrants,
      playMatch: ({ left, right, seed, signal }) => competitionMatch(
        [decisions.get(left.sha256)!, decisions.get(right.sha256)!] as Pair<Decision>, CLASSIC, seed, signal),
    })
    if (run.status !== 'complete') throw new Error(run.error ?? 'The published league did not complete.')
    return run
  } finally {
    await Promise.all(sessions.map((session) => session.release()))
  }
}

const cell = (text: string) => text.replaceAll('|', '\\|').replace(/[\r\n]/g, ' ')

export function leagueMarkdown(run: CompetitionRun) {
  const rows = standings(run).map((row) => {
    const entrant = row.entrant
    const preset = presets.find((p) => p.id === entrant.id)
    const kind = entrant.type === 'script' ? 'Script' : preset?.handcrafted ? 'Handwritten weights' : 'Learned / baseline weights'
    const size = entrant.type === 'script' ? '—' : smallestDivision(entrant) === 'small' ? 'Small' : 'Large'
    return `| ${row.rank} | ${cell(entrant.name)} | ${kind} | ${size} | ${row.leaguePoints} / ${row.played} | ${row.wins} / ${row.draws} / ${row.losses} | ${entrant.params ?? '—'} |`
  })
  return [
    '# PongArena league', '',
    `All ${run.spec.entrants.length} bundled controllers, ${run.spec.seeds.length} paired serve sets per matchup, ${run.totalMatches} games.`, '',
    '| Rank | Controller | Kind | Size | Points | W / D / L | Parameters |',
    '| ---: | --- | --- | --- | ---: | --- | ---: |', ...rows, '',
    'Each contest combines two five-rally games with swapped sides on the same seed. A contest win earns one point; a tie earns half. Equal points share rank.', '',
    `Rules: \`${run.spec.rulesId}\` · Contract: \`${run.spec.contractId}\` · Protocol: \`${run.protocol}\` · Seeds: \`${LEAGUE_SEED_SET}\`.`, '',
    `Benchmark: \`${run.benchmark}\`.`, '',
    'Generated from the repository’s model bytes and scripts by `npm run league`, using ONNX Runtime Web CPU/WASM with one thread. CI reruns the league and checks both committed files. Review the passing workflow on the relevant commit before treating a proposed PR table as verified.', '',
    'The public seeds are a repeatable development benchmark, not a private training holdout or a universal strength rating. Compare scores only within the same benchmark.', '',
    '[Full results, controller hashes and serve seeds](league.json)', '',
  ].join('\n')
}

async function main() {
  const { values } = parseArgs({ options: { check: { type: 'boolean' }, output: { type: 'string' } } })
  const started = performance.now()
  const run = await generateLeague()
  const artifacts = new Map([[LEAGUE_JSON, JSON.stringify(run, null, 2) + '\n'], [LEAGUE_MARKDOWN, leagueMarkdown(run)]])
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

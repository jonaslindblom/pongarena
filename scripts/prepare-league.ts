import { readFile, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { restoreCompetition, type CompetitionEntrant } from '../src/competition'
import { readLeagueInputs } from './evaluation-plan'
import { generateLeague, leagueMarkdown } from './generate-league'
import { verifyEvaluationBundle } from './promote-evaluation'
import { PONG } from '../src/rules'

const roster = (entries: readonly CompetitionEntrant[]) => entries.map(({ id, name, sha256, params, macs }) =>
  ({ id, name, sha256, params, macs })).sort((a, b) => a.id.localeCompare(b.id))

/** Fresh clones get development standings; a build preserves matching promoted results. */
export async function prepareLeague(directory = 'public') {
  const inputs = await readLeagueInputs()
  const file = (name: string) => resolve(directory, name)
  if (process.env.PONGARENA_PUBLISHED_BUILD === '1') {
    try {
      const { plan, run } = await verifyEvaluationBundle(
        JSON.parse(await readFile(file('evaluation-plan.json'), 'utf8')),
        JSON.parse(await readFile(file('league.json'), 'utf8')), inputs)
      if (JSON.stringify(roster(run.spec.entrants)) !== JSON.stringify(roster(inputs.entrants)))
        throw new Error('The promoted model labels do not match the catalog being built.')
      if (await readFile(file('league.md'), 'utf8') !== leagueMarkdown(run, plan))
        throw new Error('The promoted league table differs from its verified results.')
      return
    } catch (error) {
      throw new Error('Published builds require the verified 512-set evaluation bundle; refusing to replace it with development standings. ' +
        (error instanceof Error ? error.message : String(error)))
    }
  }
  try {
    const run = await restoreCompetition(JSON.parse(await readFile(file('league.json'), 'utf8')))
    if (run.spec.rulesId === PONG.id && run.spec.contractId === PONG.contractId && run.spec.division === 'all' &&
      JSON.stringify(roster(run.spec.entrants)) === JSON.stringify(roster(inputs.entrants))) {
      await readFile(file('league.md'), 'utf8')
      return
    }
  } catch (error) {
    if (!(error instanceof Error)) throw error
    // Missing or stale local standings are derived data and can be regenerated.
  }
  const run = await generateLeague(undefined, inputs)
  await writeFile(file('league.json'), JSON.stringify(run, null, 2) + '\n')
  await writeFile(file('league.md'), leagueMarkdown(run))
  console.log(`Generated development standings for ${run.spec.entrants.length} discovered models.`)
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href)
  prepareLeague().catch(error => { console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1 })

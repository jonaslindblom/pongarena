import { lstat, readFile, readdir, stat } from 'node:fs/promises'
import { basename, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'
import * as ort from 'onnxruntime-web/wasm'
import { sha256, smallestDivision, type CompetitionEntrant } from '../src/competition'
import { choice, type Scores } from '../src/game'
import { presets } from '../src/roster'
import { MAX_BYTES, validateArenaModel, type ModelStats } from '../src/validate'
import { readSubmissionRoster } from './submission-roster'

export interface ValidatedModel {
  bytes: Uint8Array
  entrant: CompetitionEntrant
  stats: ModelStats
}

/** Freeze and validate every submission before seeds are selected or games start. */
export async function validateModelCollection(
  directory: string | URL = new URL('../public/models/', import.meta.url),
  roster: readonly { id: string; name: string }[] = presets,
): Promise<ValidatedModel[]> {
  const folder = typeof directory === 'string' ? pathToFileURL(resolve(directory) + '/') : directory
  const directoryInfo = await lstat(resolve(fileURLToPath(folder)))
  if (!directoryInfo.isDirectory() || directoryInfo.isSymbolicLink())
    throw new Error('The submitted models directory must be a regular directory, without links.')
  const ids = new Set<string>()
  for (const entry of roster) {
    if (typeof entry.id !== 'string' || !/^[A-Za-z0-9_-]+$/.test(entry.id) || basename(entry.id) !== entry.id ||
      ids.has(entry.id) || typeof entry.name !== 'string' || !entry.name.trim())
      throw new Error('The active roster needs unique file-safe model IDs and nonempty names.')
    ids.add(entry.id)
  }
  const entries = await readdir(folder, { withFileTypes: true })
  if (entries.some((entry) => entry.isDirectory() || entry.isSymbolicLink()))
    throw new Error('Store model submissions as regular .onnx files directly in public/models, without subdirectories or links.')
  const files = entries.filter((entry) => entry.name.toLowerCase().endsWith('.onnx')).map((entry) => entry.name).sort()
  if (JSON.stringify(files) !== JSON.stringify(roster.map((entry) => entry.id + '.onnx').sort()))
    throw new Error('The active roster and public/models files differ. Register every model submission in src/roster.ts.')
  const hashes = new Set<string>(), models: ValidatedModel[] = []
  for (const preset of roster) {
    try {
      const path = new URL(preset.id + '.onnx', folder), info = await stat(path)
      if (!info.isFile() || info.size < 1 || info.size > MAX_BYTES)
        throw new Error('Choose a regular ONNX file up to 256 KiB.')
      const bytes = await readFile(path)
      const stats = validateArenaModel(bytes)
      if (!smallestDivision(stats)) throw new Error('The model exceeds both competition classes.')
      const hash = await sha256(bytes)
      if (hashes.has(hash)) throw new Error('Identical model bytes cannot enter the league twice.')
      hashes.add(hash)
      models.push({ bytes, stats, entrant: {
        id: preset.id, name: preset.name, sha256: hash, params: stats.params, macs: stats.macs,
      } })
    } catch (error) {
      throw new Error(`${preset.id}.onnx: ${error instanceof Error ? error.message : String(error)}`)
    }
  }
  return models
}

/** Runtime validity checks supplement structural validation; league inference also checks every decision. */
export async function smokeTestModels(models: readonly ValidatedModel[]) {
  ort.env.wasm.numThreads = 1
  // Contact-plane, wall and velocity extremes can expose overflow that a zero
  // input would miss. These are legal observations, independent of league seeds.
  const inputs: Float32Array[] = [new Float32Array(6)]
  for (const x of [-11 / 12, 0, 11 / 12])
    for (const y of [-.95, 0, .95])
      for (const vx of [-13 / 6, -5 / 6, 5 / 6, 13 / 6])
        for (const vy of [-1.5, 0, 1.5])
          for (const own of [-.8125, .8125])
            inputs.push(Float32Array.of(x, y, vx, vy, own, -own))
  for (const model of models) {
    const session = await ort.InferenceSession.create(model.bytes, { executionProviders: ['wasm'] })
    try {
      if (session.inputNames.length !== 1 || session.inputNames[0] !== 'observation' ||
        session.outputNames.length !== 1 || session.outputNames[0] !== 'action_scores')
        throw new Error('Expected only observation input and action_scores output.')
      for (const input of inputs) {
        const result = await session.run({ observation: new ort.Tensor('float32', input, [1, 6]) })
        const output = result.action_scores
        if (!output || output.type !== 'float32' || output.dims.length !== 2 ||
          output.dims[0] !== 1 || output.dims[1] !== 3 || output.data.length !== 3)
          throw new Error('Expected float32 action_scores with exact shape [1, 3].')
        choice(Array.from(output.data as Float32Array) as Scores)
      }
    } catch (error) {
      throw new Error(`${model.entrant.id}.onnx: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      await session.release()
    }
  }
}

async function main() {
  const { values } = parseArgs({ options: {
    'models-dir': { type: 'string' }, roster: { type: 'string' },
  } })
  const roster = values.roster ? await readSubmissionRoster(values.roster) : undefined
  const models = await validateModelCollection(values['models-dir'], roster)
  await smokeTestModels(models)
  for (const { entrant, stats } of models)
    console.log(`${entrant.id}: ${entrant.sha256} · ${entrant.params} parameters · ${entrant.macs} MACs · ${stats.architecture.widths.join(' → ')} · ${smallestDivision(stats)}`)
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href)
  main().catch((error) => { console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1 })

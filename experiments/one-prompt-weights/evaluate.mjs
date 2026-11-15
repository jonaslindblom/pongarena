import { randomInt } from 'node:crypto'
import { readFile, writeFile } from 'node:fs/promises'
import { basename } from 'node:path'
import * as ort from 'onnxruntime-web/wasm'
import { encode, network } from '../../tests/modelFactory.ts'
import { runCompetition, sha256, smallestDivision, standings } from '../../src/competition.ts'
import { competitionMatch } from '../../src/competitionMatch.ts'
import { leagueScoreIntervals } from '../../src/leagueUncertainty.ts'
import { PONG, choice } from '../../src/rules.ts'
import { validateArenaModel } from '../../src/validate.ts'

const usage = `Usage:
  node --import tsx experiments/one-prompt-weights/evaluate.mjs seeds SEEDS.json COUNT
  node --import tsx experiments/one-prompt-weights/evaluate.mjs compile WIDTHS Relu|Tanh WEIGHTS.json OUTPUT.onnx
  node --import tsx experiments/one-prompt-weights/evaluate.mjs evaluate SEEDS.json CANDIDATE.onnx OPPONENT [OPPONENT...]

WIDTHS is comma-delimited, e.g. 6,6,4,3,3. WEIGHTS.json is a completed attempt record or literal {layers:[{weights,bias}, ...]}.
Use one output neuron per weights row and one input coefficient per column, with a flat bias array. OPPONENT is a bundled model ID or .onnx path.
Each result uses the same seed set with sides swapped and reports the paired W/D/L, league score and rally totals, plus raw results.`

function validFloats(values, length) {
  return Array.isArray(values) && values.length === length && values.every(value =>
    typeof value === 'number' && Number.isFinite(Math.fround(value)))
}

function float32Bytes(values) {
  const raw = new Uint8Array(4 * values.length)
  const view = new DataView(raw.buffer)
  for (const [index, value] of values.entries()) view.setFloat32(index * 4, value, true)
  return raw
}

async function compile(widthsArg, activation, weightsPath, output) {
  const widths = widthsArg.split(',').map(Number)
  if (widths.length < 2 || widths.length > 5 || widths[0] !== 6 || widths.at(-1) !== 3 ||
      widths.some(width => !Number.isInteger(width) || width < 1 || width > 256) || !['Relu', 'Tanh'].includes(activation))
    throw new Error('Expected legal widths and uniform Relu or Tanh activation.')
  let data = JSON.parse(await readFile(weightsPath, 'utf8'))
  if (data && Object.hasOwn(data, 'outputText')) {
    if (data.status !== 'completed' || typeof data.outputText !== 'string')
      throw new Error('Only completed attempts with a literal JSON response can be compiled.')
    if (JSON.stringify(data.widths) !== JSON.stringify(widths))
      throw new Error('The requested architecture does not match the recorded attempt.')
    data = JSON.parse(data.outputText)
  }
  const layers = data?.layers
  if (!Array.isArray(layers) || layers.length !== widths.length - 1)
    throw new Error(`Expected ${widths.length - 1} dense layers.`)
  const model = network(widths, activation)
  model.graph.name = 'One-shot weights'
  for (const [index, layer] of layers.entries()) {
    const input = widths[index], output = widths[index + 1]
    if (!Array.isArray(layer?.weights) || layer.weights.length !== output ||
        layer.weights.some(row => !validFloats(row, input)) || !validFloats(layer.bias, output))
      throw new Error(`Layer ${index + 1} needs ${output}×${input} finite float32 weights and ${output} biases.`)
    model.graph.initializer[2 * index].rawData = float32Bytes(layer.weights.flat())
    model.graph.initializer[2 * index + 1].rawData = float32Bytes(layer.bias)
  }
  const bytes = encode(model)
  const stats = validateArenaModel(bytes)
  if (!smallestDivision(stats)) throw new Error('The candidate exceeds the arena parameter or MAC budget.')
  await writeFile(output, bytes, { flag: 'wx' })
  console.log(JSON.stringify({ file: output, sha256: await sha256(bytes), stats }))
}

async function createSeeds(file, countArg) {
  const count = Number(countArg)
  if (!Number.isInteger(count) || count < 1 || count > 512) throw new Error('Expected 1–512 seeds.')
  const seeds = new Set()
  while (seeds.size < count) seeds.add(randomInt(0, 0x100000000))
  await writeFile(file, JSON.stringify([...seeds].sort((a, b) => a - b)) + '\n', { flag: 'wx' })
  console.log(`Saved ${count} independent serve seeds to ${file}`)
}

async function openModel(file) {
  const bytes = await readFile(file)
  const stats = validateArenaModel(bytes)
  const id = basename(file, '.onnx')
  const entrant = { id, name: id, sha256: await sha256(bytes), params: stats.params, macs: stats.macs }
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
  async function decide(input) {
    const result = await session.run({ observation: new ort.Tensor('float32', input, [1, PONG.inputSize]) })
    const scores = Array.from(result.action_scores.data)
    choice(scores)
    return scores
  }
  return { entrant, session, decide, stats }
}

async function evaluate(seedFile, candidateFile, opponents) {
  ort.env.wasm.numThreads = 1
  const seeds = JSON.parse(await readFile(seedFile, 'utf8'))
  const candidate = await openModel(candidateFile)
  const matchups = []
  try {
    for (const spec of opponents) {
      const file = spec.endsWith('.onnx') ? spec : `public/models/${spec}.onnx`
      const opponent = await openModel(file)
      try {
        const decisions = new Map([[candidate.entrant.sha256, candidate.decide], [opponent.entrant.sha256, opponent.decide]])
        const run = await runCompetition({
          rulesId: PONG.id, contractId: PONG.contractId, division: 'all', seeds,
          entrants: [candidate.entrant, opponent.entrant],
          playMatch: ({ left, right, seed, signal }) => competitionMatch(
            [decisions.get(left.sha256), decisions.get(right.sha256)], PONG, seed, signal),
        })
        if (run.status !== 'complete') throw new Error(run.error ?? 'Evaluation did not complete.')
        const record = standings(run).find(row => row.entrant.sha256 === candidate.entrant.sha256)
        const interval = leagueScoreIntervals(run).get(candidate.entrant.sha256)
        const result = { opponent: opponent.entrant, record, interval, run }
        matchups.push(result)
        console.error(`${candidate.entrant.id} vs ${opponent.entrant.id}: ${record.wins}/${record.draws}/${record.losses} W/D/L, ${record.leaguePoints}/${record.played} points, rallies ${record.rallyPointsFor}/${record.rallyDraws}/${record.rallyPointsAgainst}`)
      } finally {
        await opponent.session.release()
      }
    }
  } finally {
    await candidate.session.release()
  }
  console.log(JSON.stringify({ candidate: candidate.entrant, stats: candidate.stats, seedFile, matchups }, null, 2))
}

const [command, ...args] = process.argv.slice(2)
try {
  if (command === 'seeds' && args.length === 2) await createSeeds(args[0], args[1])
  else if (command === 'compile' && args.length === 4) await compile(args[0], args[1], args[2], args[3])
  else if (command === 'evaluate' && args.length >= 3) await evaluate(args[0], args[1], args.slice(2))
  else throw new Error(usage)
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error))
  process.exitCode = 1
}

/** Research-only parallel replay of an existing frozen evaluation plan.
 * Every game uses the unchanged generateLeague engine and one-thread WASM.
 * This wrapper never draws seeds, edits models, or changes the trusted runner.
 */
import assert from 'node:assert/strict'
import { fork, type ChildProcess } from 'node:child_process'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'
import { benchmarkIdentity, restoreCompetition, type CompetitionRun, type CompetitionSpec } from '../../src/competition'
import { PONG } from '../../src/rules'
import { generateLeague, leagueMarkdown } from '../../scripts/generate-league'
import { readLeagueInputs, validateEvaluationPlan, type EvaluationPlan, type LeagueInputOptions,
  type LeagueInputs } from '../../scripts/evaluation-plan'
import { readSubmissionRoster } from '../../scripts/submission-roster'

interface Task {
  planPath: string
  options: LeagueInputOptions
  seeds: number[]
  index: number
  checkpointDirectory: string
}
type WorkerMessage = { kind: 'progress'; completed: number; total: number; elapsed: number } |
  { kind: 'complete'; path: string } | { kind: 'failed'; error: string }

function send(message: WorkerMessage) {
  return new Promise<void>((accept, reject) => {
    if (!process.send) return reject(new Error('Worker IPC channel is unavailable'))
    process.send(message, error => error ? reject(error) : accept())
  })
}

async function frozen(task: Pick<Task, 'planPath' | 'options'>) {
  const inputs = await readLeagueInputs(task.options)
  const plan = validateEvaluationPlan(JSON.parse(await readFile(task.planPath, 'utf8')), inputs)
  return { inputs, plan }
}

async function mergeRuns(parts: CompetitionRun[], seeds: number[], inputs: LeagueInputs) {
  const spec: CompetitionSpec = {
    rulesId: PONG.id, contractId: PONG.contractId, division: 'all', seeds: [...seeds], entrants: inputs.entrants,
  }
  const contests = parts.flatMap(part => part.contests).sort((left, right) =>
    left.a.localeCompare(right.a) || left.b.localeCompare(right.b) || left.seed - right.seed)
  const totalMatches = inputs.entrants.length * (inputs.entrants.length - 1) * seeds.length
  const run: CompetitionRun = {
    protocol: 'paired-round-robin', benchmark: await benchmarkIdentity(spec), spec,
    status: 'complete', contests, completedMatches: totalMatches, totalMatches,
  }
  // Canonicalization checks the full benchmark, all paired games, duplicates,
  // every expected seed/model pair, and the final match count.
  return restoreCompetition(run)
}

async function checkpoint(path: string, seeds: number[], inputs: LeagueInputs) {
  let saved: string
  try { saved = await readFile(path, 'utf8') } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined
    throw error
  }
  const run = await restoreCompetition(JSON.parse(saved))
  assert.deepEqual(run.spec.seeds, [...seeds].sort((a, b) => a - b), 'Checkpoint seed set differs')
  assert.deepEqual(run.spec.entrants, [...inputs.entrants].sort((a, b) => a.sha256.localeCompare(b.sha256)),
    'Checkpoint roster metadata differs from the frozen inputs')
  const expected = await benchmarkIdentity({ rulesId: PONG.id, contractId: PONG.contractId,
    division: 'all', seeds: [...seeds], entrants: inputs.entrants })
  assert.equal(run.benchmark, expected, 'Checkpoint differs from the frozen input roster')
  return run
}

async function work(task: Task) {
  const { inputs, plan } = await frozen(task)
  assert(task.seeds.length > 0 && new Set(task.seeds).size === task.seeds.length)
  assert(task.seeds.every(seed => plan.seeds.includes(seed)), 'Workers may only replay frozen plan seeds')
  const parts: CompetitionRun[] = []
  const started = performance.now()
  let completed = 0
  const progress = () => process.send?.({ kind: 'progress', completed, total: task.seeds.length,
    elapsed: (performance.now() - started) / 1000 } satisfies WorkerMessage)
  // Four-seed chunks give useful progress while amortizing session startup.
  const heartbeat = setInterval(progress, 15000)
  try {
    await mkdir(task.checkpointDirectory, { recursive: true })
    progress()
    for (let offset = 0; offset < task.seeds.length; offset += 4) {
      const seeds = task.seeds.slice(offset, offset + 4)
      const path = resolve(task.checkpointDirectory, `${plan.sha256}-p${task.index}-${seeds.join('-')}.json`)
      let run = await checkpoint(path, seeds, inputs)
      if (!run) {
        run = await generateLeague(seeds, inputs)
        // Publish each completed chunk atomically. A crash cannot replace a
        // valid checkpoint with a partial JSON file.
        const temporary = `${path}.${process.pid}.tmp`
        await writeFile(temporary, JSON.stringify(run) + '\n', { flag: 'wx' })
        const { rename } = await import('node:fs/promises')
        await rename(temporary, path)
      }
      parts.push(run)
      completed += seeds.length
      progress()
    }
    const after = await frozen(task)
    assert.equal(after.inputs.identity, inputs.identity, 'Frozen inputs changed during worker replay')
    assert.equal(after.plan.sha256, plan.sha256, 'Recorded plan changed during worker replay')
    return mergeRuns(parts, task.seeds, inputs)
  } finally {
    clearInterval(heartbeat)
  }
}

async function parallel(planPath: string, options: LeagueInputOptions, seeds: number[], workers: number,
                        inputs: LeagueInputs, checkpointDirectory: string) {
  const partitions = Array.from({ length: Math.min(workers, seeds.length) }, () => [] as number[])
  seeds.forEach((seed, index) => partitions[index % partitions.length].push(seed))
  const children: ChildProcess[] = []
  try {
    const results = await Promise.all(partitions.map((partition, index) => new Promise<CompetitionRun>((accept, reject) => {
      const child = fork(fileURLToPath(import.meta.url), ['--worker'], {
        execArgv: ['--import', 'tsx'], stdio: ['ignore', 'inherit', 'inherit', 'ipc'],
      })
      children.push(child)
      let completed = false
      child.on('message', (message: WorkerMessage) => {
        if (message.kind === 'progress') {
          console.log(`Worker ${index + 1}/${partitions.length}: ${message.completed}/${message.total} frozen seeds; ${message.elapsed.toFixed(1)}s`)
        } else if (message.kind === 'complete') {
          completed = true
          checkpoint(message.path, partition, inputs).then(run => {
            assert(run, 'Worker result file is missing')
            accept(run)
          }).catch(reject)
        } else {
          reject(new Error(`Worker ${index + 1}: ${message.error}`))
        }
      })
      child.on('error', reject)
      child.on('exit', (code, signal) => {
        if (!completed) reject(new Error(`Worker ${index + 1} exited ${code ?? signal} before completing`))
      })
      child.send({ planPath, options, seeds: partition, index, checkpointDirectory } satisfies Task)
    })))
    return await mergeRuns(results, seeds, inputs)
  } finally {
    for (const child of children) if (child.connected) child.disconnect()
    for (const child of children) if (child.exitCode === null) child.kill()
  }
}

async function main() {
  const { values } = parseArgs({ options: {
    plan: { type: 'string' }, output: { type: 'string' }, workers: { type: 'string', default: '4' },
    'models-dir': { type: 'string' }, roster: { type: 'string' },
    'verify-only': { type: 'boolean' }, worker: { type: 'boolean' },
  } })
  if (values.worker) {
    process.once('message', async (task: Task) => {
      try {
        const run = await work(task)
        const path = resolve(task.checkpointDirectory, `worker-${task.index}-${process.pid}.json`)
        await writeFile(path, JSON.stringify(run) + '\n', { flag: 'wx' })
        await send({ kind: 'complete', path })
      } catch (error) {
        await send({ kind: 'failed', error: error instanceof Error ? error.stack ?? error.message : String(error) })
        process.exitCode = 1
      } finally {
        if (process.connected) process.disconnect?.()
      }
    })
    return
  }
  assert(values.plan, '--plan must name an already-recorded plan')
  assert(values.output || values['verify-only'], '--output is required for a full replay')
  const workers = Number(values.workers)
  assert(Number.isInteger(workers) && workers >= 1 && workers <= 16, 'Choose 1–16 workers')
  const options: LeagueInputOptions = {
    modelsDirectory: values['models-dir'] ? resolve(values['models-dir']) : undefined,
    roster: values.roster ? await readSubmissionRoster(resolve(values.roster)) : undefined,
  }
  const planPath = resolve(values.plan)
  const { inputs, plan } = await frozen({ planPath, options })
  const started = performance.now()
  if (values['verify-only']) {
    const seeds = plan.seeds.slice(0, 2)
    assert.equal(seeds.length, 2, 'Equivalence verification needs two frozen seeds')
    console.log(`Verifying serial/parallel equivalence on existing seeds ${seeds.join(', ')}`)
    const serial = await generateLeague(seeds, inputs)
    const { mkdtemp } = await import('node:fs/promises')
    const verificationDirectory = await mkdtemp(resolve(tmpdir(), 'pongarena-wasm-verification-'))
    const replayed = await parallel(planPath, options, seeds, Math.min(2, workers), inputs, verificationDirectory)
    assert.equal(JSON.stringify(replayed), JSON.stringify(serial), 'Serial and parallel league JSON differ')
    assert.equal(leagueMarkdown(replayed), leagueMarkdown(serial), 'Serial and parallel Markdown differ')
    const after = await frozen({ planPath, options })
    assert.equal(after.inputs.identity, inputs.identity)
    assert.equal(after.plan.sha256, plan.sha256)
    console.log(`Exact serial/parallel JSON and Markdown equivalence verified: ${serial.totalMatches} games, benchmark ${serial.benchmark}`)
    if (values.output) {
      const path = resolve(values.output)
      await mkdir(dirname(path), { recursive: true })
      await writeFile(path, JSON.stringify({ schema: 'pongarena-parallel-equivalence-v1', planSha256: plan.sha256,
        benchmark: serial.benchmark, seeds, games: serial.totalMatches, exactJson: true, exactMarkdown: true }, null, 2) + '\n', { flag: 'wx' })
    }
    return
  }
  const run = await parallel(planPath, options, plan.seeds, workers, inputs, resolve(values.output!, 'checkpoints'))
  const after = await frozen({ planPath, options })
  assert.equal(after.inputs.identity, inputs.identity, 'Frozen inputs changed during full replay')
  assert.equal(after.plan.sha256, plan.sha256, 'Recorded plan changed during full replay')
  validateEvaluationPlan(plan, after.inputs)
  const directory = resolve(values.output!)
  await mkdir(directory, { recursive: true })
  await writeFile(resolve(directory, 'league.json'), JSON.stringify(run, null, 2) + '\n', { flag: 'wx' })
  await writeFile(resolve(directory, 'league.md'), leagueMarkdown(run, plan), { flag: 'wx' })
  await writeFile(resolve(directory, 'evaluation-plan.json'), JSON.stringify(plan, null, 2) + '\n', { flag: 'wx' })
  console.log(`Replayed ${run.totalMatches} games in ${((performance.now() - started) / 1000).toFixed(1)}s; benchmark ${run.benchmark}`)
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(error => { console.error(error); process.exitCode = 1 })
}

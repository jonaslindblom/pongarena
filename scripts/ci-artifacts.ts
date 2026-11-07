/** Select immutable plans and accepted PR results from the trusted evaluator. */
import { appendFile, readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'
import { EVALUATOR_FILES } from './evaluation-plan'

const WORKFLOW = '.github/workflows/model-evaluation.yml'
export type GetJson = (path: string) => Promise<unknown>

interface Artifact {
  id: number
  name: string
  expired: boolean
  created_at: string
  workflow_run?: { id: number }
}

interface WorkflowRun {
  id: number
  path: string
  event: string
  status?: string
  conclusion?: string | null
  pull_requests?: { number: number }[]
  head_repository?: { full_name: string } | null
  head_sha?: string
  head_branch?: string
  display_title?: string
}

interface PullRequest {
  number: number
  merged_at?: string | null
  merge_commit_sha?: string | null
}

export interface ArtifactSelection {
  found: boolean
  bootstrap: boolean
  artifactId?: number
  artifactName?: string
  runId?: number
  prNumber?: number
}

function validateInputs(repository: string, identity: string) {
  if (!/^[\w.-]+\/[\w.-]+$/.test(repository) || !/^[a-f0-9]{64}$/.test(identity))
    throw new Error('Expected a repository owner/name and SHA-256 evaluation identity.')
}

function checkPrNumber(prNumber: number) {
  if (!Number.isSafeInteger(prNumber) || prNumber < 1)
    throw new Error('Expected a positive pull request number.')
}

async function artifactsNamed(get: GetJson, repository: string, name: string) {
  const artifacts: Artifact[] = []
  for (let page = 1; ; page++) {
    const value = await get(`/repos/${repository}/actions/artifacts?name=${encodeURIComponent(name)}&per_page=100&page=${page}`) as
      { artifacts?: Artifact[] }
    if (!Array.isArray(value.artifacts)) throw new Error('GitHub returned an invalid artifact listing.')
    artifacts.push(...value.artifacts.filter(artifact => artifact.name === name))
    if (value.artifacts.length < 100) break
  }
  return artifacts
}

export function trustedRun(run: WorkflowRun, repository: string, prNumber: number, successful: boolean) {
  return run.event === 'pull_request_target' && run.path?.split('@')[0] === WORKFLOW &&
    (!run.pull_requests?.length || run.pull_requests.some(pr => pr.number === prNumber)) &&
    (!successful || (run.conclusion === 'success' && (!run.status || run.status === 'completed')))
}

export function trustedPagesRun(run: WorkflowRun) {
  return ['push', 'workflow_dispatch'].includes(run.event) &&
    run.path?.split('@')[0] === '.github/workflows/pages.yml' && run.head_branch === 'main'
}

function canonicalArtifact(artifacts: Artifact[], name: string, prNumber?: number): ArtifactSelection | undefined {
  // The oldest trusted plan/results are canonical. A later upload must not
  // replace an expired plan with a newly sampled, more favorable seed set.
  artifacts.sort((a, b) => Date.parse(a.created_at) - Date.parse(b.created_at) || a.id - b.id)
  const selected = artifacts[0]
  if (!selected) return undefined
  if (selected.expired)
    throw new Error(`The recorded ${name} artifact expired. Refusing to sample replacement serves; restore the saved artifact or have a maintainer start a documented new evaluation.`)
  return { found: true, bootstrap: false, artifactId: selected.id,
    artifactName: selected.name, runId: selected.workflow_run!.id, ...(prNumber === undefined ? {} : { prNumber }) }
}

async function selectArtifact(get: GetJson, repository: string, prNumber: number,
  name: string, successful: boolean): Promise<ArtifactSelection | undefined> {
  const artifacts = await artifactsNamed(get, repository, name)
  const trusted: Artifact[] = []
  for (const artifact of artifacts) {
    if (!Number.isSafeInteger(artifact.id) || !artifact.workflow_run?.id) continue
    const run = await get(`/repos/${repository}/actions/runs/${artifact.workflow_run.id}`) as WorkflowRun
    if (trustedRun(run, repository, prNumber, successful)) trusted.push(artifact)
  }
  return canonicalArtifact(trusted, name, prNumber)
}

async function selectPagesArtifact(get: GetJson, repository: string, name: string) {
  const artifacts = await artifactsNamed(get, repository, name)
  const trusted: Artifact[] = []
  for (const artifact of artifacts) {
    if (!Number.isSafeInteger(artifact.id) || !artifact.workflow_run?.id) continue
    const run = await get(`/repos/${repository}/actions/runs/${artifact.workflow_run.id}`) as WorkflowRun
    // Deployment can fail after validation/build uploaded the immutable league.
    // Reuse it regardless of the later deployment conclusion.
    if (trustedPagesRun(run)) trusted.push(artifact)
  }
  return canonicalArtifact(trusted, name)
}

export async function findPlan(get: GetJson, options: {
  repository: string; identity: string; prNumber: number; runAttempt?: number
  headSha?: string; baseSha?: string; runId?: number
  prCreatedAt?: string
}): Promise<ArtifactSelection> {
  validateInputs(options.repository, options.identity)
  checkPrNumber(options.prNumber)
  const selected = await selectArtifact(get, options.repository, options.prNumber,
    `evaluation-plan-${options.prNumber}-${options.identity}`, false)
  if (selected) return selected
  if ((options.runAttempt ?? 1) > 1)
    throw new Error('The saved evaluation plan is unavailable on this retry. Refusing to reroll serves.')
  if (options.headSha && options.baseSha)
    await rejectMissingHistoricalPlan(get, options.repository, options.prNumber,
      options.headSha, options.baseSha, options.runId, options.prCreatedAt)
  return { found: false, bootstrap: false, prNumber: options.prNumber }
}

async function readTree(get: GetJson, repository: string, sha: string) {
  const value = await get(`/repos/${repository}/git/trees/${sha}?recursive=1`) as
    { truncated?: boolean; tree?: { path: string; type: string; sha: string }[] }
  if (value.truncated || !Array.isArray(value.tree))
    throw new Error('Cannot inspect the complete prior model submission. Refusing to reroll serves.')
  return value.tree
}

function treeIdentities(get: GetJson, repository: string) {
  const trees = new Map<string, Awaited<ReturnType<typeof readTree>>>()
  const tree = async (sha: string) => {
    if (!trees.has(sha)) trees.set(sha, await readTree(get, repository, sha))
    return trees.get(sha)!
  }
  const modelTreeIdentity = async (sha: string) => JSON.stringify((await tree(sha)).filter(entry =>
    entry.type === 'blob' && entry.path.startsWith('public/models/') && entry.path.endsWith('.onnx'))
    .map(entry => entry.sha).sort())
  const evaluatorTreeIdentity = async (sha: string) => {
    const files = new Map((await tree(sha)).filter(entry => entry.type === 'blob').map(entry => [entry.path, entry.sha]))
    return JSON.stringify(EVALUATOR_FILES.map(path => [path, files.get(path) ?? null]))
  }
  return { modelTreeIdentity, evaluatorTreeIdentity }
}

async function runPersistedPlan(get: GetJson, repository: string, runId: number, main: boolean) {
  const persistenceName = main ? 'Persist the trusted-main plan before evaluation'
    : 'Persist the serve plan before running any matches'
  const evaluationName = main ? 'Bootstrap a trusted-main evaluation for infrastructure changes'
    : 'Evaluate the complete proposed roster on paired serves'
  type Step = { name: string; status?: string; conclusion?: string | null }
  const stages: Step[] = []
  for (let page = 1; ; page++) {
    const listing = await get(`/repos/${repository}/actions/runs/${runId}/jobs?filter=latest&per_page=100&page=${page}`) as
      { jobs?: { steps?: Step[] }[] }
    if (!Array.isArray(listing.jobs) || listing.jobs.some(job => !Array.isArray(job.steps)))
      throw new Error('Prior evaluation job stages are unavailable. Refusing to reroll serves.')
    stages.push(...listing.jobs.flatMap(job => job.steps!))
    if (listing.jobs.length < 100) break
    if (page === 10) throw new Error('Prior evaluation job history may be truncated. Refusing to reroll serves.')
  }
  const persistence = stages.find(step => step.name === persistenceName)
  const evaluation = stages.find(step => step.name === evaluationName)
  if (!persistence || !evaluation)
    throw new Error('Prior evaluation does not expose its plan and match stages. Refusing to reroll serves.')
  if (persistence.conclusion === 'success' || (evaluation.status === 'in_progress') ||
    (evaluation.status === 'completed' && evaluation.conclusion !== 'skipped')) return true
  if (evaluation.conclusion === 'skipped' && persistence.status === 'completed' &&
    ['skipped', 'failure', 'cancelled'].includes(persistence.conclusion ?? '')) return false
  throw new Error('Prior evaluation stage state is incomplete. Refusing to reroll serves.')
}

async function rejectMissingHistoricalPlan(get: GetJson, repository: string, prNumber: number,
  headSha: string, baseSha: string, currentRunId?: number, prCreatedAt?: string) {
  if (![headSha, baseSha].every(value => /^[a-f0-9]{40,64}$/.test(value)))
    throw new Error('Expected immutable candidate and base commit SHAs.')
  const { modelTreeIdentity, evaluatorTreeIdentity } = treeIdentities(get, repository)
  if (prCreatedAt !== undefined && !Number.isFinite(Date.parse(prCreatedAt)))
    throw new Error('The PR creation timestamp is invalid; cannot bound its evaluation history.')
  const createdFilter = prCreatedAt === undefined ? ''
    : `&created=${encodeURIComponent('>=' + new Date(prCreatedAt).toISOString())}`
  let currentModels: string | undefined
  let currentEvaluator: string | undefined
  for (let page = 1; ; page++) {
    const listing = await get(`/repos/${repository}/actions/workflows/model-evaluation.yml/runs?event=pull_request_target${createdFilter}&per_page=100&page=${page}`) as
      { workflow_runs?: WorkflowRun[] }
    if (!Array.isArray(listing.workflow_runs)) throw new Error('GitHub returned invalid evaluation run history.')
    for (const run of listing.workflow_runs) {
      const title = /^Model PR #(\d+) at ([a-f0-9]{40,64})$/.exec(run.display_title ?? '')
      if (run.id === currentRunId || !title || Number(title[1]) !== prNumber ||
        !trustedRun(run, repository, prNumber, false)) continue
      if (!run.head_sha || !/^[a-f0-9]{40,64}$/.test(run.head_sha))
        throw new Error('Prior evaluation lacks a verifiable base commit. Refusing to reroll serves.')
      if (run.head_sha !== baseSha) {
        currentEvaluator ??= await evaluatorTreeIdentity(baseSha)
        if (currentEvaluator !== await evaluatorTreeIdentity(run.head_sha)) continue
      }
      currentModels ??= await modelTreeIdentity(headSha)
      const previousModels = title[2] === headSha ? currentModels : await modelTreeIdentity(title[2])
      if (previousModels === currentModels && await runPersistedPlan(get, repository, run.id, false))
        throw new Error('An earlier run evaluated these same model bytes, but its saved plan is unavailable. Refusing to reroll serves; restore the recorded plan or request a documented maintainer evaluation.')
    }
    if (listing.workflow_runs.length < 100) return
    if (page === 10)
      throw new Error('Evaluation run history may be truncated. Refusing to sample potentially repeated submissions.')
  }
}

export async function findMainPlan(get: GetJson, options: {
  repository: string; identity: string; sha: string; runAttempt?: number; runId?: number
}): Promise<ArtifactSelection> {
  validateInputs(options.repository, options.identity)
  if (!/^[a-f0-9]{40,64}$/.test(options.sha)) throw new Error('Expected an immutable main commit SHA.')
  const selected = await selectPagesArtifact(get, options.repository, `main-evaluation-plan-${options.identity}`)
  if (selected) return selected
  if ((options.runAttempt ?? 1) > 1)
    throw new Error('The saved main evaluation plan is unavailable on this retry. Refusing to reroll serves.')
  const { modelTreeIdentity, evaluatorTreeIdentity } = treeIdentities(get, options.repository)
  let currentModels: string | undefined, currentEvaluator: string | undefined
  for (let page = 1; ; page++) {
    const listing = await get(`/repos/${options.repository}/actions/workflows/pages.yml/runs?per_page=100&page=${page}`) as
      { workflow_runs?: WorkflowRun[] }
    if (!Array.isArray(listing.workflow_runs)) throw new Error('GitHub returned invalid main evaluation history.')
    for (const run of listing.workflow_runs) {
      if (run.id === options.runId || !trustedPagesRun(run)) continue
      if (!run.head_sha || !/^[a-f0-9]{40,64}$/.test(run.head_sha))
        throw new Error('Prior publication lacks a verifiable main commit. Refusing to reroll serves.')
      currentEvaluator ??= await evaluatorTreeIdentity(options.sha)
      if (currentEvaluator !== await evaluatorTreeIdentity(run.head_sha)) continue
      currentModels ??= await modelTreeIdentity(options.sha)
      if (currentModels === await modelTreeIdentity(run.head_sha) &&
        await runPersistedPlan(get, options.repository, run.id, true))
        throw new Error('An earlier main run used these same model bytes and evaluator, but its saved plan is unavailable. Refusing to reroll serves.')
    }
    if (listing.workflow_runs.length < 100) return { found: false, bootstrap: true }
    if (page === 10) throw new Error('Main evaluation history may be truncated. Refusing to reroll serves.')
  }
}

interface ChangedFile { filename: string; previous_filename?: string }

export function changesModels(files: readonly ChangedFile[]) {
  const modelPath = (path: string) => path === 'src/roster.ts' || path === 'public/models' || path.startsWith('public/models/')
  return files.some(file => modelPath(file.filename) || (file.previous_filename && modelPath(file.previous_filename)))
}

async function pullRequestFiles(get: GetJson, repository: string, prNumber: number) {
  const files: ChangedFile[] = []
  for (let page = 1; ; page++) {
    const batch = await get(`/repos/${repository}/pulls/${prNumber}/files?per_page=100&page=${page}`) as ChangedFile[]
    if (!Array.isArray(batch)) throw new Error('GitHub returned an invalid pull request file listing.')
    files.push(...batch)
    if (files.length >= 3000) throw new Error('Pull request file listing may be truncated; cannot safely decide whether models changed.')
    if (batch.length < 100) return files
  }
}

export async function checkScope(get: GetJson, options: { repository: string; prNumber: number }) {
  validateInputs(options.repository, '0'.repeat(64))
  checkPrNumber(options.prNumber)
  const files = await pullRequestFiles(get, options.repository, options.prNumber)
  const paths = files.flatMap(file => [file.filename, ...(file.previous_filename ? [file.previous_filename] : [])])
  const modelChanges = paths.some(path => path.startsWith('public/models/') && path.endsWith('.onnx'))
  const protectedPaths = new Set(['src/game.ts', 'src/rules.ts', 'src/validate.ts',
    'src/competition.ts', 'src/competitionMatch.ts', 'package.json', 'package-lock.json'])
  const evaluatorChanges = paths.some(path => protectedPaths.has(path) ||
    path.startsWith('scripts/') || path.startsWith('.github/workflows/'))
  if (modelChanges && evaluatorChanges)
    throw new Error('Submit model weights separately from evaluator, rules, runtime or workflow changes. The trusted model gate evaluates the base specification; combine neither change in one model PR.')
  return { modelChanges, evaluatorChanges }
}

export async function findResults(get: GetJson, options: {
  repository: string; identity: string; sha: string; before?: string
}): Promise<ArtifactSelection> {
  validateInputs(options.repository, options.identity)
  if (!/^[a-f0-9]{40,64}$/.test(options.sha)) throw new Error('Expected the merged commit SHA.')
  const published = await selectPagesArtifact(get, options.repository, `published-league-${options.identity}`)
  if (published) return published
  const associated = await get(`/repos/${options.repository}/commits/${options.sha}/pulls?per_page=100`) as PullRequest[]
  if (!Array.isArray(associated)) throw new Error('GitHub returned invalid commit pull request associations.')
  const merged: PullRequest[] = []
  for (const candidate of associated) {
    checkPrNumber(candidate.number)
    const pr = await get(`/repos/${options.repository}/pulls/${candidate.number}`) as PullRequest
    if (pr.merged_at && pr.merge_commit_sha === options.sha) merged.push(pr)
  }
  if (merged.length > 1) throw new Error('Multiple merged pull requests claim this commit; cannot select its evaluation.')
  const pr = merged[0]
  if (pr) {
    const selected = await selectArtifact(get, options.repository, pr.number,
      `model-league-${pr.number}-${options.identity}`, true)
    if (selected) return selected
    if (changesModels(await pullRequestFiles(get, options.repository, pr.number)))
      throw new Error(`Model PR #${pr.number} has no successful trusted evaluation for the merged model bytes and evaluator. Publication is blocked.`)
    // The initial infrastructure rollout has no earlier trusted target workflow.
    // Only a PR without model/roster edits can bootstrap from trusted main.
    return { found: false, bootstrap: true, prNumber: pr.number }
  }
  let before = options.before
  if (!before) {
    // workflow_dispatch has no push.before. Inspect the current commit's first
    // parent before deciding whether a trusted-main bootstrap is permitted.
    const commit = await get(`/repos/${options.repository}/commits/${options.sha}`) as
      { parents?: { sha: string }[] }
    if (!Array.isArray(commit.parents)) throw new Error('Cannot inspect the prior main commit for bootstrap.')
    if (!commit.parents.length) return { found: false, bootstrap: true }
    before = commit.parents[0].sha
  }
  if (!/^[a-f0-9]{40,64}$/.test(before))
    throw new Error('Cannot identify an accepted model evaluation or a prior main commit for bootstrap.')
  if (/^0+$/.test(before)) return { found: false, bootstrap: true }
  const comparison = await get(`/repos/${options.repository}/compare/${before}...${options.sha}`) as
    { files?: ChangedFile[] }
  if (!Array.isArray(comparison.files) || comparison.files.length >= 3000)
    throw new Error('The main-branch comparison is incomplete; cannot safely bootstrap publication.')
  if (changesModels(comparison.files))
    throw new Error('Direct model changes on main have no accepted PR evaluation. Publication is blocked.')
  return { found: false, bootstrap: true }
}

async function main() {
  const { values, positionals } = parseArgs({ allowPositionals: true,
    options: { identity: { type: 'string' } } })
  const command = positionals[0]
  if (positionals.length !== 1 || !['find-plan', 'find-main-plan', 'find-results', 'check-scope'].includes(command) ||
    (command !== 'check-scope' && !values.identity))
    throw new Error('Use find-plan|find-main-plan|find-results --identity SHA256, or check-scope.')
  const repository = process.env.GITHUB_REPOSITORY ?? ''
  const token = process.env.GITHUB_TOKEN
  if (!token) throw new Error('GITHUB_TOKEN is required to inspect trusted evaluation artifacts.')
  const event = JSON.parse(await readFile(process.env.GITHUB_EVENT_PATH ?? '', 'utf8'))
  const get: GetJson = async path => {
    const response = await fetch(`${process.env.GITHUB_API_URL ?? 'https://api.github.com'}${path}`, {
      headers: { Authorization: `Bearer ${token}`, Accept: 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28' },
    })
    if (!response.ok) throw new Error(`GitHub artifact lookup failed (${response.status}) for ${path}.`)
    return response.json()
  }
  const prNumber = Number(process.env.PR_NUMBER ?? event.pull_request?.number ?? event.number)
  if (command === 'check-scope') {
    const scope = await checkScope(get, { repository, prNumber })
    const output = `model-changes=${scope.modelChanges}\nevaluator-changes=${scope.evaluatorChanges}\n`
    if (process.env.GITHUB_OUTPUT) await appendFile(process.env.GITHUB_OUTPUT, output)
    else process.stdout.write(output)
    return
  }
  const selection = command === 'find-plan'
    ? await findPlan(get, { repository, identity: values.identity!,
      prNumber: Number(process.env.PR_NUMBER ?? event.pull_request?.number ?? event.number),
      runAttempt: Number(process.env.GITHUB_RUN_ATTEMPT ?? 1),
      headSha: event.pull_request?.head?.sha, baseSha: event.pull_request?.base?.sha,
      prCreatedAt: event.pull_request?.created_at,
      runId: process.env.GITHUB_RUN_ID ? Number(process.env.GITHUB_RUN_ID) : undefined })
    : command === 'find-main-plan'
    ? await findMainPlan(get, { repository, identity: values.identity!, sha: process.env.GITHUB_SHA ?? '',
      runAttempt: Number(process.env.GITHUB_RUN_ATTEMPT ?? 1),
      runId: process.env.GITHUB_RUN_ID ? Number(process.env.GITHUB_RUN_ID) : undefined })
    : await findResults(get, { repository, identity: values.identity!,
      sha: process.env.GITHUB_SHA ?? '', before: event.before })
  const output = Object.entries({ found: String(selection.found), bootstrap: String(selection.bootstrap),
    'artifact-id': selection.artifactId ?? '', 'run-id': selection.runId ?? '',
    'artifact-name': selection.artifactName ?? '',
    'pr-number': selection.prNumber ?? '' }).map(([key, value]) => `${key}=${value}`).join('\n') + '\n'
  if (process.env.GITHUB_OUTPUT) await appendFile(process.env.GITHUB_OUTPUT, output)
  else process.stdout.write(output)
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href)
  main().catch(error => { console.error(error instanceof Error ? error.message : error); process.exitCode = 1 })

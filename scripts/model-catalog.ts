import { execFile } from 'node:child_process'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseArgs, promisify } from 'node:util'
import type { CatalogEntry } from '../src/modelCatalog'
import { modelIdentity } from '../src/modelIdentity'
import { methodLabel } from '../src/modelMetadata'
import { isGitHubUsername } from '../src/submitter'
import { MAX_BYTES } from '../src/validate'
import { requireCoreModels, validateModelCollection, type ValidatedModel } from './validate-models'

const exec = promisify(execFile)

/** Search trusted main history by network identity, including renamed files and prose edits. */
export async function modelIntroductions(hashes: readonly string[], cwd = process.cwd()) {
  const introductions = new Map<string, string>()
  if (!hashes.length) return introductions
  const git = async (...args: string[]) => (await exec('git', args, { cwd, encoding: 'buffer', maxBuffer: 16 * 1024 * 1024 })).stdout
  if ((await git('rev-parse', '--is-shallow-repository')).toString().trim() !== 'false')
    throw new Error('Submitter credit requires complete trusted git history (checkout fetch-depth: 0).')
  const commits = (await git('log', '--first-parent', '--reverse', '--format=%H', '--', 'public/models/'))
    .toString().trim().split('\n').filter(Boolean)
  const identities = new Map<string, string | undefined>()
  const wanted = new Set(hashes)
  for (const commit of commits) {
    const entries = (await git('ls-tree', '-rlz', commit, '--', 'public/models/')).toString().split('\0')
    for (const entry of entries) {
      const match = /^100(?:644|755) blob ([a-f0-9]+)\s+(\d+)\tpublic\/models\/[^/]+\.onnx$/.exec(entry)
      if (!match || Number(match[2]) > MAX_BYTES) continue
      const blob = match[1]
      if (!identities.has(blob)) {
        const bytes = await git('cat-file', 'blob', blob)
        // Historical contracts may differ; only networks valid under today's contract can match.
        let identity: string | undefined
        try { identity = await modelIdentity(bytes) } catch { /* Older, incompatible graph. */ }
        identities.set(blob, identity)
      }
      const hash = identities.get(blob)
      if (hash && wanted.has(hash) && !introductions.has(hash)) introductions.set(hash, commit)
    }
    if (introductions.size === wanted.size) break
  }
  return introductions
}

type GetJson = (path: string) => Promise<unknown>

/** GitHub records submission credit; text embedded in a model cannot claim it. */
export async function submissionCredits(models: readonly ValidatedModel[], introductions: ReadonlyMap<string, string>,
  get: GetJson, repository: string, submitter?: string) {
  if (!/^[\w.-]+\/[\w.-]+$/.test(repository)) throw new Error('Expected the GitHub owner/repository for submission credit.')
  if (submitter !== undefined && !isGitHubUsername(submitter)) throw new Error('Expected the authenticated PR submitter.')
  const credits = new Map<string, string>()
  const authors = new Map<string, string>()
  for (const model of models) {
    const hash = model.entrant.sha256
    if (model.origin === 'core' && model.submittedBy) { credits.set(hash, model.submittedBy); continue }
    const commit = introductions.get(hash)
    if (!commit) {
      if (!submitter) throw new Error(`${model.entrant.id}: no accepted PR introduces these weights in trusted main history.`)
      credits.set(hash, submitter)
      continue
    }
    if (!authors.has(commit)) {
      const response = await get(`/repos/${repository}/commits/${commit}/pulls?per_page=100`)
      if (!Array.isArray(response) || response.length >= 100)
        throw new Error('Cannot verify the complete merged PR history for model attribution.')
      // The commit/PR association also covers rebase merges: the introducing
      // commit can precede merge_commit_sha (the PR's final rebased commit).
      const matches = response.filter(pr => pr.merged_at && /^[a-f0-9]{40,64}$/.test(pr.merge_commit_sha) &&
        pr.base?.repo?.full_name === repository && pr.base?.ref === 'main' && isGitHubUsername(pr.user?.login))
      if (matches.length !== 1)
        throw new Error(`Expected one merged PR introducing model weights at ${commit}; refusing to guess its submitter.`)
      authors.set(commit, matches[0].user.login)
    }
    credits.set(hash, authors.get(commit)!)
  }
  return credits
}

export function modelCatalog(models: readonly ValidatedModel[], credits: ReadonlyMap<string, string> = new Map()): CatalogEntry[] {
  return models.map(model => ({
    id: model.entrant.id, ...model.metadata, kind: methodLabel(model.metadata.method),
    ...(model.metadata.method === 'programmed' ? { handcrafted: true } : {}),
    origin: model.origin,
    ...((model.submittedBy ?? credits.get(model.entrant.sha256))
      ? { submittedBy: model.submittedBy ?? credits.get(model.entrant.sha256) } : {}),
    sha256: model.entrant.sha256, fileSha256: model.fileSha256,
  }))
}

async function main() {
  const { values } = parseArgs({ options: {
    'models-dir': { type: 'string' }, output: { type: 'string', default: 'public/models.json' },
    github: { type: 'boolean' }, submitter: { type: 'string' },
  } })
  if (values.submitter && !values.github) throw new Error('--submitter requires --github attribution.')
  const models = await validateModelCollection(values['models-dir'])
  requireCoreModels(models)
  let credits = new Map<string, string>()
  if (values.github) {
    const repository = process.env.GITHUB_REPOSITORY ?? '', token = process.env.GITHUB_TOKEN
    if (!token) throw new Error('GITHUB_TOKEN is required to verify PR attribution.')
    if (values.submitter) {
      const event = JSON.parse(await readFile(process.env.GITHUB_EVENT_PATH ?? '', 'utf8'))
      if (event.pull_request?.user?.login !== values.submitter || event.repository?.full_name !== repository)
        throw new Error('--submitter must match the authenticated pull request event.')
    }
    const introductions = await modelIntroductions(models.filter(model => model.origin === 'community').map(model => model.entrant.sha256))
    credits = await submissionCredits(models, introductions, async path => {
      const response = await fetch('https://api.github.com' + path, {
        headers: { Authorization: `Bearer ${token}`, Accept: 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28' },
      })
      if (!response.ok) throw new Error(`GitHub attribution lookup failed: HTTP ${response.status}.`)
      return response.json()
    }, repository, values.submitter)
  } else {
    // Preserve attribution produced earlier in this build by the trusted --github step.
    const previous = await readFile(values.output, 'utf8').catch(error => {
      if (error.code === 'ENOENT') return '[]'
      throw error
    })
    const entries: unknown = JSON.parse(previous)
    if (!Array.isArray(entries)) throw new Error('Expected a generated model catalog array.')
    for (const entry of entries)
      if (/^[a-f0-9]{64}$/.test(entry.sha256) && isGitHubUsername(entry.submittedBy)) credits.set(entry.sha256, entry.submittedBy)
  }
  const output = resolve(values.output)
  await mkdir(dirname(output), { recursive: true })
  await writeFile(output, JSON.stringify(modelCatalog(models, credits), null, 2) + '\n')
  console.log(`Generated ${models.length} model cards${values.github ? ' with verified PR submitter credit' : ''}.`)
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href)
  main().catch(error => { console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1 })

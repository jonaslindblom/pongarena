import test from 'node:test'
import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { mkdtemp, mkdir, rename, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'
import { modelCatalog, modelIntroductions, submissionCredits } from '../scripts/model-catalog'
import type { ValidatedModel } from '../scripts/validate-models'
import { modelIdentity } from '../src/modelIdentity'
import { sha256 } from '../src/competition'
import { validateArenaModel } from '../src/validate'
import { encode, network } from './modelFactory'

const exec = promisify(execFile)
async function candidate(): Promise<ValidatedModel> {
  const bytes = encode(network([6, 3]))
  return { bytes, stats: validateArenaModel(bytes), origin: 'community', fileSha256: await sha256(bytes),
    entrant: { id: 'candidate', name: 'Candidate', sha256: await modelIdentity(bytes), params: 21, macs: 18 },
    metadata: { name: 'Candidate', description: 'Learned from self play.', method: 'trained' } }
}

test('ONNX discovery generates a community card without a manual roster entry or claimed author', async () => {
  const model = await candidate()
  const [card] = modelCatalog([model])
  assert.equal(card.origin, 'community')
  assert.equal(card.kind, 'Trained weights')
  assert.equal(card.description, model.metadata.description)
  assert.equal(card.submittedBy, undefined)
  assert.equal(card.fileSha256, model.fileSha256)
  assert.equal(modelCatalog([model], new Map([[model.entrant.sha256, 'pr-author']]))[0].submittedBy, 'pr-author')
})

test('new networks credit the authenticated PR submitter while renamed networks keep the original accepted submitter', async () => {
  const model = await candidate(), commit = 'a'.repeat(40)
  const credits = await submissionCredits([model], new Map(), async () => assert.fail('No history to look up'), 'owner/repo', 'new-user')
  assert.equal(credits.get(model.entrant.sha256), 'new-user')
  const renamed = { ...model, entrant: { ...model.entrant, id: 'renamed' } }
  const old = await submissionCredits([renamed], new Map([[model.entrant.sha256, commit]]), async path => {
    assert.equal(path, `/repos/owner/repo/commits/${commit}/pulls?per_page=100`)
    return [{ merged_at: '2026-01-01', merge_commit_sha: commit, base: { ref: 'main', repo: { full_name: 'owner/repo' } }, user: { login: 'original-user' } }]
  }, 'owner/repo', 'different-user')
  assert.equal(old.get(model.entrant.sha256), 'original-user')
  await assert.rejects(submissionCredits([model], new Map(), async () => [], 'owner/repo'), /no accepted PR/)
})

test('attribution refuses ambiguous, unmerged, unrelated or invalid GitHub identities', async () => {
  const model = await candidate(), commit = 'a'.repeat(40)
  const merged = { merged_at: '2026-01-01', merge_commit_sha: commit, base: { ref: 'main', repo: { full_name: 'owner/repo' } }, user: { login: 'original-user' } }
  for (const response of [[], [merged, merged], [{ ...merged, merged_at: null }],
    [{ ...merged, merge_commit_sha: null }], [{ ...merged, base: { repo: { full_name: 'other/repo' } } }],
    [{ ...merged, user: { login: 'https://github.com/spoof' } }]])
    await assert.rejects(submissionCredits([model], new Map([[model.entrant.sha256, commit]]), async () => response, 'owner/repo'), /Expected one merged PR/)
  await assert.rejects(submissionCredits([model], new Map(), async () => [], 'owner/repo', 'not/a/handle'), /authenticated/)
})


test('a rebased multi-commit PR credits the model-introducing commit before its final merge commit', async () => {
  const model = await candidate(), introduction = 'a'.repeat(40)
  const credits = await submissionCredits([model], new Map([[model.entrant.sha256, introduction]]), async path => {
    assert(path.includes(introduction))
    return [{ merged_at: '2026-01-01', merge_commit_sha: 'b'.repeat(40),
      base: { ref: 'main', repo: { full_name: 'owner/repo' } }, user: { login: 'original-user' } }]
  }, 'owner/repo')
  assert.equal(credits.get(model.entrant.sha256), 'original-user')
})

test('core attribution comes from project data and needs no contributor claim', async () => {
  const model = { ...await candidate(), origin: 'core' as const, submittedBy: 'maintainer' }
  const credits = await submissionCredits([model], new Map(), async () => assert.fail('Core credit is pinned'), 'owner/repo', 'new-user')
  assert.equal(credits.get(model.entrant.sha256), 'maintainer')
})

test('trusted history finds first introduction across metadata edits and file renames; changed weights get a new introduction', async () => {
  const cwd = await mkdtemp(join(tmpdir(), 'pongarena-attribution-'))
  const git = async (...args: string[]) => (await exec('git', args, { cwd })).stdout.trim()
  try {
    await git('init', '-q')
    await mkdir(join(cwd, 'public/models'), { recursive: true })
    const model = network([6, 3]), first = encode(model), firstHash = await modelIdentity(first)
    await writeFile(join(cwd, 'public/models/candidate.onnx'), first)
    const commit = async () => {
      await git('add', 'public/models')
      await git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.test', '-c', 'commit.gpgsign=false', 'commit', '-qm', 'Model revision')
      return git('rev-parse', 'HEAD')
    }
    const initial = await commit()
    model.docString = 'Revised explanation of the same weights.'
    await writeFile(join(cwd, 'public/models/candidate.onnx'), encode(model))
    await rename(join(cwd, 'public/models/candidate.onnx'), join(cwd, 'public/models/renamed.onnx'))
    await commit()
    const tensor = model.graph.initializer[0].rawData!
    new DataView(tensor.buffer, tensor.byteOffset, tensor.byteLength).setFloat32(0, 1, true)
    await writeFile(join(cwd, 'public/models/renamed.onnx'), encode(model))
    const changed = await commit(), changedHash = await modelIdentity(encode(model))
    const found = await modelIntroductions([firstHash, changedHash, '0'.repeat(64)], cwd)
    assert.equal(found.get(firstHash), initial)
    assert.equal(found.get(changedHash), changed)
    assert.equal(found.has('0'.repeat(64)), false)
  } finally { await rm(cwd, { recursive: true, force: true }) }
})

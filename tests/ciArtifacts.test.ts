import test from 'node:test'
import assert from 'node:assert/strict'
import { findPlan, findMainPlan, findResults, changesModels, checkScope, trustedRun, trustedPagesRun, type GetJson } from '../scripts/ci-artifacts'

const repository = 'example/pongarena', identity = 'a'.repeat(64)
const sha = 'b'.repeat(40), before = 'c'.repeat(40), prNumber = 9
const planName = `evaluation-plan-${prNumber}-${identity}`
const resultName = `model-league-${prNumber}-${identity}`
const trusted = { id: 100, path: '.github/workflows/model-evaluation.yml', event: 'pull_request_target',
  conclusion: 'success', status: 'completed', pull_requests: [{ number: prNumber }],
  head_repository: { full_name: repository } }
const trustedPages = { id: 101, path: '.github/workflows/pages.yml', event: 'push',
  head_branch: 'main', head_sha: sha, status: 'completed', conclusion: 'failure' }

const planStages = (main = false, persisted = true) => ({ jobs: [{ steps: [
  { name: main ? 'Persist the trusted-main plan before evaluation' : 'Persist the serve plan before running any matches',
    status: 'completed', conclusion: persisted ? 'success' : 'skipped' },
  { name: main ? 'Bootstrap a trusted-main evaluation for infrastructure changes' : 'Evaluate the complete proposed roster on paired serves',
    status: 'completed', conclusion: persisted ? 'success' : 'skipped' },
] }] })

function fixture(options: {
  artifacts?: { id: number; name: string; expired?: boolean; created_at?: string; workflow_run?: { id: number } }[]
  runs?: Record<number, unknown>
  files?: { filename: string; previous_filename?: string }[]
  associated?: { number: number }[]
  merged?: boolean
  parents?: { sha: string }[]
} = {}): GetJson {
  return async path => {
    if (path.includes('/actions/artifacts?')) return { artifacts: (options.artifacts ?? []).map(artifact =>
      ({ expired: false, created_at: '2026-11-07T00:00:00Z', workflow_run: { id: 100 }, ...artifact })) }
    if (path.includes('/actions/runs/') && path.includes('/jobs?')) return planStages()
    if (path.includes('/actions/runs/')) return options.runs?.[Number(path.split('/').pop())] ?? trusted
    if (path.includes('/commits/') && path.includes('/pulls?')) return options.associated ?? [{ number: prNumber }]
    if (path.endsWith(`/pulls/${prNumber}`)) return { number: prNumber,
      merged_at: options.merged === false ? null : '2026-11-07T01:00:00Z', merge_commit_sha: sha }
    if (path.includes(`/pulls/${prNumber}/files?`)) return options.files ?? []
    if (path.includes('/compare/')) return { files: options.files ?? [] }
    if (path.endsWith(`/commits/${sha}`)) return { parents: options.parents ?? [{ sha: before }] }
    throw new Error(`Unexpected fixture request: ${path}`)
  }
}

test('retries recover the oldest immutable trusted plan, including one saved by a failed run', async () => {
  const result = await findPlan(fixture({ artifacts: [
    { id: 20, name: planName, created_at: '2026-11-07T00:02:00Z' },
    { id: 10, name: planName, created_at: '2026-11-07T00:01:00Z', workflow_run: { id: 101 } },
  ], runs: { 101: { ...trusted, id: 101, conclusion: 'failure' } } }),
  { repository, identity, prNumber, runAttempt: 2 })
  assert.deepEqual(result, { found: true, bootstrap: false, artifactId: 10, artifactName: planName, runId: 101, prNumber })
})

test('an expired original plan cannot be replaced by a later favorable plan', async () => {
  await assert.rejects(findPlan(fixture({ artifacts: [
    { id: 1, name: planName, expired: true, created_at: '2026-11-06T00:00:00Z' },
    { id: 2, name: planName },
  ] }), { repository, identity, prNumber }), /expired.*Refusing/)
})

test('missing plans fail closed on a rerun and initial submissions may create a plan', async () => {
  await assert.rejects(findPlan(fixture(), { repository, identity, prNumber, runAttempt: 2 }), /Refusing to reroll/)
  assert.deepEqual(await findPlan(fixture(), { repository, identity, prNumber }),
    { found: false, bootstrap: false, prNumber })
})

test('untrusted workflows, events and wrong PR associations cannot supply plans', async () => {
  for (const run of [
    { ...trusted, path: '.github/workflows/check.yml' },
    { ...trusted, event: 'pull_request' },
    { ...trusted, pull_requests: [{ number: 10 }] },
  ]) {
    const result = await findPlan(fixture({ artifacts: [{ id: 1, name: planName }], runs: { 100: run } }),
      { repository, identity, prNumber })
    assert.equal(result.found, false)
  }
  assert(trustedRun({ ...trusted, pull_requests: [] }, repository, prNumber, true),
    'GitHub can omit PR associations for target workflows')
  assert(trustedRun({ ...trusted, head_repository: { full_name: 'contributor/fork' } }, repository, prNumber, true),
    'Target workflows execute base code even when the candidate lives in a fork')
})

test('a different evaluation identity cannot supply a seed plan', async () => {
  const result = await findPlan(fixture({ artifacts: [{ id: 1, name: `evaluation-plan-${prNumber}-${'d'.repeat(64)}` }] }),
    { repository, identity, prNumber })
  assert.equal(result.found, false)
})

test('merged model PRs publish only successful trusted matching results', async () => {
  const result = await findResults(fixture({ artifacts: [{ id: 3, name: resultName }],
    files: [{ filename: 'public/models/new.onnx' }] }), { repository, identity, sha, before })
  assert.deepEqual(result, { found: true, bootstrap: false, artifactId: 3, artifactName: resultName, runId: 100, prNumber })
  for (const conclusion of ['failure', 'cancelled', null])
    await assert.rejects(findResults(fixture({ artifacts: [{ id: 3, name: resultName }],
      runs: { 100: { ...trusted, conclusion } }, files: [{ filename: 'public/models/new.onnx' }] }),
    { repository, identity, sha, before }), /no successful trusted evaluation/)
})

test('expired accepted results fail publication', async () => {
  await assert.rejects(findResults(fixture({ artifacts: [{ id: 3, name: resultName, expired: true }] }),
    { repository, identity, sha, before }), /expired/)
})

test('infrastructure rollout can bootstrap, while model and roster changes require verified results', async () => {
  assert.deepEqual(await findResults(fixture({ files: [{ filename: '.github/workflows/model-evaluation.yml' }] }),
    { repository, identity, sha, before }), { found: false, bootstrap: true, prNumber })
  for (const filename of ['public/models/new.onnx', 'src/roster.ts'])
    await assert.rejects(findResults(fixture({ files: [{ filename }] }), { repository, identity, sha, before }),
      /Publication is blocked/)
})

test('direct model pushes cannot bootstrap a new published league', async () => {
  await assert.rejects(findResults(fixture({ associated: [], files: [{ filename: 'public/models/new.onnx' }] }),
    { repository, identity, sha, before }), /Direct model changes/)
  assert.deepEqual(await findResults(fixture({ associated: [], files: [{ filename: 'README.md' }] }),
    { repository, identity, sha, before }), { found: false, bootstrap: true })
  await assert.rejects(findResults(fixture({ associated: [], files: [{ filename: 'public/models/new.onnx' }] }),
    { repository, identity, sha }), /Direct model changes/)
  assert.deepEqual(await findResults(fixture({ associated: [], files: [{ filename: 'README.md' }] }),
    { repository, identity, sha }), { found: false, bootstrap: true })
})

test('renames out of the model directory still count as model changes', () => {
  assert(changesModels([{ filename: 'old-model.bin', previous_filename: 'public/models/old.onnx' }]))
  assert.equal(changesModels([{ filename: 'README.md' }]), false)
})

test('model weights cannot change alongside the evaluator or workflow', async () => {
  for (const filename of ['src/game.ts', 'src/validate.ts', 'scripts/evaluation-plan.ts',
    'scripts/ci-artifacts.ts', '.nvmrc', 'package-lock.json', '.github/workflows/check.yml'])
    await assert.rejects(checkScope(fixture({ files: [
      { filename: 'public/models/new.onnx' }, { filename },
    ] }), { repository, prNumber }), /Submit model weights separately/)
  assert.deepEqual(await checkScope(fixture({ files: [{ filename: 'scripts/ci-artifacts.ts' }] }),
    { repository, prNumber }), { modelChanges: false, evaluatorChanges: true })
  assert.deepEqual(await checkScope(fixture({ files: [
    { filename: 'public/models/new.onnx' }, { filename: 'src/roster.ts' }, { filename: 'handwritten/new.py' },
  ] }), { repository, prNumber }), { modelChanges: true, evaluatorChanges: false })
})

test('deleted historical plan cannot be rerolled by a cosmetic commit or model filename rename', async () => {
  const oldHead = 'd'.repeat(40)
  const history = { workflow_runs: [{ ...trusted, head_sha: before,
    display_title: `Model PR #${prNumber} at ${oldHead}` }] }
  const get: GetJson = async path => {
    if (path.includes('/actions/artifacts?')) return { artifacts: [] }
    if (path.includes('/jobs?')) return planStages()
    if (path.includes('/actions/workflows/')) return history
    if (path.includes(`/git/trees/${sha}`)) return { tree: [{ path: 'public/models/new-name.onnx', type: 'blob', sha: 'same-weights' }] }
    if (path.includes(`/git/trees/${oldHead}`)) return { tree: [{ path: 'public/models/old-name.onnx', type: 'blob', sha: 'same-weights' }] }
    throw new Error(`Unexpected history request ${path}`)
  }
  await assert.rejects(findPlan(get, { repository, identity, prNumber, headSha: sha, baseSha: before, runId: 200 }),
    /same model bytes.*Refusing to reroll/)
})

test('changed weights can create a new plan despite previous runs on that PR', async () => {
  const oldHead = 'd'.repeat(40)
  const get: GetJson = async path => {
    if (path.includes('/actions/artifacts?')) return { artifacts: [] }
    if (path.includes('/jobs?')) return planStages()
    if (path.includes('/actions/workflows/')) return { workflow_runs: [{ ...trusted, head_sha: before,
      display_title: `Model PR #${prNumber} at ${oldHead}` }] }
    if (path.includes('/git/trees/')) return { tree: [{ path: 'public/models/model.onnx', type: 'blob',
      sha: path.includes(oldHead) ? 'old-weights' : 'new-weights' }] }
    throw new Error(`Unexpected history request ${path}`)
  }
  assert.deepEqual(await findPlan(get, { repository, identity, prNumber, headSha: sha, baseSha: before, runId: 200 }),
    { found: false, bootstrap: false, prNumber })
})

test('documentation-only base commits do not permit a reroll after plan deletion', async () => {
  const oldHead = 'd'.repeat(40), oldBase = 'e'.repeat(40)
  const get: GetJson = async path => {
    if (path.includes('/actions/artifacts?')) return { artifacts: [] }
    if (path.includes('/jobs?')) return planStages()
    if (path.includes('/actions/workflows/')) return { workflow_runs: [{ ...trusted, head_sha: oldBase,
      display_title: `Model PR #${prNumber} at ${oldHead}` }] }
    if (path.includes(`/git/trees/${before}`) || path.includes(`/git/trees/${oldBase}`))
      return { tree: [{ path: 'src/game.ts', type: 'blob', sha: 'same-evaluator' },
        { path: 'README.md', type: 'blob', sha: path.includes(oldBase) ? 'old-docs' : 'new-docs' }] }
    if (path.includes('/git/trees/')) return { tree: [{ path: 'public/models/model.onnx', type: 'blob', sha: 'same-weights' }] }
    throw new Error(`Unexpected history request ${path}`)
  }
  await assert.rejects(findPlan(get, { repository, identity, prNumber, headSha: sha, baseSha: before, runId: 200 }),
    /same model bytes.*Refusing to reroll/)
})

test('an actual trusted evaluator change permits a new evaluation plan', async () => {
  const oldHead = 'd'.repeat(40), oldBase = 'e'.repeat(40)
  const get: GetJson = async path => {
    if (path.includes('/actions/artifacts?')) return { artifacts: [] }
    if (path.includes('/jobs?')) return planStages()
    if (path.includes('/actions/workflows/')) return { workflow_runs: [{ ...trusted, head_sha: oldBase,
      display_title: `Model PR #${prNumber} at ${oldHead}` }] }
    if (path.includes('/git/trees/')) return { tree: [{ path: 'src/game.ts', type: 'blob',
      sha: path.includes(oldBase) ? 'old-evaluator' : 'new-evaluator' }] }
    throw new Error(`Unexpected history request ${path}`)
  }
  assert.deepEqual(await findPlan(get, { repository, identity, prNumber, headSha: sha, baseSha: before, runId: 200 }),
    { found: false, bootstrap: false, prNumber })
})

test('trusted published results are reused before PR lookup, even after deployment failure', async () => {
  const name = `published-league-${identity}`
  const result = await findResults(fixture({ artifacts: [{ id: 4, name, workflow_run: { id: 101 } }],
    runs: { 101: trustedPages } }), { repository, identity, sha })
  assert.deepEqual(result, { found: true, bootstrap: false, artifactId: 4, artifactName: name, runId: 101 })
  assert(trustedPagesRun({ ...trustedPages, event: 'workflow_dispatch' }))
  assert.equal(trustedPagesRun({ ...trustedPages, head_branch: 'contributor-branch' }), false)
  assert.equal(trustedPagesRun({ ...trustedPages, event: 'pull_request' }), false)
})

test('untrusted publications cannot bypass accepted model PR results', async () => {
  for (const run of [{ ...trustedPages, head_branch: 'feature' }, { ...trustedPages, event: 'pull_request' },
    { ...trustedPages, path: '.github/workflows/check.yml' }])
    await assert.rejects(findResults(fixture({ artifacts: [{ id: 4, name: `published-league-${identity}`, workflow_run: { id: 101 } }],
      runs: { 101: run }, files: [{ filename: 'public/models/new.onnx' }] }),
    { repository, identity, sha, before }), /Publication is blocked/)
})

test('trusted-main retries restore the exact persisted plan', async () => {
  const name = `main-evaluation-plan-${identity}`
  assert.deepEqual(await findMainPlan(fixture({ artifacts: [{ id: 5, name, workflow_run: { id: 101 } }],
    runs: { 101: trustedPages } }), { repository, identity, sha, runAttempt: 2 }),
  { found: true, bootstrap: false, artifactId: 5, artifactName: name, runId: 101 })
  await assert.rejects(findMainPlan(fixture(), { repository, identity, sha, runAttempt: 2 }),
    /main evaluation plan.*Refusing to reroll/)
  await assert.rejects(findMainPlan(fixture({ artifacts: [{ id: 5, name, expired: true, workflow_run: { id: 101 } }],
    runs: { 101: trustedPages } }), { repository, identity, sha }), /expired/)
})

test('deleted main plans cannot be rerolled by manual dispatch or documentation-only commits', async () => {
  const oldSha = 'e'.repeat(40)
  const get: GetJson = async path => {
    if (path.includes('/actions/artifacts?')) return { artifacts: [] }
    if (path.includes('/actions/workflows/pages.yml/runs?')) return { workflow_runs: [{ ...trustedPages, head_sha: oldSha }] }
    if (path.includes('/jobs?')) return planStages(true)
    if (path.includes('/git/trees/')) return { tree: [
      { path: 'public/models/model.onnx', type: 'blob', sha: 'same-weights' },
      { path: 'src/game.ts', type: 'blob', sha: 'same-evaluator' },
      { path: 'README.md', type: 'blob', sha: path.includes(oldSha) ? 'old-docs' : 'new-docs' },
    ] }
    throw new Error(`Unexpected main history request ${path}`)
  }
  await assert.rejects(findMainPlan(get, { repository, identity, sha, runId: 200 }),
    /earlier main run.*Refusing to reroll/)
})

test('validation failures before plan persistence permit corrected roster metadata', async () => {
  const oldHead = 'd'.repeat(40)
  const get: GetJson = async path => {
    if (path.includes('/actions/artifacts?')) return { artifacts: [] }
    if (path.includes('/actions/workflows/')) return { workflow_runs: [{ ...trusted, head_sha: before,
      display_title: `Model PR #${prNumber} at ${oldHead}` }] }
    if (path.includes('/jobs?')) return planStages(false, false)
    if (path.includes('/git/trees/')) return { tree: [{ path: 'public/models/model.onnx', type: 'blob', sha: 'same-weights' }] }
    throw new Error(`Unexpected history request ${path}`)
  }
  assert.deepEqual(await findPlan(get, { repository, identity, prNumber, headSha: sha, baseSha: before, runId: 200 }),
    { found: false, bootstrap: false, prNumber })
})

test('missing job stage metadata fails closed when a prior matching model revision exists', async () => {
  const get: GetJson = async path => {
    if (path.includes('/actions/artifacts?')) return { artifacts: [] }
    if (path.includes('/actions/workflows/')) return { workflow_runs: [{ ...trusted, head_sha: before,
      display_title: `Model PR #${prNumber} at ${sha}` }] }
    if (path.includes('/jobs?')) return { jobs: [{ steps: [] }] }
    if (path.includes('/git/trees/')) return { tree: [{ path: 'public/models/model.onnx', type: 'blob', sha: 'same-weights' }] }
    throw new Error(`Unexpected history request ${path}`)
  }
  await assert.rejects(findPlan(get, { repository, identity, prNumber, headSha: sha, baseSha: before, runId: 200 }),
    /does not expose.*Refusing to reroll/)
})

test('PR history lookup is bounded by its creation time', async () => {
  const get: GetJson = async path => {
    if (path.includes('/actions/artifacts?')) return { artifacts: [] }
    assert(path.includes('&created=%3E%3D2026-11-07T01%3A00%3A00.000Z&'))
    return { workflow_runs: [] }
  }
  assert.deepEqual(await findPlan(get, { repository, identity, prNumber, headSha: sha, baseSha: before,
    runId: 200, prCreatedAt: '2026-11-07T01:00:00Z' }), { found: false, bootstrap: false, prNumber })
})

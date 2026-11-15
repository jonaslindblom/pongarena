import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import {
  selectEvaluation, standingsComment, upsertComment, verifyProvenance,
} from '../scripts/model-evaluation-comment.mjs'

const repository = 'owner/pongarena'
const runId = 40
const pr = 15
const headSha = 'a'.repeat(40)
const baseSha = 'b'.repeat(40)
const identity = 'c'.repeat(64)
const artifact = `model-league-${pr}-${identity}`
const report = { artifact, pr, runId, headSha, identity }
const runUrl = `https://github.com/${repository}/actions/runs/${runId}`
const title = `Model PR #${pr} at ${headSha}`
const modelRun = {
  id: runId, repository: { full_name: repository }, event: 'pull_request_target',
  name: title, path: '.github/workflows/model-evaluation.yml',
  display_title: title, head_sha: headSha, status: 'completed', conclusion: 'success',
  pull_requests: [{ number: pr, head: { sha: headSha }, base: { sha: baseSha } }],
}

test('only the successful trusted model run with a matching artifact can produce a comment', async () => {
  const request = async (path: string) => {
    if (path.endsWith('/artifacts?per_page=100')) {
      return { total_count: 2, artifacts: [
        { name: 'model-league-16-' + identity },
        { name: artifact, expired: false },
      ] }
    }
    assert.equal(path, `/repos/${repository}/actions/runs/${runId}`)
    return modelRun
  }
  assert.deepEqual(await selectEvaluation(request, repository, String(runId)), report)
  assert.equal(await selectEvaluation(async () => ({ ...modelRun, conclusion: 'failure' }), repository, String(runId)), undefined)
  assert.equal(await selectEvaluation(async (path: string) => path.includes('/artifacts')
    ? { total_count: 0, artifacts: [] } : modelRun, repository, String(runId)), undefined)
  await assert.rejects(selectEvaluation(async () => ({ ...modelRun, event: 'pull_request' }), repository, String(runId)), /not a trusted/)
  await assert.rejects(selectEvaluation(async () => ({ ...modelRun, head_sha: baseSha }), repository, String(runId)), /not a trusted/)
  await assert.rejects(selectEvaluation(async () => ({ ...modelRun, pull_requests: [{ number: pr + 1 }] }), repository, String(runId)), /different PR/)
  const provenance = {
    schema: 'pongarena-pr-evaluation-v1', repository, pullRequest: pr, headSha, baseSha,
    identity, runId: String(runId),
  }
  assert.doesNotThrow(() => verifyProvenance(provenance, repository, report))
  assert.throws(() => verifyProvenance({ ...provenance, headSha: baseSha }, repository, report), /do not match/)
  const retargetedReport = await selectEvaluation(async (path: string) => path.includes('/artifacts')
    ? request(path) : { ...modelRun, pull_requests: [{ number: pr, head: { sha: headSha }, base: { sha: 'd'.repeat(40) } }] },
  repository, String(runId))
  assert.deepEqual(retargetedReport, report)
  assert(retargetedReport)
  assert.doesNotThrow(() => verifyProvenance(provenance, repository, retargetedReport))
  const forkReport = await selectEvaluation(async (path: string) => path.includes('/artifacts')
    ? request(path) : { ...modelRun, pull_requests: [] }, repository, String(runId))
  assert.deepEqual(forkReport, report)
  assert(forkReport)
  assert.doesNotThrow(() => verifyProvenance(provenance, repository, forkReport))
  assert.throws(() => verifyProvenance({ ...provenance, baseSha: 'branch-name' }, repository, forkReport), /do not match/)
})

test('untrusted model names cannot add mentions, images or extra table rows', async () => {
  const saved = JSON.parse(readFileSync(new URL('../public/league.json', import.meta.url), 'utf8'))
  saved.spec.entrants[0].name = '@someone | ![img](https://bad.invalid/image)\n<a>'
  const body = await standingsComment(saved, report, runUrl)
  assert.match(body, /&#64;&#8203;someone &#124; &#33;&#91;img&#93;/)
  assert.doesNotMatch(body, /@someone|!\[img\]|https:\/\/bad\.invalid|<a>/)
  assert.ok(body.includes(runUrl))
  assert.equal(body.split('\n').filter(line => /^\| \d+ \|/.test(line)).length, saved.spec.entrants.length)
})

test('reruns update only the workflow bot comment and an outdated run writes nothing', async () => {
  const body = '<!-- pongarena:model-evaluation -->\nStandings'
  const writes: { path: string; method: string }[] = []
  let current = headSha
  let comments = [{ id: 1, user: { login: 'participant' }, body }]
  const request = async (path: string, options?: { method: string; body: string }) => {
    if (options) {
      writes.push({ path, method: options.method })
      comments = [...comments.filter(comment => comment.id !== 2), {
        id: 2, user: { login: 'github-actions[bot]' }, body: JSON.parse(options.body).body,
      }]
      return
    }
    return path.includes('/pulls/') ? { head: { sha: current } } : comments
  }
  assert.equal(await upsertComment(request, repository, report, body), 'posted')
  assert.equal(await upsertComment(request, repository, report, body), 'unchanged')
  assert.equal(await upsertComment(request, repository, report, body + ' updated'), 'updated')
  current = baseSha
  assert.equal(await upsertComment(request, repository, report, body + ' stale'), 'outdated')
  assert.deepEqual(writes, [
    { path: `/repos/${repository}/issues/${pr}/comments`, method: 'POST' },
    { path: `/repos/${repository}/issues/comments/2`, method: 'PATCH' },
  ])
})

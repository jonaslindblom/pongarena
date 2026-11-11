import { appendFile, readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'

const marker = '<!-- pongarena:model-evaluation -->'
const sha = /^[a-f0-9]{40}$/

export async function selectEvaluation(request, repository, id) {
  const runId = Number(id)
  if (!/^[1-9]\d*$/.test(String(id)) || !Number.isSafeInteger(runId))
    throw new Error('Expected a positive model evaluation run ID.')
  const run = await request(`/repos/${repository}/actions/runs/${runId}`)
  const revision = /^Model PR #([1-9]\d*) at ([a-f0-9]{40})$/.exec(run.display_title ?? '')
  if (run.id !== runId || run.repository?.full_name !== repository || run.event !== 'pull_request_target' ||
      run.path?.split('@')[0] !== '.github/workflows/model-evaluation.yml' ||
      !revision || run.head_sha !== revision[2])
    throw new Error('The requested run is not a trusted model submission evaluation.')
  if (run.status !== 'completed' || run.conclusion !== 'success') return
  const pr = Number(revision[1])
  if (!Number.isSafeInteger(pr)) throw new Error('Invalid evaluation PR number.')
  const associated = run.pull_requests ?? []
  const revisionAssociation = associated.find(pullRequest => pullRequest.number === pr)
  if (associated.length && (!revisionAssociation || revisionAssociation.head?.sha !== revision[2]))
    throw new Error('The workflow run is associated with a different PR revision.')
  const baseSha = revisionAssociation?.base?.sha
  if (baseSha !== undefined && !sha.test(baseSha)) throw new Error('Invalid evaluation base SHA.')
  const { artifacts, total_count: total } = await request(`/repos/${repository}/actions/runs/${runId}/artifacts?per_page=100`)
  if (!Array.isArray(artifacts) || artifacts.length !== total) throw new Error('Incomplete evaluation artifact listing.')
  const pattern = new RegExp(`^model-league-${pr}-([a-f0-9]{64})$`)
  const matches = artifacts.filter(artifact => pattern.test(artifact.name))
  if (!matches.length) return // A successful check without model changes has no results.
  if (matches.length !== 1 || matches[0].expired) throw new Error('Expected exactly one available evaluation result bundle.')
  const artifact = matches[0].name
  return { artifact, pr, runId, headSha: revision[2], baseSha, identity: pattern.exec(artifact)[1] }
}

export function verifyProvenance(provenance, repository, report) {
  if (provenance.schema !== 'pongarena-pr-evaluation-v1' || provenance.repository !== repository ||
      provenance.pullRequest !== report.pr || provenance.headSha !== report.headSha ||
      !sha.test(provenance.baseSha ?? '') || (report.baseSha !== undefined && provenance.baseSha !== report.baseSha) ||
      provenance.identity !== report.identity ||
      provenance.runId !== String(report.runId))
    throw new Error('The results do not match the selected model evaluation.')
}

function modelName(name) {
  // Punctuation stays text in GitHub Markdown, and the separator breaks mentions.
  const text = Array.from(name.replace(/[\p{Cc}\p{Cf}\p{Z}]+/gu, ' ').trim())
  const shortened = text.length > 120 ? [...text.slice(0, 120), '…'] : text
  return shortened.map(char => char === '@' ? '&#64;&#8203;' :
    /[\p{L}\p{N}\p{M} ]/u.test(char) ? char : `&#${char.codePointAt(0)};`).join('')
}

export async function standingsComment(value, report, runUrl) {
  const { restoreCompetition, standings, smallestDivision } = await import('../src/competition.ts')
  const run = await restoreCompetition(value)
  if (run.status !== 'complete') throw new Error('Cannot post standings for an incomplete evaluation.')
  const rows = standings(run).map(({ rank, entrant, leaguePoints, played, wins, draws, losses }) =>
    `| ${rank} | ${modelName(entrant.name)} | ${smallestDivision(entrant)} | ${leaguePoints} / ${played} | ${wins} / ${draws} / ${losses} | ${entrant.params} |`)
  return [
    marker,
    '### Model evaluation results',
    '',
    `Evaluated commit \`${report.headSha.slice(0, 12)}\` on ${run.spec.seeds.length} paired serve sets per matchup (${run.totalMatches.toLocaleString('en-US')} games).`,
    '',
    '| Rank | Model | Size | Points | W / D / L | Parameters |',
    '| ---: | --- | --- | ---: | ---: | ---: |',
    ...rows,
    '',
    `[Trusted evaluation run and full results](${runUrl})`,
  ].join('\n')
}

export async function upsertComment(request, repository, report, body) {
  const issue = `/repos/${repository}/issues/${report.pr}`
  let existing
  for (let page = 1; ; page++) {
    const comments = await request(`${issue}/comments?per_page=100&page=${page}`)
    existing = comments.find(comment => comment.user?.login === 'github-actions[bot]' &&
      comment.body?.startsWith(marker))
    if (existing || comments.length < 100) break
  }
  // Check the current head immediately before writing. A new push may have
  // arrived while the evaluation ran or while this job waited in the queue.
  const pr = await request(`/repos/${repository}/pulls/${report.pr}`)
  if (pr.head?.sha !== report.headSha) return 'outdated'
  if (existing?.body === body) return 'unchanged'
  const path = existing ? `/repos/${repository}/issues/comments/${existing.id}` : `${issue}/comments`
  await request(path, { method: existing ? 'PATCH' : 'POST', body: JSON.stringify({ body }) })
  return existing ? 'updated' : 'posted'
}

async function main() {
  const repository = process.env.GITHUB_REPOSITORY ?? ''
  const token = process.env.GITHUB_TOKEN
  if (!/^[\w.-]+\/[\w.-]+$/.test(repository) || !token)
    throw new Error('The GitHub repository and workflow token are required.')
  const request = async (path, options = {}) => {
    const response = await fetch((process.env.GITHUB_API_URL ?? 'https://api.github.com') + path, {
      ...options,
      headers: { Authorization: `Bearer ${token}`, Accept: 'application/vnd.github+json',
        'Content-Type': 'application/json', 'X-GitHub-Api-Version': '2022-11-28' },
    })
    if (!response.ok) throw new Error(`GitHub request failed (${response.status}) for ${path}.`)
    return response.json()
  }
  if (process.argv[2] === 'select') {
    const report = await selectEvaluation(request, repository, process.env.EVALUATION_RUN_ID)
    if (!report) {
      console.log('No successful model standings to comment on.')
      return
    }
    await appendFile(process.env.GITHUB_OUTPUT, [
      `artifact=${report.artifact}`, `pr=${report.pr}`, `run=${report.runId}`,
      `report=${JSON.stringify(report)}`, '',
    ].join('\n'))
  } else if (process.argv[2] === 'post') {
    const report = JSON.parse(process.env.EVALUATION_REPORT)
    const directory = process.env.EVALUATION_ARTIFACT_DIR
    const readJson = async name => JSON.parse(await readFile(join(directory, name), 'utf8'))
    verifyProvenance(await readJson('provenance.json'), repository, report)
    const runUrl = `${process.env.GITHUB_SERVER_URL ?? 'https://github.com'}/${repository}/actions/runs/${report.runId}`
    const body = await standingsComment(await readJson('league.json'), report, runUrl)
    const result = await upsertComment(request, repository, report, body)
    console.log(`Model evaluation comment: ${result}.`)
  } else {
    throw new Error('Expected select or post.')
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  main().catch(error => { console.error(error.message); process.exitCode = 1 })

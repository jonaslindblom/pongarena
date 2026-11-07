import test from 'node:test'
import assert from 'node:assert/strict'
import { githubProfileUrl, isGitHubUsername } from '../src/submitter'

test('submitter profiles use validated GitHub usernames on the fixed GitHub origin', () => {
  for (const handle of ['jonaslindblom', 'A', 'model-author', 'a'.repeat(39)]) {
    assert.equal(isGitHubUsername(handle), true)
    assert.equal(githubProfileUrl(handle), `https://github.com/${handle}`)
  }
})

test('missing or malformed attribution never produces a profile link', () => {
  for (const handle of [undefined, null, 1, '', '-author', 'author-', 'two--hyphens',
    'a'.repeat(40), '@author', 'author/name', 'https://example.com', 'javascript:alert(1)',
    'author?token=1', 'author#fragment', 'author.name', 'author_name', 'élève', ' author']) {
    assert.equal(isGitHubUsername(handle), false)
    assert.equal(githubProfileUrl(handle), undefined)
  }
})

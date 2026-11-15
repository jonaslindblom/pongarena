/** GitHub usernames are metadata, never arbitrary link destinations. */
export function isGitHubUsername(value: unknown): value is string {
  return typeof value === 'string' && value.length <= 39 &&
    /^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$/.test(value)
}

export function githubProfileUrl(handle: unknown): string | undefined {
  return isGitHubUsername(handle) ? `https://github.com/${handle}` : undefined
}

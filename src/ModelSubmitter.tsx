import { githubProfileUrl } from './submitter'

export function ModelSubmitter({ handle }: { handle?: string }) {
  const href = githubProfileUrl(handle)
  if (!href) return null
  return (
    <p className="model-submitter">
      Submitted by{' '}
      <a href={href} target="_blank" rel="noopener noreferrer">@{handle}</a>
    </p>
  )
}

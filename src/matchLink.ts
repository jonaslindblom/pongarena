import type { Pair } from './rules'

export const DEFAULT_MODEL = 'counter'

export function readMatchLink(search: string): { mode: 'play' | 'watch'; ids: Pair<string> } {
  const params = new URLSearchParams(search)
  const model = params.get('model') || DEFAULT_MODEL
  return params.get('mode') === 'watch'
    ? { mode: 'watch', ids: [model, params.get('opponent') || 'quill'] }
    : { mode: 'play', ids: ['keyboard', model] }
}

export function writeMatchLink(url: URL, mode: 'play' | 'watch', ids: Pair<string>) {
  url.searchParams.set('model', mode === 'play' ? ids[1] : ids[0])
  if (mode === 'watch') {
    url.searchParams.set('mode', 'watch')
    url.searchParams.set('opponent', ids[1])
  } else {
    url.searchParams.delete('mode')
    url.searchParams.delete('opponent')
  }
  return url
}

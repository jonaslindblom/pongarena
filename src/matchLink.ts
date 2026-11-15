import type { Pair } from './rules'

export const DEFAULT_MODEL = 'counter'
export const DEFAULT_WATCH_IDS: Pair<string> = ['quill', DEFAULT_MODEL]

// Legacy watch URLs use model/opponent for simulation slots 0/1, not visible P1/P2.
export function readMatchLink(search: string): { mode: 'play' | 'watch'; ids: Pair<string>; demo: boolean } {
  const params = new URLSearchParams(search)
  if (!['model', 'mode', 'opponent'].some((key) => params.has(key)))
    return { mode: 'watch', ids: DEFAULT_WATCH_IDS, demo: true }
  const model = params.get('model') || DEFAULT_MODEL
  const opponent = params.get('opponent') || 'quill'
  return params.get('mode') === 'watch'
    ? { mode: 'watch', ids: [model, opponent], demo: model !== 'keyboard' && opponent !== 'keyboard' }
    : { mode: 'play', ids: ['keyboard', model], demo: false }
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

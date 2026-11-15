import type { ExplainerId } from './WeightExplainers'
import type { FindingId } from './Findings'

export type Page = 'Arena' | 'Rankings' | 'Model library' | 'How it thinks' | 'Rules' | 'Device test' | 'Findings'
export interface Route {
  page: Page
  explainer: ExplainerId
  finding?: FindingId
  catalogId?: string
}

export const siteUrl = 'https://jonaslindblom.github.io/pongarena/'
export const homeMetadata = {
  title: 'PongArena — Play Pong Against Tiny Neural Networks',
  description: 'Play Pong against programmed, prompted and trained neural networks. Explore how they work and submit your own weights to the open-source league.',
}

export const readingPages: Record<string, { route: Route; title: string; description: string }> = {
  'how-it-thinks/': {
    route: { page: 'How it thinks', explainer: 'follow' },
    title: 'How a 13-Parameter Pong Network Works — PongArena',
    description: 'Move the ball and paddle to trace Follow’s decision through one hidden neuron. Explore all 13 weights and biases in an interactive neural network explainer.',
  },
  'how-it-thinks/counter/': {
    route: { page: 'How it thinks', explainer: 'counter' },
    title: 'Switch Off a Pong Neuron: Inside Counter — PongArena',
    description: 'Explore Counter’s 97-parameter neural network. Switch off its departure neuron and see how prediction and recovery change the paddle’s decision.',
  },
  'findings/': {
    route: { page: 'Findings', explainer: 'follow', finding: 'training' },
    title: 'Training vs. Programming Tiny Pong Networks — PongArena',
    description: 'Compare programmed, trained and prompted Pong networks with 163 parameters. Explore frozen results, training methods and the limits of this experiment.',
  },
  'findings/oneshot/': {
    route: { page: 'Findings', explainer: 'follow', finding: 'oneshot' },
    title: 'One Prompt, 163 Parameters: OneShot Pong — PongArena',
    description: 'Can a single prompt produce a Pong neural network? Read OneShot’s prompt, inspect its decisions and compare its weights with trained and programmed players.',
  },
}

const pageHashes: Record<Page, string> = {
  Arena: '#play',
  Rankings: '#league',
  'Model library': '#models',
  'How it thinks': '#how-it-thinks',
  Rules: '#rules',
  Findings: '#findings',
  'Device test': '#device-test',
}

export function pagePath(page: Page, lesson: ExplainerId | FindingId = 'follow') {
  if (page === 'How it thinks') return lesson === 'counter' ? 'how-it-thinks/counter/' : 'how-it-thinks/'
  if (page === 'Findings') return lesson === 'oneshot' ? 'findings/oneshot/' : 'findings/'
  return pageHashes[page]
}

// Old shared reading links retain their query parameters while gaining a path.
export function normalizeReadingUrl(url: URL, base: URL) {
  const next = new URL(url)
  const legacy = url.hash === '#how-it-thinks/oneshot' ? 'findings/oneshot/' : url.hash.slice(1).replace(/\/$/, '') + '/'
  if (Object.hasOwn(readingPages, legacy)) {
    next.pathname = base.pathname + legacy
    next.hash = ''
  }
  const path = next.pathname.slice(base.pathname.length)
  if (Object.hasOwn(readingPages, path + '/')) next.pathname += '/'
  return next
}

export function readPage(url: URL, base: URL): Route {
  const normalized = normalizeReadingUrl(url, base)
  const path = normalized.pathname.slice(base.pathname.length)
  if (Object.hasOwn(readingPages, path)) return readingPages[path].route
  if (url.hash === '#guide') return { page: 'Rules', explainer: 'follow' }
  const catalogId = /^#models\/([A-Za-z0-9_-]+)$/.exec(url.hash)?.[1]
  if (catalogId) return { page: 'Model library', explainer: 'follow', catalogId }
  const page = (Object.keys(pageHashes) as Page[]).find(page => pageHashes[page] === url.hash) ?? 'Arena'
  return { page, explainer: 'follow' }
}

export function pageMetadata(route: Route) {
  const path = pagePath(route.page, route.finding ?? route.explainer)
  const reading = readingPages[path]
  return { ...(reading ?? homeMetadata), url: new URL(reading ? path : '', siteUrl).href }
}

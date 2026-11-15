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

export const pages: Record<string, { route: Route; title: string; description: string }> = {
  '': {
    route: { page: 'Arena', explainer: 'follow' },
    ...homeMetadata,
  },
  'models/': {
    route: { page: 'Model library', explainer: 'follow' },
    title: 'Pong Neural Network Models and Weights — PongArena',
    description: 'Explore programmed, trained and prompted Pong controllers. Inspect their architectures, download ONNX weights, or import your own neural network to play.',
  },
  'league/': {
    route: { page: 'Rankings', explainer: 'follow' },
    title: 'Neural Network Pong League — PongArena',
    description: 'Compare Pong neural networks on the same serves from both sides. Explore league standings, play against a model, or run a tournament in your browser.',
  },
  'rules/': {
    route: { page: 'Rules', explainer: 'follow' },
    title: 'Pong Rules and Controls — PongArena',
    description: 'Learn the controls for playing Pong against neural networks, watching model matches and bringing your own weights to PongArena.',
  },
  'device-test/': {
    route: { page: 'Device test', explainer: 'follow' },
    title: 'Browser Inference and Animation Test — PongArena',
    description: 'Measure neural network inference and Pong animation performance on your device with a repeatable browser workload.',
  },
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

export function pagePath(page: Page, lesson: ExplainerId | FindingId = 'follow') {
  const candidates = Object.entries(pages).filter(([, entry]) => entry.route.page === page)
  const selected = candidates.find(([, entry]) => (entry.route.finding ?? entry.route.explainer) === lesson)
  return (selected ?? candidates[0])[0]
}

export function readPage(url: URL, base: URL): Route {
  const path = url.pathname.slice(base.pathname.length).replace(/\/?$/, '/')
  const route = pages[path]?.route ?? pages[''].route
  const catalogId = route.page === 'Model library'
    ? /^#catalog-model-([A-Za-z0-9_-]+)$/.exec(url.hash)?.[1] : undefined
  return { ...route, ...(catalogId ? { catalogId } : {}) }
}

export function pageMetadata(route: Route) {
  const path = pagePath(route.page, route.finding ?? route.explainer)
  return { ...pages[path], url: new URL(path, siteUrl).href }
}

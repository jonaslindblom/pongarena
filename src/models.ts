import * as ort from 'onnxruntime-web/wasm'
import wasmUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url'
import moduleUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url'
import { validateArenaModel, validateModel } from './validate'
import { choice, type Scores } from './rules'
import { sha256 } from './competition'
import { scriptScores, type ScriptKind } from './scripted'

ort.env.wasm.numThreads = 1
ort.env.wasm.wasmPaths = {
  wasm: new URL(wasmUrl, document.baseURI).href,
  mjs: new URL(moduleUrl, document.baseURI).href,
}
export interface Controller {
  type: 'human' | 'script' | 'weights'
  id: string
  name: string
  kind: string
  handcrafted?: boolean
  description: string
  params: number
  macs: number
  sha256?: string
  inputSize?: number
  outputSize?: number
  architecture?: ReturnType<typeof validateModel>['architecture']
  report?: string
  session?: ort.InferenceSession
  script?: ScriptKind
}
export const keyboard: Controller = {
  type: 'human',
  id: 'keyboard',
  name: 'Human',
  kind: 'Human',
  description: 'Play with keyboard, touch, or pointer controls.',
  params: 0,
  macs: 0,
}
export const scripts: Controller[] = [
  {
    id: 'script-attack', name: 'Attacker', script: 'attack',
    description: 'Predicts contact and aims at the hardest reachable return for the opposing paddle. The adaptive attacker used in our Classic v4 evaluations.',
  },
  {
    id: 'script-defend', name: 'Defender', script: 'defend',
    description: 'Plans a catch and recovers toward a position that covers the next return. The adaptive defender used in our Classic v4 evaluations.',
  },
  {
    id: 'script-interceptor', name: 'Interceptor', script: 'interceptor',
    description: 'Predicts the incoming ball’s height and moves to intercept it. Recovers to cover possible returns while the ball travels away.',
  },
].map((script) => ({
  ...script, script: script.script as ScriptKind, type: 'script', kind: 'Deterministic script',
  params: 0, macs: 0, inputSize: 6, outputSize: 3,
}))
export const presets = [
  {
    id: 'pacer-v4',
    name: 'Pacer v4',
    kind: 'Fully learned · DAgger · v4',
    description: 'A plain 6→128→128→3 Tanh policy trained on gradual-speed Pong. Learns placement and recovery from an adaptive teacher; no fixed trajectory layers.',
    report: '/training/pace-study.html',
  },
  {
    id: 'apprentice-v4',
    name: 'Apprentice v4',
    kind: 'Fully learned · Early DAgger · v4',
    description: 'An earlier checkpoint of the same plain 6→128→128→3 network. A gentler opponent trained on the current v4 rules.',
    report: '/training/pace-study.html',
  },
  {
    id: 'vector-nano-v4',
    name: 'Vector Nano v4',
    kind: 'Handcrafted weights · No training',
    handcrafted: true,
    description: 'Manually designed weights, generated from fixed formulas without training. This 487-parameter dense ReLU network uses opponent-side court-end aiming. Beats Pacer and Prism Nano in fresh tests; loses to Prism Small and adaptive scripts.',
    report: '/training/vector-nano-v4.html',
  },
  {
    id: 'prism-nano-v4',
    name: 'Prism Nano v4',
    kind: 'Handcrafted weights · No training',
    handcrafted: true,
    description: 'Manually designed weights, generated from fixed formulas without training. This 503-parameter dense ReLU network uses approximate anticipation and conservative contact aiming. Beats Tracker, Anticipator, and Challenger in fresh Nano tests; loses to Pacer and adaptive scripts.',
    report: '/training/prism-nano-v4.html',
  },
  {
    id: 'prism-small-v4',
    name: 'Prism Small v4',
    kind: 'Handcrafted weights · No training',
    handcrafted: true,
    description: 'Manually designed weights, generated from fixed formulas without training. This 1,815-parameter dense ReLU network uses a cap-speed forecast and coherent contact targets. Beats Pacer and Prism Nano in fresh tests; adaptive scripts remain much stronger.',
    report: '/training/prism-small-v4.html',
  },
  {
    id: 'contender-v3',
    name: 'Contender v3',
    kind: 'Fully learned · PPO + weight search',
    description: 'Earlier v3 weights: a learned 6→64→64→3 Tanh policy, refined through PPO and weight search. Retained for comparison under v4.',
    report: '/training/fully-learned.html',
  },
  {
    id: 'selfplay-v3',
    name: 'Selfplay v3',
    kind: 'PyTorch · PPO · v3',
    description: 'Earlier v3 weights, refined through league self-play. Retained for comparison under v4.',
    report: '/training/report.html',
  },
  {
    id: 'apprentice-v3',
    name: 'Apprentice v3',
    kind: 'PyTorch · Imitation · v3',
    description: 'Earlier v3 weights, trained to imitate a predictive attacker. Retained for comparison under v4.',
    report: '/training/report.html',
  },
  {
    id: 'selfplay-v2',
    name: 'Selfplay v2',
    kind: 'PyTorch · PPO · v2',
    description: 'Earlier v2 PPO weights, retained for comparison under the current v4 rules.',
    report: '/training/v2/report.html',
  },
  {
    id: 'challenger',
    name: 'Challenger',
    kind: 'Evolved model',
    description: 'Original v1 weights, evolved over 48 generations. Plays under the current rules.',
  },
  {
    id: 'tracker',
    name: 'Tracker',
    kind: 'Baseline model',
    description: 'Follows the ball’s current height. A baseline for your first match.',
  },
  {
    id: 'anticipator',
    name: 'Anticipator',
    kind: 'Hand-edited model',
    description: 'Looks seven ticks ahead using one edited velocity weight.',
  },
]
export async function loadModel(
  bytes: Uint8Array,
  info: Omit<Controller, 'type' | 'params' | 'macs' | 'session' | 'script'>,
): Promise<Controller> {
  const stats = validateArenaModel(bytes)
  const digest = await sha256(bytes)
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
  const controller: Controller = { ...info, ...stats, type: 'weights', sha256: digest, session }
  try {
    await infer(controller, new Float32Array(stats.inputSize))
  } catch (error) {
    await session.release()
    throw error
  }
  return controller
}
export async function infer(model: Controller, input: Float32Array): Promise<Scores> {
  if (input.length !== model.inputSize) throw new Error('This model belongs to a different game contract.')
  if (model.type === 'script') return scriptScores(model.script!, input)
  const result = await model.session!.run({ observation: new ort.Tensor('float32', input, [1, input.length]) })
  const output = result.action_scores
  const scores = Array.from(output.data as Float32Array) as Scores
  choice(scores)
  return scores
}
export async function loadPresets() {
  const controllers: Controller[] = [...scripts]
  for (const p of presets) {
    const response = await fetch('/models/' + p.id + '.onnx')
    if (!response.ok) throw new Error('Could not load ' + p.name + '.')
    controllers.push(await loadModel(new Uint8Array(await response.arrayBuffer()), p))
  }
  return controllers
}

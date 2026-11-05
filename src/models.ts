import * as ort from 'onnxruntime-web/wasm'
import wasmUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url'
import moduleUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url'
import { validateArenaModel, validateModel } from './validate'
import { choice, type Scores } from './rules'
import { sha256 } from './competition'

ort.env.wasm.numThreads = 1
ort.env.wasm.wasmPaths = {
  wasm: new URL(wasmUrl, document.baseURI).href,
  mjs: new URL(moduleUrl, document.baseURI).href,
}
export interface Controller {
  id: string
  name: string
  kind: string
  description: string
  params: number
  macs: number
  sha256?: string
  inputSize?: number
  outputSize?: number
  architecture?: ReturnType<typeof validateModel>['architecture']
  report?: string
  session?: ort.InferenceSession
}
export const keyboard: Controller = {
  id: 'keyboard',
  name: 'Human',
  kind: 'Human',
  description: 'Play with keyboard, touch, or pointer controls.',
  params: 0,
  macs: 0,
}
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
    id: 'contender-v3',
    name: 'Contender v3',
    kind: 'Fully learned · PPO + weight search',
    description: 'Earlier v3 weights: a learned 6→64→64→3 Tanh policy, refined through PPO and weight search. Retained for comparison under v4.',
    report: '/training/fully-learned.html',
  },
  {
    id: 'strategist-v3',
    name: 'Strategist v3',
    kind: 'Hybrid · Fixed geometry + DAgger',
    description: 'Earlier v3 hybrid: fixed trajectory features → trained target head → paddle feedback. Its fixed geometry follows the old rules.',
    report: '/training/improvement.html',
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
  info: Omit<Controller, 'params' | 'macs' | 'session'>,
): Promise<Controller> {
  const stats = validateArenaModel(bytes)
  const digest = await sha256(bytes)
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
  const controller = { ...info, ...stats, sha256: digest, session }
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
  const result = await model.session!.run({ observation: new ort.Tensor('float32', input, [1, input.length]) })
  const output = result.action_scores
  const scores = Array.from(output.data as Float32Array) as Scores
  choice(scores)
  return scores
}
export async function loadPresets() {
  const controllers: Controller[] = []
  for (const p of presets) {
    const response = await fetch('/models/' + p.id + '.onnx')
    if (!response.ok) throw new Error('Could not load ' + p.name + '.')
    controllers.push(await loadModel(new Uint8Array(await response.arrayBuffer()), p))
  }
  return controllers
}

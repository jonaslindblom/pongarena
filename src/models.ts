import * as ort from 'onnxruntime-web/wasm'
import wasmUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url'
import moduleUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url'
import { validateModel } from './validate'
import { choice, type Scores } from './game'

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
  architecture?: ReturnType<typeof validateModel>['architecture']
  report?: string
  session?: ort.InferenceSession
}
export const keyboard: Controller = {
  id: 'keyboard',
  name: 'Keyboard',
  kind: 'Human',
  description: 'Take control of a paddle.',
  params: 0,
  macs: 0,
}
export const presets = [
  {
    id: 'strategist-v3',
    name: 'Strategist',
    kind: 'PyTorch · DAgger · v3',
    description: 'Fixed trajectory features → trained 8→128→128→64→1 Tanh target head → paddle feedback. Six inputs, three action scores.',
    report: '/training/improvement.html',
  },
  {
    id: 'selfplay-v3',
    name: 'Selfplay',
    kind: 'PyTorch · PPO · v3',
    description: 'Refined through league self-play after a predictive-controller warm start.',
    report: '/training/report.html',
  },
  {
    id: 'apprentice-v3',
    name: 'Apprentice',
    kind: 'PyTorch · Imitation · v3',
    description: 'Learns a predictive attacker’s decisions from states it visits during play.',
    report: '/training/report.html',
  },
  {
    id: 'selfplay-v2',
    name: 'Selfplay v2',
    kind: 'PyTorch · PPO · v2',
    description: 'Previous PPO weights, retained for comparison under the current v3 rules.',
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
  const stats = validateModel(bytes)
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
  const controller = { ...info, ...stats, session }
  try {
    await infer(controller, new Float32Array(6))
  } catch (error) {
    await session.release()
    throw error
  }
  return controller
}
export async function infer(model: Controller, input: Float32Array): Promise<Scores> {
  const result = await model.session!.run({ observation: new ort.Tensor('float32', input, [1, 6]) })
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

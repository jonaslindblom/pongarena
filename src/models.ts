import * as ort from 'onnxruntime-web/wasm'
import wasmUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url'
import moduleUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url'
import { validateArenaModel, validateModel } from './validate'
import { choice, type Scores } from './rules'
import { sha256 } from './competition'
import { scriptScores, type ScriptKind } from './scripted'
import scriptedSource from './scripted.ts?raw'

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
  modelBytes?: Uint8Array
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
    id: 'robust-strong-v4',
    name: 'Robust Strong v4',
    kind: 'Fully learned · PPO · v4',
    description: 'A plain 6→128→128→3 Tanh policy with 17,795 parameters. The robustness study’s selected full-size player beats the previous winner on fresh confirmation seeds; adaptive Attacker and Defender remain stronger.',
  },
  {
    id: 'plain-nano-v4',
    name: 'Plain Nano v4',
    kind: 'Learned weights · Distillation + CMA refinement',
    description: 'A trained 6→16→12→8→3 ReLU policy with 447 parameters and 408 MACs. Beats unchanged Pacer in the retained confirmation study; Defender remains a major weakness. Uses only dense layers with distinct weights and biases.',
  },
  {
    id: 'quill-plus-v4',
    name: 'Quill Plus v4',
    kind: 'Handcrafted weights · No training',
    handcrafted: true,
    description: 'Four handwritten refinements to Quill, still 163 parameters and 141 MACs. Better long-flight prediction and speed-dependent recovery improve the ten-opponent held-out average. Weaker against Plain Nano and Vector Nano; no clear head-to-head advantage over original Quill.',
    report: '/training/quill-plus-v4.html',
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
    id: 'prism-small-v4',
    name: 'Prism Small v4',
    kind: 'Handcrafted weights · No training',
    handcrafted: true,
    description: 'Manually designed weights, generated from fixed formulas without training. This 1,815-parameter dense ReLU network uses a cap-speed forecast and coherent contact targets. Beats Pacer and Prism Nano in fresh tests; adaptive scripts remain much stronger.',
    report: '/training/prism-small-v4.html',
  },
  {
    id: 'tracker',
    name: 'Tracker',
    kind: 'Baseline model',
    description: 'Follows the ball’s current height. A baseline for your first match.',
  },
]
export async function loadModel(
  bytes: Uint8Array,
  info: Omit<Controller, 'type' | 'params' | 'macs' | 'session' | 'script'>,
): Promise<Controller> {
  const stats = validateArenaModel(bytes)
  const digest = await sha256(bytes)
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
  const controller: Controller = { ...info, ...stats, type: 'weights', sha256: digest, session, modelBytes: bytes.slice() }
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
  // Hash the policy and its complete implementation, including planning helpers.
  const controllers: Controller[] = await Promise.all(scripts.map(async (script) => ({
    ...script,
    sha256: await sha256(new TextEncoder().encode(JSON.stringify({ policy: script.script, source: scriptedSource }))),
  })))
  for (const p of presets) {
    const response = await fetch('/models/' + p.id + '.onnx')
    if (!response.ok) throw new Error('Could not load ' + p.name + '.')
    controllers.push(await loadModel(new Uint8Array(await response.arrayBuffer()), p))
  }
  return controllers
}

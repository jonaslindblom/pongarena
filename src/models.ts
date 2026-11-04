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
    id: 'challenger',
    name: 'Challenger',
    kind: 'Evolved model',
    description: 'Trained over 48 generations against a pool of opponents.',
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

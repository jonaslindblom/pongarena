import * as ort from 'onnxruntime-web/wasm'
import wasmUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url'
import moduleUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url'
import { validateArenaModel, type ModelStats } from './validate'
import { choice, type Scores } from './rules'
import { sha256 } from './competition'
import { assetPath } from './assets'
import { presets } from './roster'
export { presets } from './roster'

ort.env.wasm.numThreads = 1
ort.env.wasm.wasmPaths = {
  wasm: new URL(wasmUrl, document.baseURI).href,
  mjs: new URL(moduleUrl, document.baseURI).href,
}
export interface Controller {
  type: 'human' | 'weights'
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
  architecture?: ModelStats['architecture']
  modelBytes?: Uint8Array
  session?: ort.InferenceSession
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
export async function loadModel(
  bytes: Uint8Array,
  info: Omit<Controller, 'type' | 'params' | 'macs' | 'session'>,
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
  const result = await model.session!.run({ observation: new ort.Tensor('float32', input, [1, input.length]) })
  const output = result.action_scores
  const scores = Array.from(output.data as Float32Array) as Scores
  choice(scores)
  return scores
}
export async function loadPresets() {
  const controllers: Controller[] = []
  for (const p of presets) {
    const response = await fetch(assetPath('models/' + p.id + '.onnx'))
    if (!response.ok) throw new Error('Could not load ' + p.name + '.')
    controllers.push(await loadModel(new Uint8Array(await response.arrayBuffer()), p))
  }
  return controllers
}

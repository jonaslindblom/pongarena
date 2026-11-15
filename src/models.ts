import * as ort from 'onnxruntime-web/wasm'
import wasmUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url'
import moduleUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url'
import { validateArenaModel, type ModelStats } from './validate'
import { choice, type Scores } from './rules'
import { sha256 } from './competition'
import { assetPath } from './assets'
import { modelIdentity } from './modelIdentity'
import { methodLabel, readModelMetadata, type ModelMethod, type ModelOrigin } from './modelMetadata'
import { isGitHubUsername } from './submitter'
import type { CatalogEntry } from './modelCatalog'

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
  submittedBy?: string
  origin?: ModelOrigin
  method?: ModelMethod
  sourceUrl?: string
  description: string
  params: number
  macs: number
  sha256?: string
  fileSha256?: string
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
  const digest = await modelIdentity(bytes)
  const fileSha256 = await sha256(bytes)
  if ((info.sha256 && info.sha256 !== digest) || (info.fileSha256 && info.fileSha256 !== fileSha256))
    throw new Error(info.name + ' does not match the published model catalog.')
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
  const controller: Controller = { ...info, ...stats, type: 'weights', sha256: digest, fileSha256, session, modelBytes: bytes.slice() }
  try {
    await infer(controller, new Float32Array(stats.inputSize))
  } catch (error) {
    await session.release()
    throw error
  }
  return controller
}
export async function loadImportedModel(bytes: Uint8Array, id: string, fallbackName: string) {
  const metadata = readModelMetadata(bytes, fallbackName)
  return loadModel(bytes, {
    ...metadata, id, origin: 'local', kind: methodLabel(metadata.method),
    handcrafted: metadata.method === 'programmed',
    description: metadata.description || 'Imported into this browser.',
  })
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
  const response = await fetch(assetPath('models.json'))
  if (!response.ok) throw new Error('Could not load the model catalog.')
  const catalog = await response.json() as unknown
  const ids = new Set<string>()
  if (!Array.isArray(catalog) || !catalog.length) throw new Error('The model catalog is empty or invalid.')
  for (const entry of catalog) {
    if (!entry || typeof entry !== 'object' || typeof entry.id !== 'string' ||
      !/^[A-Za-z0-9_-]+$/.test(entry.id) || entry.id === 'keyboard' || entry.id.startsWith('local-') || ids.has(entry.id) ||
      typeof entry.name !== 'string' || !entry.name.trim() || typeof entry.description !== 'string' ||
      typeof entry.kind !== 'string' || !['core', 'community'].includes(entry.origin) ||
      !['programmed', 'trained', 'prompted', 'mixed'].includes(entry.method) ||
      !/^[a-f0-9]{64}$/.test(entry.sha256) || !/^[a-f0-9]{64}$/.test(entry.fileSha256) ||
      (entry.submittedBy !== undefined && !isGitHubUsername(entry.submittedBy)) ||
      (entry.sourceUrl !== undefined && (typeof entry.sourceUrl !== 'string' || !entry.sourceUrl.startsWith('https://'))))
      throw new Error('The model catalog contains an invalid entry.')
    ids.add(entry.id)
  }
  const controllers: Controller[] = []
  try {
    for (const p of catalog as CatalogEntry[]) {
      const modelResponse = await fetch(assetPath('models/' + p.id + '.onnx'))
      if (!modelResponse.ok) throw new Error('Could not load ' + p.name + '.')
      controllers.push(await loadModel(new Uint8Array(await modelResponse.arrayBuffer()), p))
    }
  } catch (error) {
    await Promise.all(controllers.map(controller => controller.session?.release()))
    throw error
  }
  return controllers
}

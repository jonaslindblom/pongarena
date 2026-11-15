import pb from 'onnx-proto'
import { MAX_BYTES } from './validate'

export const MODEL_METHODS = ['programmed', 'trained', 'prompted', 'mixed'] as const
export type ModelMethod = typeof MODEL_METHODS[number]
export type ModelOrigin = 'core' | 'community' | 'local'

export interface ModelMetadata {
  name: string
  description: string
  method?: ModelMethod
  sourceUrl?: string
}

export function methodLabel(method?: ModelMethod) {
  switch (method) {
    case 'programmed': return 'Programmed weights'
    case 'trained': return 'Trained weights'
    case 'prompted': return 'Prompted weights'
    case 'mixed': return 'Mixed methods'
    default: return 'Imported model'
  }
}

/** Contributor-authored text only. Origin and submitter credit come from trusted project data. */
export function readModelMetadata(bytes: Uint8Array, fallbackName: string, required = false): ModelMetadata {
  if (!bytes.length || bytes.length > MAX_BYTES) throw new Error('Choose an ONNX file up to 256 KiB.')
  const model = pb.onnx.ModelProto.decode(bytes)
  const props = new Map<string, string>()
  if (model.metadataProps.length > 64) throw new Error('Use at most 64 ONNX metadata entries.')
  for (const entry of model.metadataProps) {
    const key = entry.key ?? '', value = entry.value ?? ''
    if (!key.trim() || key.length > 128 || /[\u0000-\u001f\u007f]/.test(key) || value.length > 8192)
      throw new Error('ONNX metadata keys must be nonempty and at most 128 characters; values at most 8,192.')
    if (props.has(key)) throw new Error(`Duplicate ONNX metadata key: ${key}.`)
    props.set(key, value)
  }
  const name = (props.get('pongarena.name') ?? fallbackName).trim()
  if (!name || name.length > 80 || /[\u0000-\u001f\u007f]/.test(name))
    throw new Error('The model name must be one line of 1–80 characters.')
  const description = model.docString.trim()
  if (description.length > 4000 || /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(description))
    throw new Error('The ONNX doc_string must be plain text of at most 4,000 characters.')
  if (required && !description)
    throw new Error('Describe how the weights were created in the ONNX model doc_string.')
  const method = props.get('pongarena.method')
  if ((required || method !== undefined) && !MODEL_METHODS.includes(method as ModelMethod))
    throw new Error('Set pongarena.method to programmed, trained, prompted, or mixed.')
  let sourceUrl: string | undefined
  if (props.has('pongarena.source')) {
    const source = props.get('pongarena.source')!.trim()
    let url: URL
    try { url = new URL(source) }
    catch { throw new Error('pongarena.source must be an HTTPS URL.') }
    if (source.length > 2048 || /[\u0000-\u001f\u007f]/.test(source) ||
      url.protocol !== 'https:' || url.username || url.password)
      throw new Error('pongarena.source must be an HTTPS URL of at most 2,048 characters, without credentials.')
    sourceUrl = url.href
  }
  return { name, description, ...(method === undefined ? {} : { method: method as ModelMethod }),
    ...(sourceUrl === undefined ? {} : { sourceUrl }) }
}

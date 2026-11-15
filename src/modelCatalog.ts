import type { ModelMetadata, ModelOrigin } from './modelMetadata'

/** Generated from ONNX files; submission credit comes from trusted PR history. */
export interface CatalogEntry extends ModelMetadata {
  id: string
  kind: string
  handcrafted?: boolean
  origin: Exclude<ModelOrigin, 'local'>
  submittedBy?: string
  sha256: string
  fileSha256: string
}

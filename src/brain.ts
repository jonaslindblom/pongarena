import pb from 'onnx-proto'
import { validateArenaModel } from './validate'

export interface BrainLayer {
  name: string
  width: number
  kind: 'input' | 'hidden' | 'output'
  activation?: string
}

// Expose post-activation tensors on a copy; the match keeps its original graph and session.
export function createBrainGraph(bytes: Uint8Array) {
  const stats = validateArenaModel(bytes)
  const model = pb.onnx.ModelProto.decode(bytes)
  const graph = model.graph!
  const layers: BrainLayer[] = [{ name: 'observation', width: stats.inputSize, kind: 'input' }]
  for (const operation of stats.architecture.operations) {
    if (operation.op !== 'Relu' && operation.op !== 'Tanh') continue
    const width = operation.shape[1]
    layers.push({ name: operation.output, width, kind: 'hidden', activation: operation.op })
    graph.output!.push({ name: operation.output, type: { tensorType: {
      elemType: 1, shape: { dim: [{ dimValue: 1 }, { dimValue: width }] },
    } } })
  }
  layers.push({ name: 'action_scores', width: stats.outputSize, kind: 'output' })
  return { bytes: pb.onnx.ModelProto.encode(model).finish(), layers }
}

// Normalize each layer independently so small, real activations remain visible.
export function activationLevels(values: readonly number[]) {
  const peak = Math.max(0, ...values.map(Math.abs))
  return { peak, intensities: values.map((value) => peak ? Math.sqrt(Math.abs(value) / peak) : 0) }
}

import pb from 'onnx-proto'
import type { onnx } from 'onnx-proto'
import { sha256 } from './competition'
import { validateArenaModel } from './validate'

function values(tensor: onnx.ITensorProto) {
  if (tensor.rawData?.length) {
    const view = new DataView(tensor.rawData.buffer, tensor.rawData.byteOffset, tensor.rawData.byteLength)
    return Array.from({ length: tensor.rawData.length / 4 }, (_, index) => view.getFloat32(index * 4, true))
  }
  return tensor.floatData!
}

/** Hash the validated dense computation, independently of its documentation and protobuf encoding. */
export async function modelIdentity(bytes: Uint8Array) {
  const stats = validateArenaModel(bytes)
  const graph = pb.onnx.ModelProto.decode(bytes).graph!
  const tensors = new Map(graph.initializer!.map(tensor => [tensor.name, tensor]))
  const descriptor = new TextEncoder().encode(JSON.stringify({
    schema: 'pongarena-dense-inference-v1', opset: 18,
    widths: stats.architecture.widths, operations: graph.node!.map(node => node.opType),
  }))
  // Fixed little-endian float32 encoding preserves signed zero and makes rawData
  // and floatData equivalent. The separator cannot be part of the JSON header.
  const canonical = new Uint8Array(descriptor.length + 1 + stats.params * 4)
  canonical.set(descriptor)
  const view = new DataView(canonical.buffer)
  let offset = descriptor.length + 1
  for (const [index, node] of graph.node!.filter(node => node.opType === 'Gemm').entries()) {
    const weight = tensors.get(node.input![1])!, bias = tensors.get(node.input![2])!
    const matrix = values(weight), vector = values(bias)
    const transposed = Number(node.attribute?.find(attribute => attribute.name === 'transB')?.i ?? 0) === 1
    const inputs = stats.architecture.widths[index], outputs = stats.architecture.widths[index + 1]
    // Store each effective output row in the same order regardless of transB.
    for (let output = 0; output < outputs; output++) {
      for (let input = 0; input < inputs; input++) {
        view.setFloat32(offset, matrix[transposed ? output * inputs + input : input * outputs + output], true)
        offset += 4
      }
    }
    for (const value of vector) {
      view.setFloat32(offset, value, true)
      offset += 4
    }
  }
  return sha256(canonical)
}

import pb from 'onnx-proto'
import type { onnx } from 'onnx-proto'

export const MAX_BYTES = 256 * 1024
export const MAX_MACS = 65536
export const MAX_DENSE_HIDDEN_LAYERS = 4

export class DenseModelError extends Error {
  constructor(reason: string) {
    super('Use a plain dense network with fixed weights and biases: ' + reason)
    this.name = 'DenseModelError'
  }
}

function check(ok: unknown, message: string): asserts ok {
  if (!ok) throw new DenseModelError(message)
}

function checkPort(list: onnx.IValueInfoProto[], name: string, width: number) {
  const port = list[0], type = port?.type?.tensorType, dims = type?.shape?.dim
  check(list.length === 1 && port.name === name && type?.elemType === 1 &&
    dims?.length === 2 && dims.every((dim) => !dim.dimParam) &&
    Number(dims[0].dimValue) === 1 && Number(dims[1].dimValue) === width,
    `expected ${name} as float32[1, ${width}].`)
}

export function validateArenaModel(bytes: Uint8Array) {
  check(bytes.length > 0 && bytes.length <= MAX_BYTES, 'choose an ONNX file up to 256 KiB.')
  let model: onnx.ModelProto
  try { model = pb.onnx.ModelProto.decode(bytes) }
  catch { throw new DenseModelError('this file is not a readable ONNX model.') }
  const graph = model.graph
  check(graph && model.opsetImport.length === 1 && !model.opsetImport[0].domain &&
    Number(model.opsetImport[0].version) === 18, 'use standard ONNX opset 18.')
  check(!model.functions.length && !model.trainingInfo.length && !graph.sparseInitializer?.length,
    'use a self-contained inference graph.')
  checkPort(graph.input ?? [], 'observation', 6)
  checkPort(graph.output ?? [], 'action_scores', 3)

  const initializers = new Map<string, onnx.ITensorProto>()
  let params = 0, macs = 0
  for (const tensor of graph.initializer ?? []) {
    const dims = (tensor.dims ?? []).map(Number), count = dims.reduce((a, b) => a * b, 1)
    check(tensor.name && tensor.name !== 'observation' && tensor.name !== 'action_scores' &&
      !initializers.has(tensor.name) && tensor.dataType === 1 && !tensor.dataLocation &&
      !tensor.externalData?.length && dims.length >= 1 && dims.length <= 2 &&
      dims.every((dim) => Number.isSafeInteger(dim) && dim > 0) && count <= 65536,
      'weights must be uniquely named, embedded float32 vectors or matrices of bounded size.')
    const raw = tensor.rawData, floats = tensor.floatData
    check((raw?.length === count * 4 && !floats?.length) ||
      (floats?.length === count && !raw?.length), 'store every parameter exactly once as float32 values.')
    check(!tensor.doubleData?.length && !tensor.int32Data?.length && !tensor.int64Data?.length &&
      !tensor.uint64Data?.length && !tensor.stringData?.length && !tensor.segment,
      'parameter tensors may contain only their embedded float32 values.')
    if (raw?.length) {
      const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength)
      for (let i = 0; i < count; i++)
        check(Number.isFinite(view.getFloat32(i * 4, true)), 'every weight and bias must be finite.')
    } else check(floats?.every(Number.isFinite), 'every weight and bias must be finite.')
    initializers.set(tensor.name, tensor)
    params += count
  }

  const nodes = graph.node ?? []
  check(nodes.length % 2 === 1 && nodes.length <= 2 * MAX_DENSE_HIDDEN_LAYERS + 1,
    `use zero to ${MAX_DENSE_HIDDEN_LAYERS} hidden layers, each followed by the same ReLU or Tanh activation.`)
  const used = new Set<string>(), outputs = new Set(['observation', ...initializers.keys()])
  const widths = [6], operations: { op: string; inputs: string[]; output: string; shape: number[] }[] = []
  let previous = 'observation', width = 6, activation: string | undefined
  for (const [index, node] of nodes.entries()) {
    const inputs = node.input ?? [], output = node.output?.[0]
    check(!node.domain && output && node.output?.length === 1 && !outputs.has(output),
      'each standard operator needs one unique output.')
    check(inputs[0] === previous, 'layers must form one sequential chain from observation to action_scores.')
    if (index % 2 === 1) {
      check((node.opType === 'Relu' || node.opType === 'Tanh') && inputs.length === 1 &&
        !node.attribute?.length, 'hidden layers require an unmodified ReLU or Tanh activation.')
      activation ??= node.opType
      check(node.opType === activation, 'use the same activation throughout the hidden layers.')
    } else {
      check(node.opType === 'Gemm' && inputs.length === 3,
        'each dense layer must be a Gemm of its input, an embedded weight matrix, and an embedded bias vector.')
      const attributes = new Map<string, onnx.IAttributeProto>()
      for (const attr of node.attribute ?? []) {
        const name = attr.name ?? ''
        check(!attributes.has(name), 'Gemm attributes must not repeat.')
        check(((name === 'transA' || name === 'transB') && attr.type === 2 &&
          (name === 'transA' ? Number(attr.i) === 0 : [0, 1].includes(Number(attr.i)))) ||
          ((name === 'alpha' || name === 'beta') && attr.type === 1 && attr.f === 1),
          'Gemm permits only transB=0/1, transA=0, and unit alpha/beta scaling.')
        check(!attr.g && !attr.graphs?.length && !attr.t && !attr.tensors?.length &&
          !attr.floats?.length && !attr.ints?.length && !attr.s?.length && !attr.strings?.length,
          'Gemm attributes cannot contain additional tensors or subgraphs.')
        attributes.set(name, attr)
      }
      const weight = initializers.get(inputs[1]), bias = initializers.get(inputs[2])
      check(weight && bias && !used.has(inputs[1]) && !used.has(inputs[2]) && inputs[1] !== inputs[2],
        'each layer must own distinct embedded weights and biases, used once.')
      const dims = (weight.dims ?? []).map(Number), transposed = Number(attributes.get('transB')?.i ?? 0) === 1
      const inputWidth = dims[transposed ? 1 : 0], outputWidth = dims[transposed ? 0 : 1]
      check(dims.length === 2 && inputWidth === width && bias.dims?.length === 1 &&
        Number(bias.dims[0]) === outputWidth,
        'weights must be dense matrices and biases must be one vector matching the layer width.')
      macs += inputWidth * outputWidth
      check(macs <= MAX_MACS, 'the graph exceeds the per-decision compute budget.')
      used.add(inputs[1])
      used.add(inputs[2])
      width = outputWidth
      widths.push(width)
    }
    check(index === nodes.length - 1 || output !== 'action_scores',
      'only the final linear layer may produce action_scores.')
    outputs.add(output)
    operations.push({ op: node.opType!, inputs, output, shape: [1, width] })
    previous = output
  }
  check(previous === 'action_scores' && width === 3, 'the final layer must directly produce three linear action scores.')
  check(used.size === initializers.size, 'unused constants and parameter tensors are not permitted.')
  return { inputSize: 6, outputSize: 3, params, macs, nodes: nodes.length, architecture: { widths, operations } }
}

export type ModelStats = ReturnType<typeof validateArenaModel>

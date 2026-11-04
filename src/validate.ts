import pb from 'onnx-proto'
import type { onnx } from 'onnx-proto'

export const MAX_BYTES = 256 * 1024
export const MAX_MACS = 65536
type Shape = number[]
const product = (dims: Shape) => dims.reduce((a, b) => a * b, 1)
function check(ok: unknown, message: string): asserts ok {
  if (!ok) throw new Error(message)
}
function broadcast(a: Shape, b: Shape): Shape {
  const rank = Math.max(a.length, b.length)
  return Array.from({ length: rank }, (_, i) => {
    const x = a[i - rank + a.length] ?? 1,
      y = b[i - rank + b.length] ?? 1
    check(x === y || x === 1 || y === 1, 'Incompatible tensor shapes.')
    return Math.max(x, y)
  })
}
function checkPort(list: onnx.IValueInfoProto[], name: string, size: number) {
  const p = list[0],
    t = p?.type?.tensorType
  const dims = t?.shape?.dim?.map((d) => Number(d.dimValue))
  check(
    list.length === 1 && p.name === name && t?.elemType === 1 && dims?.join() === '1,' + size,
    'Expected ' + name + ' as float32[1, ' + size + '].',
  )
}

export function validateModel(bytes: Uint8Array) {
  check(bytes.length > 0 && bytes.length <= MAX_BYTES, 'Choose an ONNX file up to 256 KiB.')
  let model: onnx.ModelProto
  try {
    model = pb.onnx.ModelProto.decode(bytes)
  } catch {
    throw new Error('This file is not a readable ONNX model.')
  }
  const g = model.graph
  check(
    g &&
      model.opsetImport.length === 1 &&
      !model.opsetImport[0].domain &&
      Number(model.opsetImport[0].version) === 18,
    'Use standard ONNX opset 18.',
  )
  check(
    !model.functions.length && !model.trainingInfo.length && !g.sparseInitializer?.length,
    'Use a self-contained inference graph.',
  )
  checkPort(g.input ?? [], 'observation', 6)
  checkPort(g.output ?? [], 'action_scores', 3)
  const shapes = new Map<string, Shape>([['observation', [1, 6]]])
  let params = 0,
    macs = 0,
    elements = 0
  for (const p of g.initializer ?? []) {
    const dims = (p.dims ?? []).map(Number),
      n = product(dims)
    check(
      p.name &&
        !shapes.has(p.name) &&
        p.dataType === 1 &&
        !p.dataLocation &&
        !p.externalData?.length &&
        dims.length <= 8 &&
        dims.every((d) => Number.isSafeInteger(d) && d > 0) &&
        n <= 65536,
      'Weights must be embedded float32 tensors of bounded size.',
    )
    check(
      p.rawData?.length === n * 4 || p.floatData?.length === n,
      'A weight tensor has missing data.',
    )
    shapes.set(p.name, dims)
    params += n
  }
  const nodes = g.node ?? []
  const operations: { op: string; inputs: string[]; output: string; shape: Shape }[] = []
  const widths = [6]
  let sequential = true
  let previous = 'observation'
  const constants = new Set((g.initializer ?? []).map((p) => p.name))
  check(nodes.length <= 256, 'Limit the graph to 256 operators.')
  for (const node of nodes) {
    const op = node.opType
    check(
      !node.domain && op && ['Gemm', 'MatMul', 'Add', 'Relu', 'Tanh'].includes(op),
      'Unsupported operator: ' + op,
    )
    const attrs = new Map((node.attribute ?? []).map((a) => [a.name, a]))
    for (const [name, attr] of attrs) {
      check(
        op === 'Gemm' &&
          ['transA', 'transB', 'alpha', 'beta'].includes(name ?? '') &&
          (attr.type === 1 || attr.type === 2),
        'Unsupported operator attribute.',
      )
    }
    const inputs = (node.input ?? []).map((name) => {
      const shape = shapes.get(name)
      check(shape, 'Each node must use an earlier result or embedded weights.')
      return shape
    })
    const [a, b, c] = inputs
    let out: Shape
    if (op === 'Gemm') {
      check(
        (inputs.length === 2 || inputs.length === 3) && a.length === 2 && b.length === 2,
        'Gemm requires matrices.',
      )
      const [m, k] = Number(attrs.get('transA')?.i) ? [...a].reverse() : a
      const [k2, n] = Number(attrs.get('transB')?.i) ? [...b].reverse() : b
      out = [m, n]
      check(
        k === k2 && (!c || broadcast(out, c).join() === out.join()),
        'Incompatible Gemm dimensions.',
      )
      macs += m * k * n
    } else if (op === 'MatMul') {
      check(inputs.length === 2 && a.length && b.length, 'MatMul requires two inputs.')
      const aa = a.length === 1 ? [1, ...a] : a,
        bb = b.length === 1 ? [...b, 1] : b
      const batch = broadcast(aa.slice(0, -2), bb.slice(0, -2))
      const m = aa.at(-2)!,
        k = aa.at(-1)!,
        n = bb.at(-1)!
      check(k === bb.at(-2), 'Incompatible MatMul dimensions.')
      out = [...batch, ...(a.length === 1 ? [] : [m]), ...(b.length === 1 ? [] : [n])]
      macs += product(batch) * m * k * n
    } else if (op === 'Add') {
      check(inputs.length === 2, 'Add requires two inputs.')
      out = broadcast(a, b)
    } else {
      check(inputs.length === 1, op + ' requires one input.')
      out = a
    }
    elements += product(out)
    check(
      macs <= MAX_MACS && elements <= 262144,
      'The graph exceeds the per-decision compute budget.',
    )
    const name = node.output?.[0]
    check(
      name && node.output?.length === 1 && !shapes.has(name),
      'Each node needs one unique output.',
    )
    shapes.set(name, out)
    const dynamicInputs = (node.input ?? []).filter((input) => !constants.has(input))
    if (
      dynamicInputs.length !== 1 ||
      dynamicInputs[0] !== previous ||
      out.length !== 2 ||
      out[0] !== 1
    )
      sequential = false
    if (op === 'Gemm' || op === 'MatMul') widths.push(out.at(-1)!)
    else if (out.at(-1) !== widths.at(-1)) sequential = false
    operations.push({ op, inputs: node.input ?? [], output: name, shape: out })
    previous = name
  }
  check(
    shapes.get('action_scores')?.join() === '1,3',
    'The graph must produce action_scores[1, 3].',
  )
  return {
    params,
    macs,
    nodes: nodes.length,
    architecture: {
      widths: sequential && previous === 'action_scores' ? widths : null,
      operations,
    },
  }
}

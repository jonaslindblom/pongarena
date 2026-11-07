import pb from 'onnx-proto'

export function tensor(name: string, dims: number[]) {
  return { name, dims, dataType: 1, rawData: new Uint8Array(dims.reduce((a, b) => a * b, 1) * 4) }
}

export function network(widths = [6, 4, 4, 3], activation = 'Tanh') {
  const input = (name: string, width: number) => ({ name, type: { tensorType: {
    elemType: 1, shape: { dim: [{ dimValue: 1 }, { dimValue: width }] },
  } } })
  const graph = pb.onnx.GraphProto.create({
    name: 'Dense contract fixture', input: [input('observation', 6)],
    output: [input('action_scores', 3)], initializer: [], node: [],
  })
  let previous = 'observation'
  for (let i = 0; i < widths.length - 1; i++) {
    const final = i === widths.length - 2, output = final ? 'action_scores' : `linear${i}`
    graph.initializer.push(tensor(`w${i}`, [widths[i + 1], widths[i]]), tensor(`b${i}`, [widths[i + 1]]))
    graph.node.push({ opType: 'Gemm', input: [previous, `w${i}`, `b${i}`], output: [output],
      attribute: [{ name: 'transB', type: 2, i: 1 }] })
    previous = output
    if (!final) {
      previous = `hidden${i}`
      graph.node.push({ opType: activation, input: [output], output: [previous] })
    }
  }
  return pb.onnx.ModelProto.create({ irVersion: 10, opsetImport: [{ domain: '', version: 18 }], graph })
}

export const encode = (model: ReturnType<typeof network>) => pb.onnx.ModelProto.encode(model).finish()

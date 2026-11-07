import { presets } from '../src/roster'
import { modelFile } from './modelFiles'
import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import pb from 'onnx-proto'
import { DenseModelError, validateArenaModel, validateModel } from '../src/validate'
import { fitsDivision, smallestDivision } from '../src/competition'

function tensor(name: string, dims: number[]) {
  return { name, dims, dataType: 1, rawData: new Uint8Array(dims.reduce((a, b) => a * b, 1) * 4) }
}

function network(widths = [6, 4, 4, 3], activation = 'Tanh') {
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

const encode = (model: ReturnType<typeof network>) => pb.onnx.ModelProto.encode(model).finish()

test('the active contract allows width and depth tradeoffs through four hidden layers', () => {
  for (const widths of [[6, 3], [6, 16, 3], [6, 7, 11, 3], [6, 16, 12, 8, 3], [6, 8, 8, 8, 8, 3]]) {
    for (const activation of ['Relu', 'Tanh']) {
      const stats = validateArenaModel(encode(network(widths, activation)))
      assert.deepEqual(stats.architecture.widths, widths)
    }
  }
  const ordinary = network()
  ordinary.graph!.node[0].attribute!.push(
    { name: 'transA', type: 2, i: 0 }, { name: 'alpha', type: 1, f: 1 }, { name: 'beta', type: 1, f: 1 },
  )
  assert.deepEqual(validateArenaModel(encode(ordinary)).architecture.widths, [6, 4, 4, 3])
})

test('deeper dense networks stay Small only within the same parameter and compute budgets', () => {
  const compact = validateArenaModel(encode(network([6, 4, 4, 3], 'Relu')))
  assert.deepEqual([compact.params, compact.macs, smallestDivision(compact)], [63, 52, 'small'])
  const wider = validateArenaModel(encode(network([6, 16, 12, 8, 3], 'Relu')))
  assert.equal(wider.params, 447)
  assert.equal(fitsDivision(wider, 'small'), false, 'Extra depth does not grant a larger parameter budget')
})

test('five hidden layers exceed the structural limit even within Small resource budgets', () => {
  const bytes = encode(network([6, 2, 2, 2, 2, 2, 3]))
  assert.equal(smallestDivision(validateModel(bytes)), 'small')
  assert.throws(() => validateArenaModel(bytes), DenseModelError,
    'A small parameter count does not permit an unbounded instruction chain')
})

test('matrix orientation does not change eligibility when every layer is dense', () => {
  const model = network()
  for (const node of model.graph!.node.filter((node) => node.opType === 'Gemm')) {
    const weight = model.graph!.initializer.find((tensor) => tensor.name === node.input![1])!
    weight.dims.reverse()
    node.attribute = []
  }
  assert.deepEqual(validateArenaModel(encode(model)).architecture.widths, [6, 4, 4, 3])
})

test('all existing plain policies remain playable under the dense contract', () => {
  for (const name of ['tracker', 'anticipator', 'challenger', 'pacer-v4', 'apprentice-v4',
    'contender-v3', 'selfplay-v3', 'apprentice-v3', 'selfplay-v2']) {
    const stats = validateArenaModel(readFileSync(modelFile(`${name}`)))
    assert.ok(stats.architecture.widths)
    assert.deepEqual([stats.inputSize, stats.outputSize], [6, 3])
  }
})

test('physics programs and hybrid features remain inspectable archives but cannot enter the arena', () => {
  for (const name of ['handcrafted-nano-v4', 'handcrafted-small-v4', 'counter-attacker-v4', 'strategist-v3']) {
    const bytes = readFileSync(modelFile(`${name}`))
    assert.equal(validateModel(bytes).inputSize, 6)
    assert.throws(() => validateArenaModel(bytes), DenseModelError)
    const disguised = pb.onnx.ModelProto.decode(bytes)
    disguised.metadataProps = [{ key: 'architecture', value: '6-16-16-3 Tanh dense network' }]
    assert.throws(() => validateArenaModel(encode(disguised)), DenseModelError,
      'A label must not turn an arithmetic program into a dense policy')
  }
})

test('runtime-computed weights and residual branches do not count as dense layers', () => {
  const dynamic = network([6, 3]), graph = dynamic.graph!
  graph.node[0].output = ['runtimeRow']
  graph.initializer.push(tensor('column', [6, 1]))
  graph.node.push({ opType: 'MatMul', input: ['column', 'runtimeRow'], output: ['runtimeWeights'] },
    { opType: 'Gemm', input: ['observation', 'runtimeWeights', 'b0'], output: ['action_scores'] })
  assert.equal(validateModel(encode(dynamic)).outputSize, 3)
  assert.throws(() => validateArenaModel(encode(dynamic)), DenseModelError)

  const residual = network([6, 4, 3])
  residual.graph!.node[2].input![0] = 'residual'
  residual.graph!.node.splice(2, 0,
    { opType: 'Add', input: ['linear0', 'hidden0'], output: ['residual'] })
  assert.equal(validateModel(encode(residual)).outputSize, 3)
  assert.throws(() => validateArenaModel(encode(residual)), DenseModelError)
})

test('dense weights and biases belong to one layer and no constants may be hidden unused', () => {
  const shared = network([6, 6, 6, 3]), graph = shared.graph!
  graph.node[2].input = ['hidden0', 'w0', 'b0']
  graph.initializer = graph.initializer.filter((tensor) => !['w1', 'b1'].includes(tensor.name!))
  assert.equal(validateModel(encode(shared)).outputSize, 3)
  assert.throws(() => validateArenaModel(encode(shared)), DenseModelError)

  const unused = network()
  unused.graph!.initializer.push(tensor('unusedLookupTable', [10]))
  assert.throws(() => validateArenaModel(encode(unused)), DenseModelError)
})

test('scaling constants cannot carry extra instructions outside the dense weights', () => {
  for (const [name, type, value] of [['alpha', 1, 2], ['beta', 1, 0], ['alpha', 2, 1]] as const) {
    const model = network()
    model.graph!.node[0].attribute!.push({ name, type, ...(type === 1 ? { f: value } : { i: value }) })
    assert.throws(() => validateArenaModel(encode(model)), DenseModelError)
  }
  const transposedBatch = network([6, 1, 3])
  transposedBatch.graph!.node[2].attribute!.push({ name: 'transA', type: 2, i: 1 })
  assert.equal(validateModel(encode(transposedBatch)).outputSize, 3)
  assert.throws(() => validateArenaModel(encode(transposedBatch)), DenseModelError)

  const duplicate = network()
  duplicate.graph!.node[0].attribute!.push({ name: 'transB', type: 2, i: 1 })
  assert.throws(() => validateArenaModel(encode(duplicate)), DenseModelError)
})

test('hidden activations are uniform and the final action scores are linear', () => {
  const mixed = network()
  mixed.graph!.node[3].opType = 'Relu'
  assert.throws(() => validateArenaModel(encode(mixed)), DenseModelError)

  const activatedOutput = network()
  activatedOutput.graph!.node[4].output = ['linearOutput']
  activatedOutput.graph!.node.push({ opType: 'Tanh', input: ['linearOutput'], output: ['action_scores'] })
  assert.throws(() => validateArenaModel(encode(activatedOutput)), DenseModelError)
})

test('every parameter must be finite and stored only once in its float32 tensor', () => {
  for (const index of [0, 1]) for (const value of [NaN, Infinity, -Infinity]) {
    const model = network(), raw = model.graph!.initializer[index].rawData!
    new DataView(raw.buffer, raw.byteOffset, raw.byteLength).setFloat32(0, value, true)
    assert.throws(() => validateArenaModel(encode(model)), DenseModelError)
  }
  const repeatedStorage = network()
  repeatedStorage.graph!.initializer[0].floatData = Array(24).fill(0)
  assert.throws(() => validateArenaModel(encode(repeatedStorage)), DenseModelError)

  const extraTypedStorage = network()
  extraTypedStorage.graph!.initializer[0].int32Data = [123]
  assert.throws(() => validateArenaModel(encode(extraTypedStorage)), DenseModelError)
})

test('the shipped collection contains only current eligible models', () => {
  const directory = new URL('../public/models/', import.meta.url)
  const files = readdirSync(directory).filter((name) => name.endsWith('.onnx')).sort()
  assert.deepEqual(files, presets.map((preset) => preset.id + '.onnx').sort(),
    'Every shipped model must be registered in the active roster, with no extra files')
  for (const file of files) assert.ok(validateArenaModel(readFileSync(new URL(file, directory))).architecture.widths)
})

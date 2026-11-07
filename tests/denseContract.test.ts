import { presets } from '../src/roster'
import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { DenseModelError, MAX_PARAMS, validateArenaModel } from '../src/validate'
import { fitsDivision, smallestDivision } from '../src/competition'
import { encode, network, tensor } from './modelFactory'

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

test('runtime-computed weights and residual branches do not count as dense layers', () => {
  const dynamic = network([6, 3]), graph = dynamic.graph!
  graph.node[0].output = ['runtimeRow']
  graph.initializer.push(tensor('column', [6, 1]))
  graph.node.push({ opType: 'MatMul', input: ['column', 'runtimeRow'], output: ['runtimeWeights'] },
    { opType: 'Gemm', input: ['observation', 'runtimeWeights', 'b0'], output: ['action_scores'] })
  assert.throws(() => validateArenaModel(encode(dynamic)), DenseModelError)

  const residual = network([6, 4, 3])
  residual.graph!.node[2].input![0] = 'residual'
  residual.graph!.node.splice(2, 0,
    { opType: 'Add', input: ['linear0', 'hidden0'], output: ['residual'] })
  assert.throws(() => validateArenaModel(encode(residual)), DenseModelError)
})

test('dense weights and biases belong to one layer and no constants may be hidden unused', () => {
  const shared = network([6, 6, 6, 3]), graph = shared.graph!
  graph.node[2].input = ['hidden0', 'w0', 'b0']
  graph.initializer = graph.initializer.filter((tensor) => !['w1', 'b1'].includes(tensor.name!))
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

test('labels cannot make a branched graph eligible or override its real parameter count', () => {
  const model = network([6, 4, 3])
  model.metadataProps = [{ key: 'architecture', value: 'plain dense network' }, { key: 'params', value: '1' }]
  assert.equal(validateArenaModel(encode(model)).params, 43)
  model.graph!.node[2].input![0] = 'residual'
  model.graph!.node.splice(2, 0, { opType: 'Add', input: ['linear0', 'hidden0'], output: ['residual'] })
  assert.throws(() => validateArenaModel(encode(model)), DenseModelError)
})

test('uploads are self-contained bounded inference files with the exact game ports', () => {
  const malformed: [string, (model: ReturnType<typeof network>) => void][] = [
    ['different opset', (m) => { m.opsetImport[0].version = 17 }],
    ['custom domain', (m) => { m.graph!.node[0].domain = 'custom' }],
    ['extra input', (m) => { m.graph!.input.push(m.graph!.input[0]) }],
    ['wrong input width', (m) => { m.graph!.input[0].type!.tensorType!.shape!.dim![1].dimValue = 8 }],
    ['dynamic batch', (m) => { m.graph!.input[0].type!.tensorType!.shape!.dim![0].dimParam = 'batch' }],
    ['wrong output width', (m) => { m.graph!.output[0].type!.tensorType!.shape!.dim![1].dimValue = 5 }],
    ['wrong input type', (m) => { m.graph!.input[0].type!.tensorType!.elemType = 11 }],
    ['external weights', (m) => { m.graph!.initializer[0].externalData = [{ key: 'location', value: 'weights.bin' }] }],
    ['wrong parameter type', (m) => { m.graph!.initializer[0].dataType = 11 }],
    ['missing parameter bytes', (m) => { m.graph!.initializer[0].rawData = new Uint8Array() }],
    ['oversized parameter dimensions', (m) => { m.graph!.initializer[0].dims = [Number.MAX_SAFE_INTEGER, 2] }],
    ['duplicate parameter names', (m) => { m.graph!.initializer[1].name = m.graph!.initializer[0].name }],
    ['output overwrites input', (m) => { m.graph!.node[0].output = ['observation'] }],
    ['output overwrites weight', (m) => { m.graph!.node[0].output = ['w0'] }],
    ['extra operator output', (m) => { m.graph!.node[0].output!.push('extra') }],
    ['wrong final layer width', (m) => { m.graph!.initializer.at(-1)!.dims = [4] }],
    ['missing bias', (m) => { m.graph!.node[0].input!.pop() }],
    ['control flow', (m) => { m.graph!.node[1].opType = 'Loop' }],
    ['subgraph attribute', (m) => { m.graph!.node[0].attribute!.push({ name: 'alpha', type: 1, f: 1, g: { name: 'hidden' } }) }],
  ]
  for (const [name, mutate] of malformed) {
    const model = network()
    mutate(model)
    assert.throws(() => validateArenaModel(encode(model)), DenseModelError, name)
  }
  for (const bytes of [new Uint8Array(), new Uint8Array([255]), new Uint8Array(262145), encode(network([6, 256, 256, 3]))])
    assert.throws(() => validateArenaModel(bytes), DenseModelError)
})

test('ordinary ONNX float_data storage has the same costs as raw float32 bytes', () => {
  const model = network(), expected = validateArenaModel(encode(model))
  for (const tensor of model.graph!.initializer) {
    tensor.floatData = Array(tensor.rawData!.length / 4).fill(0)
    tensor.rawData = new Uint8Array()
  }
  assert.deepEqual(validateArenaModel(encode(model)), expected)
  model.graph!.initializer[0].floatData![0] = Infinity
  assert.throws(() => validateArenaModel(encode(model)), DenseModelError)
})

test('ports cannot hide contradictory non-tensor type declarations', () => {
  for (const port of ['input', 'output'] as const) {
    for (const declaration of ['sequenceType', 'mapType', 'optionalType', 'sparseTensorType'] as const) {
      const model = network()
      model.graph![port][0].type![declaration] = {}
      assert.throws(() => validateArenaModel(encode(model)), DenseModelError, `${port}: ${declaration}`)
    }
  }
})

test('Gemm attributes contain only their declared scalar payload', () => {
  const malformed: [string, Record<string, unknown>][] = [
    ['missing integer payload', { name: 'transB', type: 2 }],
    ['integer with extra float payload', { name: 'transB', type: 2, i: 1, f: 0 }],
    ['float with extra integer payload', { name: 'alpha', type: 1, f: 1, i: 0 }],
    ['referenced attribute', { name: 'alpha', type: 1, f: 1, refAttrName: 'hidden' }],
    ['sparse tensor payload', { name: 'alpha', type: 1, f: 1, sparseTensor: {} }],
    ['sparse tensor list', { name: 'alpha', type: 1, f: 1, sparseTensors: [{}] }],
    ['type payload', { name: 'alpha', type: 1, f: 1, tp: {} }],
    ['type list', { name: 'alpha', type: 1, f: 1, typeProtos: [{}] }],
  ]
  for (const [name, attr] of malformed) {
    const model = network()
    if (attr.name === 'transB') model.graph!.node[0].attribute = [attr]
    else model.graph!.node[0].attribute!.push(attr)
    assert.throws(() => validateArenaModel(encode(model)), DenseModelError, name)
  }
})

test('intermediate type annotations must agree with actual dense tensors', () => {
  const model = network()
  model.graph!.valueInfo = [{ name: 'hidden0', type: { tensorType: {
    elemType: 1, shape: { dim: [{ dimValue: 1 }, { dimValue: 4 }] },
  } } }]
  assert.deepEqual(validateArenaModel(encode(model)).architecture.widths, [6, 4, 4, 3])
  model.graph!.valueInfo[0].type!.tensorType!.shape!.dim![1].dimValue = 5
  assert.throws(() => validateArenaModel(encode(model)), DenseModelError)
  model.graph!.valueInfo[0].type!.tensorType!.shape!.dim![1].dimValue = 4
  model.graph!.valueInfo.push(model.graph!.valueInfo[0])
  assert.throws(() => validateArenaModel(encode(model)), DenseModelError)
  model.graph!.valueInfo = [{ name: 'unconnected' }]
  assert.throws(() => validateArenaModel(encode(model)), DenseModelError)
})

test('Large budgets count stored biases and zeros as well as matrix weights', () => {
  const within = validateArenaModel(encode(network([6, 245, 245, 3])))
  assert.ok(within.params <= MAX_PARAMS)
  assert.equal(smallestDivision(within), 'large')
  // All values are zero; the stored tensors still exceed the published cap.
  assert.throws(() => validateArenaModel(encode(network([6, 251, 251, 3]))), DenseModelError)
})

test('the shipped collection contains only current eligible models', () => {
  const directory = new URL('../public/models/', import.meta.url)
  const files = readdirSync(directory).filter((name) => name.endsWith('.onnx')).sort()
  assert.deepEqual(files, presets.map((preset) => preset.id + '.onnx').sort(),
    'Every shipped model must be registered in the active roster, with no extra files')
  for (const file of files) assert.ok(validateArenaModel(readFileSync(new URL(file, directory))).architecture.widths)
})

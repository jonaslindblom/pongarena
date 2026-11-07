import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import pb from 'onnx-proto'
import { encode, network } from './modelFactory'
import { activationLevels, createBrainGraph } from '../src/brain'

test('inspection exposes every hidden activation without changing the competing network or its identity', () => {
  for (const name of readdirSync(new URL('../public/models/', import.meta.url))) {
    if (!name.endsWith('.onnx')) continue
    const bytes = new Uint8Array(readFileSync(new URL(`../public/models/${name}`, import.meta.url)))
    const original = bytes.slice()
    const before = pb.onnx.ModelProto.decode(bytes)
    const inspection = createBrainGraph(bytes)
    const after = pb.onnx.ModelProto.decode(inspection.bytes)
    assert.deepEqual(bytes, original, 'Inspection must not mutate the uploaded file or its hash')
    assert.deepEqual(after.graph!.node, before.graph!.node, 'Playing decisions must retain the original operators')
    assert.deepEqual(after.graph!.initializer!.map((tensor) => pb.onnx.TensorProto.encode(tensor).finish()),
      before.graph!.initializer!.map((tensor) => pb.onnx.TensorProto.encode(tensor).finish()), 'Inspection must not change weights')
    assert.deepEqual(after.graph!.output![0], before.graph!.output![0], 'Original scores stay available for parity checks')
    const hidden = before.graph!.node!.filter((node) => node.opType === 'Relu' || node.opType === 'Tanh')
    assert.deepEqual(inspection.layers.filter((layer) => layer.kind === 'hidden').map((layer) => layer.name), hidden.map((node) => node.output![0]),
      'Show post-activation neuron values, not pre-activation affine sums')
    assert.deepEqual(after.graph!.output!.slice(1).map((output) => output.name), hidden.map((node) => node.output![0]))
    for (const layer of inspection.layers.slice(1)) {
      const output = after.graph!.output!.find((output) => output.name === layer.name)!
      assert.deepEqual(output.type!.tensorType!.shape!.dim!.map((dim) => Number(dim.dimValue)), [1, layer.width])
    }
  }
})

test('inspection supports an eligible linear policy with no hidden layers', () => {
  const tensor = (name: string, dims: number[]) => ({ name, dims, dataType: 1, floatData: Array(dims.reduce((a,b) => a*b,1)).fill(0) })
  const port = (name: string, width: number) => ({ name, type: { tensorType: { elemType: 1, shape: { dim: [{ dimValue: 1 }, { dimValue: width }] } } } })
  const model = pb.onnx.ModelProto.create({ irVersion: 10, opsetImport: [{ version: 18 }], graph: {
    name: 'Linear policy', input: [port('observation',6)], output: [port('action_scores',3)],
    initializer: [tensor('weight',[6,3]),tensor('bias',[3])],
    node: [{ opType: 'Gemm', input: ['observation','weight','bias'], output: ['action_scores'] }],
  } })
  const inspection = createBrainGraph(pb.onnx.ModelProto.encode(model).finish())
  assert.deepEqual(inspection.layers.map((layer) => layer.kind), ['input','output'])
  assert.equal(pb.onnx.ModelProto.decode(inspection.bytes).graph!.output!.length,1)
})

test('inspection cannot make a branched graph eligible for play', () => {
  const model = network([6, 4, 3])
  model.graph!.node[2].input![0] = 'residual'
  model.graph!.node.splice(2, 0, { opType: 'Add', input: ['linear0', 'hidden0'], output: ['residual'] })
  assert.throws(() => createBrainGraph(encode(model)), /plain dense network/)
})


test('small real activations remain visible while genuine zeros stay inactive', () => {
  const small = activationLevels([0, 0.00004, -0.00001])
  assert.equal(small.peak, 0.00004)
  assert.deepEqual(small.intensities, [0, 1, 0.5], 'A hidden layer below one must still use the visible intensity range')
  assert.deepEqual(activationLevels([0, 0, 0]), { peak: 0, intensities: [0, 0, 0] }, 'An inactive ReLU layer must never acquire artificial activity')
  assert.deepEqual(activationLevels([0, 4, -1]).intensities, small.intensities, 'Display intensity depends on relative magnitude, not a model-specific numeric scale')
})

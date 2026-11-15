import test from 'node:test'
import assert from 'node:assert/strict'
import { modelIdentity } from '../src/modelIdentity'
import { sha256 } from '../src/competition'
import { encode, network } from './modelFactory'

function populated() {
  const model = network([6, 4, 3])
  for (const [tensorIndex, tensor] of model.graph.initializer.entries()) {
    const data = tensor.rawData!
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength)
    for (let index = 0; index < data.byteLength / 4; index++)
      view.setFloat32(index * 4, (tensorIndex + 1) * (index + 1) / 32, true)
  }
  return model
}

test('description and producer edits keep evaluation identity while file integrity still changes', async () => {
  const model = populated(), before = encode(model)
  model.docString = 'A revised explanation of how these weights were trained.'
  model.metadataProps = [{ key: 'pongarena.name', value: 'New display name' }]
  model.producerName = 'Different exporter'
  model.producerVersion = '2'
  model.modelVersion = 42
  model.graph.name = 'Another graph name'
  model.graph.docString = 'Graph documentation'
  model.graph.node[0].docString = 'Layer documentation'
  model.graph.initializer[0].docString = 'Tensor documentation'
  const after = encode(model)
  assert.notEqual(await sha256(after), await sha256(before))
  assert.equal(await modelIdentity(after), await modelIdentity(before), 'prose edits must not reroll evaluation serves')
})

test('internal renaming and initializer ordering cannot disguise a duplicate model', async () => {
  const model = populated(), before = encode(model)
  const rename = (name: string) => ['observation', 'action_scores'].includes(name) ? name : 'renamed_' + name
  for (const tensor of model.graph.initializer) tensor.name = rename(tensor.name!)
  for (const node of model.graph.node) {
    node.name = 'Documentation-only node name'
    node.input = node.input!.map(rename)
    node.output = node.output!.map(rename)
  }
  model.graph.initializer.reverse()
  assert.equal(await modelIdentity(encode(model)), await modelIdentity(before))
})

test('equivalent tensor encodings, explicit default attributes and matrix storage share identity', async () => {
  const model = populated(), before = encode(model)
  for (const tensor of model.graph.initializer) {
    const data = tensor.rawData!, view = new DataView(data.buffer, data.byteOffset, data.byteLength)
    tensor.floatData = Array.from({ length: data.byteLength / 4 }, (_, index) => view.getFloat32(index * 4, true))
    delete tensor.rawData
  }
  for (const node of model.graph.node.filter(node => node.opType === 'Gemm')) {
    const weight = model.graph.initializer.find(tensor => tensor.name === node.input![1])!
    const [outputs, inputs] = weight.dims!.map(Number), data = weight.floatData!
    weight.floatData = Array.from({ length: data.length }, (_, index) => data[(index % outputs) * inputs + Math.floor(index / outputs)])
    weight.dims = [inputs, outputs]
    node.attribute = [{ name: 'alpha', type: 1, f: 1 }, { name: 'beta', type: 1, f: 1 }, { name: 'transA', type: 2, i: 0 }]
  }
  assert.equal(await modelIdentity(encode(model)), await modelIdentity(before))
})

test('weight, bias, activation and layout changes require a different evaluation', async () => {
  const baseline = await modelIdentity(encode(populated()))
  for (const tensorIndex of [0, 1]) {
    const model = populated(), data = model.graph.initializer[tensorIndex].rawData!
    new DataView(data.buffer, data.byteOffset, data.byteLength).setFloat32(0, 100, true)
    assert.notEqual(await modelIdentity(encode(model)), baseline)
  }
  const changedActivation = populated()
  changedActivation.graph.node[1].opType = 'Relu'
  assert.notEqual(await modelIdentity(encode(changedActivation)), baseline)
  assert.notEqual(await modelIdentity(encode(network([6, 3, 3]))), baseline)
})

test('signed zero is represented explicitly and invalid graphs never receive an inference identity', async () => {
  const model = network([6, 3]), positive = await modelIdentity(encode(model))
  new DataView(model.graph.initializer[0].rawData!.buffer).setFloat32(0, -0, true)
  const negative = await modelIdentity(encode(model))
  assert.notEqual(negative, positive)
  const weight = model.graph.initializer[0]
  weight.floatData = Array(18).fill(0)
  weight.floatData[0] = -0
  delete weight.rawData
  assert.equal(await modelIdentity(encode(model)), negative)
  model.graph.node[0].opType = 'Add'
  await assert.rejects(modelIdentity(encode(model)), /Gemm/)
})

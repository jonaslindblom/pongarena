import test from 'node:test'
import assert from 'node:assert/strict'
import { readModelMetadata, methodLabel } from '../src/modelMetadata'
import { encode, network } from './modelFactory'

function submission() {
  const model = network()
  model.docString = 'Trained from random weights with reinforcement learning.\nSelected on held-out serves.'
  model.metadataProps = [{ key: 'pongarena.method', value: 'trained' }]
  return model
}

test('one ONNX file carries the contributor description, method, optional name and source', () => {
  const model = submission()
  model.metadataProps.push({ key: 'pongarena.name', value: 'Falcon' },
    { key: 'pongarena.source', value: 'https://example.com/training' })
  assert.deepEqual(readModelMetadata(encode(model), 'falcon', true), {
    name: 'Falcon', description: model.docString, method: 'trained', sourceUrl: 'https://example.com/training',
  })
  assert.equal(methodLabel('programmed'), 'Programmed weights')
})

test('new submissions explain their creation while existing local files need no embedded metadata', () => {
  const model = network()
  assert.deepEqual(readModelMetadata(encode(model), 'My model'), { name: 'My model', description: '' })
  assert.throws(() => readModelMetadata(encode(model), 'candidate', true), /doc_string/)
  model.docString = 'Weights calculated from formulas.'
  assert.throws(() => readModelMetadata(encode(model), 'candidate', true), /pongarena.method/)
  const valid = submission()
  assert.equal(readModelMetadata(encode(valid), 'candidate', true).name, 'candidate')
  assert.equal(methodLabel(), 'Imported model')
})

test('embedded claims cannot set trusted origin or submitter credit', () => {
  const model = submission()
  model.docString = '<script>alert("untrusted")</script>'
  model.metadataProps.push({ key: 'pongarena.origin', value: 'core' },
    { key: 'submittedBy', value: 'maintainer' }, { key: 'model_author', value: 'Someone else' })
  assert.deepEqual(readModelMetadata(encode(model), 'candidate', true), {
    name: 'candidate', description: model.docString, method: 'trained',
  })
})

test('ambiguous or invalid metadata fails instead of silently changing how the site presents a model', () => {
  for (const entry of [
    { key: 'pongarena.name', value: '' }, { key: 'pongarena.name', value: 'x'.repeat(81) },
    { key: 'pongarena.name', value: 'First\nSecond' },
    { key: 'pongarena.method', value: 'unknown' }, { key: 'pongarena.method', value: '' },
    { key: 'pongarena.source', value: 'javascript:alert(1)' },
    { key: 'pongarena.source', value: 'http://example.com' },
    { key: 'pongarena.source', value: 'https://user:password@example.com' },
    { key: 'pongarena.source', value: 'https://example.com/\npath' },
    { key: '', value: 'empty key' }, { key: 'x'.repeat(129), value: 'oversized key' },
    { key: 'extra', value: 'x'.repeat(8193) },
  ]) {
    const model = submission()
    model.metadataProps = model.metadataProps.filter(prop => prop.key !== entry.key)
    model.metadataProps.push(entry)
    assert.throws(() => readModelMetadata(encode(model), 'candidate', true), entry.key + ': ' + entry.value.slice(0, 80))
  }
  const repeated = submission()
  repeated.metadataProps.push({ key: 'pongarena.method', value: 'prompted' })
  assert.throws(() => readModelMetadata(encode(repeated), 'candidate'), /Duplicate/)
  const oversized = submission()
  oversized.docString = 'x'.repeat(4001)
  assert.throws(() => readModelMetadata(encode(oversized), 'candidate'), /4,000/)
  oversized.docString = 'Description'
  oversized.metadataProps = Array.from({ length: 65 }, (_, index) => ({ key: 'key' + index, value: '' }))
  assert.throws(() => readModelMetadata(encode(oversized), 'candidate'), /64 ONNX metadata entries/)
})

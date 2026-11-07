import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { validateModelCollection, smokeTestModels, validateSubmissionAttribution } from '../scripts/validate-models'
import { encode, network } from './modelFactory'

const roster = [{ id: 'candidate', name: 'Candidate', submittedBy: 'jonaslindblom' }]

async function collection(action: (folder: string) => Promise<void>) {
  const folder = await mkdtemp(join(tmpdir(), 'pong-model-contract-'))
  try {
    await writeFile(join(folder, 'candidate.onnx'), encode(network()))
    await action(folder)
  } finally { await rm(folder, { recursive: true, force: true }) }
}

test('submission gate freezes actual model bytes, hash and validated resource counts', async () => {
  await collection(async (folder) => {
    const models = await validateModelCollection(folder, roster)
    assert.equal(models.length, 1)
    assert.deepEqual(models[0].stats.architecture.widths, [6, 4, 4, 3])
    assert.equal(models[0].entrant.params, 63)
    assert.equal(models[0].entrant.macs, 52)
    assert.match(models[0].entrant.sha256, /^[a-f0-9]{64}$/)
    assert.equal(models[0].submittedBy, 'jonaslindblom')
    assert.deepEqual(models[0].bytes, await readFile(join(folder, 'candidate.onnx')))
    await smokeTestModels(models)
  })
})

test('new model bytes credit the PR author and existing bytes preserve their original submitter', () => {
  const oldModel = { entrant: { id: 'existing', sha256: 'a'.repeat(64) }, submittedBy: 'original-user' }
  const newModel = { entrant: { id: 'new', sha256: 'b'.repeat(64) }, submittedBy: 'new-user' }
  assert.doesNotThrow(() => validateSubmissionAttribution([oldModel, newModel], [oldModel], 'new-user'))
  assert.doesNotThrow(() => validateSubmissionAttribution([
    { ...oldModel, entrant: { ...oldModel.entrant, id: 'renamed' }, submittedBy: 'ORIGINAL-USER' },
    { ...newModel, submittedBy: 'New-User' },
  ], [oldModel], 'new-user'))
  assert.throws(() => validateSubmissionAttribution([
    { ...newModel, submittedBy: 'someone-else' },
  ], [oldModel], 'new-user'), /PR author @new-user/)
  assert.throws(() => validateSubmissionAttribution([
    { ...oldModel, submittedBy: 'new-user' },
  ], [oldModel], 'new-user'), /existing submitter @original-user/)
  assert.throws(() => validateSubmissionAttribution([
    { ...oldModel, entrant: { ...oldModel.entrant, sha256: 'c'.repeat(64) } },
  ], [oldModel], 'new-user'), /PR author @new-user/)
  assert.throws(() => validateSubmissionAttribution([newModel], [oldModel], 'https://github.com/new-user'), /authenticated PR author/)
})

test('unregistered, missing, nested and duplicate model files fail submission validation', async () => {
  await collection(async (folder) => {
    await assert.rejects(validateModelCollection(folder, []), /roster/)
    await assert.rejects(validateModelCollection(folder, [...roster, { id: 'missing', name: 'Missing', submittedBy: 'jonaslindblom' }]), /roster/)
    await writeFile(join(folder, 'unregistered.onnx'), encode(network()))
    await assert.rejects(validateModelCollection(folder, roster), /roster/)
    await assert.rejects(validateModelCollection(folder, [...roster, { id: 'unregistered', name: 'Copy', submittedBy: 'jonaslindblom' }]), /Identical/)
    await mkdir(join(folder, 'nested'))
    await writeFile(join(folder, 'nested', 'hidden.onnx'), encode(network()))
    await assert.rejects(validateModelCollection(folder, roster), /subdirectories/)
  })
})

test('submission validation requires a GitHub submitter handle and rejects arbitrary profile URLs', async () => {
  await collection(async folder => {
    for (const submittedBy of ['', '-name', 'name-', 'two--hyphens', 'has_underscore',
      'a'.repeat(40), 'https://github.com/name', 'javascript:alert(1)', 'name/other', 'nämé'])
      await assert.rejects(validateModelCollection(folder, [{ ...roster[0], submittedBy }]), /submittedBy/)
    const missing = [{ id: 'candidate', name: 'Candidate' }] as unknown as typeof roster
    await assert.rejects(validateModelCollection(folder, missing), /submittedBy/)
  })
})

test('invalid shape, depth, storage and file-size submissions fail before league play', async () => {
  await collection(async (folder) => {
    const invalid = [
      encode(network([6, 2, 2, 2, 2, 2, 3])),
      encode(network([6, 251, 251, 3])),
      new Uint8Array(256 * 1024 + 1),
    ]
    const wrongInput = network()
    wrongInput.graph!.input[0].type!.tensorType!.shape!.dim![1].dimValue = 7
    invalid.push(encode(wrongInput))
    const wrongOutput = network()
    wrongOutput.graph!.output[0].type!.tensorType!.shape!.dim![1].dimValue = 4
    invalid.push(encode(wrongOutput))
    for (const bytes of invalid) {
      await writeFile(join(folder, 'candidate.onnx'), bytes)
      await assert.rejects(validateModelCollection(folder, roster), /candidate\.onnx:/)
    }
  })
})

test('submitted model directories and files cannot redirect validation through symlinks', async () => {
  await collection(async folder => {
    const link = folder + '-link'
    try {
      await symlink(folder, link, 'dir')
      await assert.rejects(validateModelCollection(link, roster), /without links/)
      await symlink(join(folder, 'candidate.onnx'), join(folder, 'redirect.onnx'))
      await assert.rejects(validateModelCollection(folder, roster), /links/)
    } finally { await rm(link, { force: true }) }
  })
})

test('finite weights that overflow at legal observations fail the runtime gate', async () => {
  await collection(async (folder) => {
    const model = network([6, 3])
    const raw = model.graph!.initializer[0].rawData!
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength)
    // Zero input passes, while a legal positive-velocity extreme overflows.
    view.setFloat32(2 * 4, 3e38, true)
    await writeFile(join(folder, 'candidate.onnx'), encode(model))
    const models = await validateModelCollection(folder, roster)
    await assert.rejects(smokeTestModels(models), /candidate\.onnx:.*finite/)
  })
})

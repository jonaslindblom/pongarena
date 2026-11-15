import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, mkdir, readFile, rename, rm, symlink, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import pb from 'onnx-proto'
import { requireCoreModels, validateModelCollection, smokeTestModels } from '../scripts/validate-models'
import { sha256 } from '../src/competition'
import { encode, network } from './modelFactory'

function submittedNetwork(widths?: number[]) {
  const model = network(widths)
  model.docString = 'Fixed weights calculated directly to test the submission contract.'
  model.metadataProps = [{ key: 'pongarena.method', value: 'programmed' }]
  return model
}

async function collection(action: (folder: string) => Promise<void>) {
  const folder = await mkdtemp(join(tmpdir(), 'pong-model-contract-'))
  try {
    await writeFile(join(folder, 'candidate.onnx'), encode(submittedNetwork()))
    await action(folder)
  } finally { await rm(folder, { recursive: true, force: true }) }
}

test('one ONNX file supplies its description and validated resources without a roster or training script', async () => {
  await collection(async folder => {
    const models = await validateModelCollection(folder)
    assert.equal(models.length, 1)
    assert.deepEqual(models[0].stats.architecture.widths, [6, 4, 4, 3])
    assert.equal(models[0].entrant.id, 'candidate')
    assert.equal(models[0].entrant.name, 'candidate')
    assert.equal(models[0].entrant.params, 63)
    assert.equal(models[0].entrant.macs, 52)
    assert.equal(models[0].metadata.description, submittedNetwork().docString)
    assert.equal(models[0].metadata.method, 'programmed')
    assert.equal(models[0].origin, 'community')
    assert.equal(models[0].submittedBy, undefined, 'GitHub attribution is resolved separately by the trusted catalog builder')
    assert.match(models[0].entrant.sha256, /^[a-f0-9]{64}$/)
    const bytes = await readFile(join(folder, 'candidate.onnx'))
    assert.equal(models[0].fileSha256, await sha256(bytes))
    assert.deepEqual(models[0].bytes, bytes)
    await smokeTestModels(models)
  })
})

test('model discovery follows filenames automatically and preserves network identity across a rename', async () => {
  await collection(async folder => {
    const original = await validateModelCollection(folder)
    await rename(join(folder, 'candidate.onnx'), join(folder, 'renamed.onnx'))
    const renamed = await validateModelCollection(folder)
    assert.equal(renamed[0].entrant.id, 'renamed')
    assert.equal(renamed[0].entrant.name, 'renamed')
    assert.equal(renamed[0].entrant.sha256, original[0].entrant.sha256)
    await writeFile(join(folder, 'second.onnx'), encode(submittedNetwork([6, 3])))
    assert.deepEqual((await validateModelCollection(folder)).map(model => model.entrant.id), ['renamed', 'second'])
  })
})

test('community submissions require an embedded creation description and method', async () => {
  await collection(async folder => {
    const candidate = network()
    await writeFile(join(folder, 'candidate.onnx'), encode(candidate))
    await assert.rejects(validateModelCollection(folder), /candidate\.onnx:.*doc_string/)
    candidate.docString = 'An explanation of how the weights were created.'
    await writeFile(join(folder, 'candidate.onnx'), encode(candidate))
    await assert.rejects(validateModelCollection(folder), /candidate\.onnx:.*pongarena.method/)
  })
})

test('changed metadata cannot disguise an identical network as a second league entrant', async () => {
  await collection(async folder => {
    const duplicate = submittedNetwork()
    duplicate.docString = 'A different description of the same weights.'
    duplicate.metadataProps.push({ key: 'pongarena.name', value: 'Another model' })
    await writeFile(join(folder, 'duplicate.onnx'), encode(duplicate))
    await assert.rejects(validateModelCollection(folder), /Identical networks.*different metadata/)
  })
})

test('only pinned core weights receive core status; changed weights need a new ID and community metadata', async () => {
  await collection(async folder => {
    const bytes = await readFile(new URL('../public/models/quill.onnx', import.meta.url))
    await writeFile(join(folder, 'quill.onnx'), bytes)
    const original = (await validateModelCollection(folder)).find(model => model.entrant.id === 'quill')!
    assert.equal(original.origin, 'core')
    assert.equal(original.submittedBy, 'jonaslindblom')
    const changed = pb.onnx.ModelProto.decode(bytes)
    const raw = changed.graph!.initializer![0].rawData!
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength)
    view.setFloat32(0, view.getFloat32(0, true) + 1, true)
    changed.docString = ''
    changed.metadataProps = []
    await writeFile(join(folder, 'quill.onnx'), pb.onnx.ModelProto.encode(changed).finish())
    await assert.rejects(validateModelCollection(folder), /quill\.onnx:.*Core model IDs are reserved/)
    await rename(join(folder, 'quill.onnx'), join(folder, 'quill-variant.onnx'))
    await assert.rejects(validateModelCollection(folder), /quill-variant\.onnx:.*doc_string/)
    changed.docString = 'A modified network submitted by a contributor.'
    changed.metadataProps = [
      { key: 'pongarena.method', value: 'programmed' },
      { key: 'pongarena.origin', value: 'core' },
      { key: 'pongarena.submittedBy', value: 'someone-else' },
      { key: 'model_author', value: 'Someone Else' },
    ]
    await writeFile(join(folder, 'quill-variant.onnx'), pb.onnx.ModelProto.encode(changed).finish())
    const submitted = (await validateModelCollection(folder)).find(model => model.entrant.id === 'quill-variant')!
    assert.equal(submitted.origin, 'community')
    assert.equal(submitted.submittedBy, undefined, 'embedded text must not claim project status or verified GitHub credit')
  })
})

test('published collections retain the core examples used by lessons and defaults', async () => {
  await collection(async folder => {
    assert.throws(() => requireCoreModels([]), /core|Core/)
    const submitted = await validateModelCollection(folder)
    assert.throws(() => requireCoreModels(submitted), /core|Core/)
    const complete = await validateModelCollection()
    assert.doesNotThrow(() => requireCoreModels(complete))
  })
})

test('nested files and reserved or unsafe model IDs cannot enter the published catalog', async () => {
  await collection(async folder => {
    await rm(join(folder, 'candidate.onnx'))
    for (const file of ['local-shadow.onnx', 'LOCAL-shadow.onnx', 'keyboard.onnx', 'Keyboard.onnx', 'two words.onnx', 'uppercase.ONNX']) {
      await writeFile(join(folder, file), encode(submittedNetwork()))
      await assert.rejects(validateModelCollection(folder), /file-safe|reserved local-/)
      await rm(join(folder, file))
    }
    await mkdir(join(folder, 'nested'))
    await writeFile(join(folder, 'nested', 'hidden.onnx'), encode(submittedNetwork()))
    await assert.rejects(validateModelCollection(folder), /subdirectories/)
  })
})

test('invalid shape, depth, storage and file-size submissions fail before league play', async () => {
  await collection(async folder => {
    const invalid = [
      encode(submittedNetwork([6, 2, 2, 2, 2, 2, 3])),
      encode(submittedNetwork([6, 251, 251, 3])),
      new Uint8Array(256 * 1024 + 1),
    ]
    const wrongInput = submittedNetwork()
    wrongInput.graph.input[0].type!.tensorType!.shape!.dim![1].dimValue = 7
    invalid.push(encode(wrongInput))
    const wrongOutput = submittedNetwork()
    wrongOutput.graph.output[0].type!.tensorType!.shape!.dim![1].dimValue = 4
    invalid.push(encode(wrongOutput))
    for (const bytes of invalid) {
      await writeFile(join(folder, 'candidate.onnx'), bytes)
      await assert.rejects(validateModelCollection(folder), /candidate\.onnx:/)
    }
  })
})

test('submitted model directories and files cannot redirect validation through symlinks', async () => {
  await collection(async folder => {
    const link = folder + '-link'
    try {
      await symlink(folder, link, 'dir')
      await assert.rejects(validateModelCollection(link), /without links/)
      await symlink(join(folder, 'candidate.onnx'), join(folder, 'redirect.onnx'))
      await assert.rejects(validateModelCollection(folder), /links/)
    } finally { await rm(link, { force: true }) }
  })
})

test('finite weights that overflow at legal observations fail the runtime gate', async () => {
  await collection(async folder => {
    const model = submittedNetwork([6, 3])
    const raw = model.graph.initializer[0].rawData!
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength)
    // Zero input passes, while a legal positive-velocity extreme overflows.
    view.setFloat32(2 * 4, 3e38, true)
    await writeFile(join(folder, 'candidate.onnx'), encode(model))
    const models = await validateModelCollection(folder)
    await assert.rejects(smokeTestModels(models), /candidate\.onnx:.*finite/)
  })
})

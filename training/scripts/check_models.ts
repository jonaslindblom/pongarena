import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFileSync, readdirSync, existsSync } from 'node:fs'
import { validateArenaModel } from '../../src/validate'

const digest = (path: string) => createHash('sha256').update(readFileSync(path)).digest('hex')
const paths = process.argv.slice(2)
if (paths.length) {
  for (const path of paths) console.log(JSON.stringify({ file: path, sha256: digest(path), ...validateArenaModel(readFileSync(path)) }))
} else {
  const catalog = JSON.parse(readFileSync('training/catalog.json', 'utf8'))
  assert.deepEqual(catalog.models.map((m: { id: string }) => m.id).sort(),
    readdirSync('public/models').filter((name) => name.endsWith('.onnx')).map((name) => name.slice(0, -5)).sort())
  for (const model of catalog.models) {
    assert.equal(digest(model.file), model.sha256, model.id)
    if (model.checkpoint) {
      if (existsSync(model.checkpoint)) assert.equal(digest(model.checkpoint), model.checkpoint_sha256, model.id + ' checkpoint')
      else console.log(`${model.id}: local checkpoint check skipped (file not present)` )
    }
    const stats = validateArenaModel(readFileSync(model.file))
    console.log(`${model.id}: ${stats.params} parameters, ${stats.macs} MACs`)
  }
}

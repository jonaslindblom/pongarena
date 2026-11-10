import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { setTimeout } from 'node:timers/promises'

const directory = new URL('./attempts/', import.meta.url)
const template = await readFile(new URL('./prompt.txt', import.meta.url), 'utf8')
const model = 'gpt-6-astra'
const attempts = [
  ['astra-97-a', [6, 6, 4, 3, 3]],
  ['astra-97-b', [6, 6, 4, 3, 3]],
  ['astra-97-c', [6, 6, 4, 3, 3]],
  ['astra-163-a', [6, 12, 4, 3, 3]],
  ['astra-163-b', [6, 12, 4, 3, 3]],
  ['astra-97-low', [6, 6, 4, 3, 3], 'low'],
  ['astra-163-low', [6, 12, 4, 3, 3], 'low'],
  ['astra-97-low-b', [6, 6, 4, 3, 3], 'low'],
  ['astra-163-low-b', [6, 12, 4, 3, 3], 'low'],
  ['astra-97-high-bg', [6, 6, 4, 3, 3]],
  ['astra-163-high-bg', [6, 12, 4, 3, 3]],
  ['astra-97-medium-bg', [6, 6, 4, 3, 3], 'medium'],
  ['rules-163-a', [6, 12, 4, 3, 3], 'low', 'rules'],
  ['rules-163-b', [6, 12, 4, 3, 3], 'low', 'rules'],
  ['tactical-163-a', [6, 12, 4, 3, 3], 'low', 'tactical'],
  ['tactical-163-b', [6, 12, 4, 3, 3], 'low', 'tactical'],
]
const numericArray = { type: 'array', items: { type: 'number' } }
const layerSchema = {
  type: 'object',
  properties: {
    weights: { type: 'array', items: numericArray },
    bias: numericArray,
  },
  required: ['weights', 'bias'],
  additionalProperties: false,
}
const schema = {
  type: 'object',
  properties: { layers: { type: 'array', items: layerSchema } },
  required: ['layers'],
  additionalProperties: false,
}

const key = process.env.OPENAI_API_KEY
if (!key) throw new Error('Set OPENAI_API_KEY before running this experiment.')
await mkdir(directory, { recursive: true })

const ids = process.argv.slice(2)
if (!ids.length) throw new Error('Select an attempt: ' + attempts.map(([id]) => id).join(', '))
for (const id of ids) {
  if (!attempts.some(([attempt]) => attempt === id)) throw new Error('Unknown attempt ' + id)
}
const selected = attempts.filter(([id]) => ids.includes(id))
const results = await Promise.allSettled(selected.map(async ([id, widths, effort = 'high', condition]) => {
  const maxOutputTokens = effort === 'high' ? 12_000 : 6_000
  const outputDirectory = condition ? new URL('./follow-up/attempts/', import.meta.url) : directory
  await mkdir(outputDirectory, { recursive: true })
  const rules = template.replace('{{WIDTHS}}', widths.join(' → '))
  const input = condition === 'tactical'
    ? rules + '\n' + await readFile(new URL('./follow-up/tactical-brief.txt', import.meta.url), 'utf8')
    : rules
  // Reserve the attempt before making a paid request to prevent accidental reruns.
  const file = new URL(id + '.json', outputDirectory)
  await writeFile(file, JSON.stringify({ id, model, effort, widths, status: 'started' }), { flag: 'wx' })
  const started = Date.now()
  const endpoint = 'https://api.openai.com/v1/responses'
  const headers = { Authorization: 'Bearer ' + key, 'Content-Type': 'application/json' }
  let response = await fetch(endpoint, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      model, input,
      reasoning: { effort }, max_output_tokens: maxOutputTokens, store: false, background: true,
      text: { format: { type: 'json_schema', name: 'pong_weights', strict: true, schema } },
    }),
    signal: AbortSignal.timeout(30_000),
  })
  let body = await response.json()
  if (body.id) {
    console.log(id, body.status, body.id)
    await writeFile(file, JSON.stringify({ id, model, effort, widths, status: 'started', responseId: body.id }))
  }
  const deadline = started + 900_000
  while (body.status === 'queued' || body.status === 'in_progress') {
    if (Date.now() >= deadline) throw new Error(id + ' exceeded the polling deadline.')
    await setTimeout(7_500)
    const polled = await fetch(endpoint + '/' + body.id, { headers, signal: AbortSignal.timeout(20_000) })
      .catch(() => null)
    if (!polled?.ok) continue
    response = polled
    body = await response.json()
  }
  const outputText = (body.output ?? []).flatMap(item => item.type === 'message' ? item.content : [])
    .filter(item => item.type === 'output_text').map(item => item.text).join('')
  const record = {
    id, widths, condition, model: body.model ?? model, effort, maxOutputTokens, background: true,
    responseId: body.id, status: body.status ?? 'api_error', httpStatus: response.status,
    incompleteDetails: body.incomplete_details, error: body.error && { type: body.error.type, code: body.error.code },
    usage: body.usage, elapsedSeconds: (Date.now() - started) / 1000, outputText,
  }
  await writeFile(file, JSON.stringify(record, null, 2) + '\n')
  console.log(id, record.status, record.httpStatus, record.error ?? '', record.usage ?? '')
  if (record.status !== 'completed') throw new Error(id + ' did not complete.')
}))
if (results.some(result => result.status === 'rejected')) {
  for (const result of results) if (result.status === 'rejected') console.error(result.reason.message)
  process.exitCode = 1
}

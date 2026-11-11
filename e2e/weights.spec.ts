import { test, expect, type Locator, type Page } from '@playwright/test'
import { readFileSync } from 'node:fs'
import pb from 'onnx-proto'
import * as ort from 'onnxruntime-web/wasm'
import { choice, observation, serve, type Frame, type Scores } from '../src/game'

const moves = ['Left', 'Stay', 'Right']
type LabFrame = Pick<Frame, 'x' | 'y' | 'vx' | 'vy' | 'paddles'>
// Physical situations are independent of the production explanation and presets.
const recoveryFrame: LabFrame = { x: 184, y: 120, vx: 13, vy: 0, paddles: [100, 80] }
const browserErrors = new WeakMap<Page, string[]>()

ort.env.wasm.numThreads = 1

test.beforeEach(async ({ page }) => {
  const errors: string[] = []
  browserErrors.set(page, errors)
  page.on('pageerror', error => errors.push(error.message))
})

test.afterEach(async ({ page }) => {
  expect(browserErrors.get(page), 'exploring weights must not cause uncaught runtime errors').toEqual([])
  await expect(page.getByRole('alert')).toHaveCount(0)
})

async function openWeights(page: Page) {
  await page.goto('/')
  await page.getByRole('navigation', { name: 'Main navigation' })
    .getByRole('link', { name: 'How it thinks', exact: true }).click()
  await expect(page.getByRole('group', { name: 'Choose a lesson' })).toBeVisible()
}

async function openCounter(page: Page) {
  await openWeights(page)
  await page.getByRole('button', { name: '2 · Know when to return', exact: true }).click()
  await expect(page.getByLabel('Counter chosen move', { exact: true })).toBeVisible({ timeout: 30_000 })
}

test('a shared lesson URL survives reload and tab navigation history', async ({ page }) => {
  await page.goto('/#how-it-thinks/counter')
  const counter = page.getByRole('heading', { name: 'Know when to stop chasing.' })
  await expect(counter).toBeVisible()
  await page.reload()
  await expect(counter).toBeVisible()

  const models = page.getByRole('navigation').getByRole('link', { name: 'Models', exact: true })
  await expect(models).toHaveAttribute('href', /\/#models$/)
  await models.click()
  await expect(page.getByRole('heading', { name: 'Model library', exact: true })).toBeVisible()
  await page.goBack()
  await expect(counter).toBeVisible()
  await page.goForward()
  await expect(models).toHaveAttribute('aria-current', 'page')
})

async function inspectWeights(page: Page) {
  const summary = page.locator('summary').filter({ hasText: 'Inspect weights and formulas' })
  const details = summary.locator('..')
  if (!await details.evaluate(element => (element as HTMLDetailsElement).open)) await summary.click()
  const inspector = page.getByRole('region', { name: 'Counter weight inspection', exact: true })
  await expect(inspector).toBeVisible()
  return inspector
}

async function setRange(slider: Locator, value: number) {
  const min = Number(await slider.getAttribute('min'))
  const max = Number(await slider.getAttribute('max'))
  const step = Number(await slider.getAttribute('step'))
  const fromEnd = value > (min + max) / 2
  await slider.press(fromEnd ? 'End' : 'Home')
  const count = Math.round(Math.abs(value - (fromEnd ? max : min)) / step)
  for (let i = 0; i < count; i++) await slider.press(fromEnd ? 'ArrowLeft' : 'ArrowRight')
  await expect(slider).toHaveValue(String(value))
}

async function expectOriginalDecision(page: Page, session: ort.InferenceSession, frame: LabFrame) {
  const result = await session.run({ observation: new ort.Tensor('float32', observation({ ...serve(1), ...frame }, 0), [1, 6]) })
  const scores = Array.from(result.action_scores.data as Float32Array) as Scores
  const card = page.getByRole('article', { name: 'Counter decision', exact: true })
  await expect(card.getByLabel('Counter chosen move', { exact: true })).toHaveText(moves[choice(scores)])
  // The logits encode signed target error; keep the displayed target unclamped.
  const target = frame.paddles[0] + 80 * (scores[2] - scores[0]) / 2
  await expect(card.getByLabel('Counter paddle target', { exact: true })).toHaveText(target.toFixed(2))
  await inspectWeights(page)
  for (let action = 0; action < moves.length; action++) {
    await expect.poll(async () => Number(await page.getByLabel(`Counter ${moves[action]} score`, { exact: true }).textContent()))
      .toBeCloseTo(scores[action], 5)
  }
  return scores
}

test('Follow stays fully inspectable and leads directly to the recovery lesson', async ({ page }) => {
  await openWeights(page)
  const picker = page.getByRole('group', { name: 'Choose a lesson' })
  await expect(picker.getByRole('button')).toHaveCount(2)
  await expect(picker.getByRole('button', { name: '1 · Follow the ball', exact: true })).toHaveAttribute('aria-pressed', 'true')
  await expect(picker.getByRole('button', { name: /Attack/ })).toHaveCount(0)
  const follow = page.getByRole('region', { name: 'Interactive Follow model' })
  await expect(follow.getByRole('img', { name: /error 0\.800, Tanh 0\.664\. Follow chooses Right/ }))
    .toBeVisible({ timeout: 30_000 })
  await setRange(page.getByRole('slider', { name: /^Paddle position/ }), .8125)
  await expect(follow.getByRole('img', { name: /error -0\.313, Tanh -0\.303\. Follow chooses Left/ })).toBeVisible()
  await setRange(page.getByRole('slider', { name: /^Ball position/ }), .8125)
  await expect(follow.getByRole('img', { name: /error 0\.000, Tanh 0\.000\. Follow chooses Stay/ })).toBeVisible()
  await follow.getByText('See all 13 parameters', { exact: false }).click()
  await expect(follow.getByText('5 nonzero · zeros count too', { exact: true })).toBeVisible()
  await expect(follow.getByRole('table')).toHaveCount(2)
  await page.getByRole('button', { name: /Next lesson/ }).click()
  await expect(page.getByRole('heading', { name: 'Know when to stop chasing.' })).toBeInViewport()
  await expect(page.getByLabel('Counter chosen move', { exact: true })).toHaveText('Left')
})

test('changing only direction switches the original model between recovery and interception', async ({ page }) => {
  await openCounter(page)
  await expect(page.getByRole('combobox', { name: 'Neuron', exact: true })).toBeHidden()
  const session = await ort.InferenceSession.create(readFileSync('public/models/counter.onnx'), { executionProviders: ['wasm'] })
  try {
    for (const [vx, direction, move, signal] of [[13, 'Away from you', 'Left', 'On'], [-13, 'Toward you', 'Right', 'Off']] as const) {
      await page.getByRole('button', { name: direction, exact: true }).click()
      await expectOriginalDecision(page, session, { ...recoveryFrame, vx })
      await expect(page.getByLabel('Counter chosen move', { exact: true })).toHaveText(move)
      await expect(page.getByLabel('Outgoing detector state', { exact: true })).toContainText(signal)
    }
  } finally { await session.release() }
  await expect(page.getByRole('link', { name: 'Download Counter', exact: true }))
    .toHaveAttribute('href', /\/models\/counter\.onnx$/)
  await expect(page.getByRole('link', { name: 'Source on GitHub', exact: true }))
    .toHaveAttribute('href', 'https://github.com/jonaslindblom/pongarena')
})

test('muting the outgoing detector changes recovery but leaves incoming decisions and the original model intact', async ({ page }) => {
  await openCounter(page)
  const session = await ort.InferenceSession.create(readFileSync('public/models/counter.onnx'), { executionProviders: ['wasm'] })
  try {
    const original = await expectOriginalDecision(page, session, recoveryFrame)
    await expect(page.getByLabel('Counter paddle target', { exact: true })).toHaveText('80.00')
    const toggle = page.getByRole('button', { name: /^(Switch off|Restore) the departure neuron$/ })
    await toggle.click()
    await expect(toggle).toHaveAttribute('aria-pressed', 'true')
    const muted = page.getByRole('article', { name: 'Muted detector decision', exact: true })
    await expect(muted).toBeVisible()
    await expect(muted.getByLabel('Muted detector chosen move', { exact: true })).toHaveText('Right')
    await expect.poll(async () => Number(await muted.getByLabel('Muted detector paddle target', { exact: true }).textContent())).not.toBe(80)
    await expectOriginalDecision(page, session, recoveryFrame)
    await page.getByRole('button', { name: 'Toward you', exact: true }).click()
    const incoming = await expectOriginalDecision(page, session, { ...recoveryFrame, vx: -13 })
    expect(incoming).not.toEqual(original)
    await expect(muted.getByLabel('Muted detector chosen move', { exact: true })).toHaveText(moves[choice(incoming)])
    const incomingTarget = recoveryFrame.paddles[0] + 80 * (incoming[2] - incoming[0]) / 2
    await expect(muted.getByLabel('Muted detector paddle target', { exact: true })).toHaveText(incomingTarget.toFixed(2))
    await toggle.click()
    await expect(toggle).toHaveAttribute('aria-pressed', 'false')
    await expect(muted).toHaveCount(0)
  } finally { await session.release() }
  await page.getByRole('button', { name: 'Play against Counter', exact: true }).click()
  await expect(page.getByRole('combobox', { name: 'Opponent', exact: true })).toHaveValue('counter')
})

test('every designed neuron role opens the matching real weights, and model cards open their lesson', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('navigation').getByRole('link', { name: 'Models', exact: true }).click()
  const card = page.locator('.library-card').filter({ has: page.getByRole('heading', { name: 'Counter', exact: true }) })
  await card.getByRole('button', { name: 'How it thinks', exact: true }).click()
  await expect(page.getByLabel('Counter chosen move', { exact: true })).toHaveText('Left')
  const roles = page.getByRole('region', { name: 'Counter neuron roles', exact: true })
  await expect(roles.getByRole('button')).toHaveCount(16)
  await expect(roles).toContainText('13 hidden neurons. A job for each.')
  await expect(roles).toContainText('not evidence that every neuron in a trained network has one neat purpose')
  for (const [layer, width] of [[0, 6], [1, 4], [2, 3], [3, 3]]) {
    for (let neuron = 0; neuron < width; neuron++) {
      const id = layer === 3 ? moves[neuron] : `H${layer + 1}.${neuron + 1}`
      await roles.getByRole('button', { name: new RegExp(`^Inspect ${id}:`) }).click()
      await expect(page.getByRole('combobox', { name: 'Neuron', exact: true })).toHaveValue(`${layer}:${neuron}`)
      await expect(page.getByLabel('Neuron activation', { exact: true })).toBeVisible()
    }
  }
  await page.getByText('See all 97 parameters', { exact: true }).click()
  await expect(page.getByRole('table', { name: /stored parameters/ })).toHaveCount(4)
})

test('advanced neuron contributions expose the real direction weight and ReLU cutoff', async ({ page }) => {
  const graph = pb.onnx.ModelProto.decode(readFileSync('public/models/counter.onnx')).graph!
  const first = graph.node!.find(node => node.opType === 'Gemm')!
  expect(Number(first.attribute!.find(attribute => attribute.name === 'transB')!.i)).toEqual(1)
  const values = (name: string) => {
    const tensor = graph.initializer!.find(tensor => tensor.name === name)!
    const raw = tensor.rawData
    if (!raw?.length) return Array.from(tensor.floatData ?? [])
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength)
    return Array.from({ length: raw.byteLength / 4 }, (_, index) => view.getFloat32(index * 4, true))
  }
  const weights = values(first.input![1]).slice(4 * 6, 5 * 6)
  const bias = values(first.input![2])[4]
  expect(weights).toEqual([0, 0, 1, 0, 0, 0])
  await openCounter(page)
  const inspector = await inspectWeights(page)
  await inspector.getByRole('combobox', { name: 'Neuron', exact: true }).selectOption('0:4')
  for (const vx of [13, -13]) {
    if (vx < 0) await page.getByRole('button', { name: 'Toward you', exact: true }).click()
    const input = observation({ ...serve(1), ...recoveryFrame, vx }, 0)
    const rows = inspector.getByRole('table', { name: 'Neuron weight contributions' }).locator('tbody tr')
    await expect(rows).toHaveCount(6)
    for (let index = 0; index < input.length; index++) {
      const cells = rows.nth(index).getByRole('cell')
      await expect(cells).toHaveCount(3)
      const expected = [input[index], weights[index], input[index] * weights[index]]
      for (let column = 0; column < expected.length; column++) {
        await expect.poll(async () => Number(await cells.nth(column).textContent())).toBeCloseTo(expected[column], 5)
      }
    }
    const sum = input.reduce((total, value, index) => total + value * weights[index], bias)
    await expect(inspector.getByLabel('Neuron bias', { exact: true })).toHaveText(bias.toFixed(6))
    await expect(inspector.getByLabel('Neuron weighted sum', { exact: true })).toHaveText(sum.toFixed(6))
    await expect(inspector.getByLabel('Neuron activation', { exact: true })).toHaveText(Math.max(0, sum).toFixed(6))
  }
})

for (const width of [390, 320]) test(`the lesson works with keyboard and reduced motion at ${width}px`, async ({ page }) => {
  await page.setViewportSize({ width, height: 844 })
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await openWeights(page)
  await expect(page.getByRole('img', { name: /Follow chooses Right/ })).toBeVisible({ timeout: 30_000 })
  await expect(page.getByRole('button', { name: 'Animate', exact: true })).toBeDisabled()
  const counter = page.getByRole('button', { name: '2 · Know when to return', exact: true })
  await counter.focus()
  await counter.press('Enter')
  await expect(counter).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByLabel('Counter chosen move', { exact: true })).toHaveText('Left')
  const mute = page.getByRole('button', { name: /^(Switch off|Restore) the departure neuron$/ })
  await mute.focus()
  await mute.press('Enter')
  await expect(mute).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByLabel('Muted detector chosen move', { exact: true })).toHaveText('Right')
  const toward = page.getByRole('button', { name: 'Toward you', exact: true })
  await toward.focus()
  await toward.press('Enter')
  await expect(page.getByText('No change this time.', { exact: true })).toBeVisible()
  for (const name of ['Counter', 'Muted detector']) {
    const card = page.getByRole('article', { name: `${name} decision`, exact: true })
    await card.scrollIntoViewIfNeeded()
    const bounds = await card.boundingBox()
    expect(bounds).not.toBeNull()
    expect(bounds!.x).toBeGreaterThanOrEqual(0)
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(width)
  }
  await page.getByRole('button', { name: 'Reset', exact: true }).click()
  await expect(page.getByLabel('Counter chosen move', { exact: true })).toHaveText('Left')
  await expect(page.getByRole('article', { name: 'Muted detector decision', exact: true })).toHaveCount(0)
  const role = page.getByRole('button', { name: 'Inspect H1.5: Outgoing detector', exact: true })
  await role.focus()
  await role.press('Enter')
  const inspector = await inspectWeights(page)
  const neuron = inspector.getByRole('combobox', { name: 'Neuron', exact: true })
  await expect(neuron).toHaveValue('0:4')
  await neuron.selectOption('3:2')
  await expect(inspector).toContainText('Linear output')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
})

import { test, expect, type Locator, type Page } from '@playwright/test'
import { readFileSync } from 'node:fs'
import pb from 'onnx-proto'
import * as ort from 'onnxruntime-web/wasm'
import { choice, observation, serve, type Frame, type Scores } from '../src/game'

const moves = ['Left', 'Stay', 'Right']
type LabFrame = Pick<Frame, 'x' | 'y' | 'vx' | 'vy' | 'paddles'>
// Physical situations are independent of the production explanation and presets.
const wallFrame: LabFrame = { x: 240, y: 80, vx: -13, vy: 3, paddles: [100, 80] }
const recoveryFrame: LabFrame = { x: 184, y: 120, vx: 13, vy: 0, paddles: [100, 80] }
const aimFrame: LabFrame = { x: 16, y: 80, vx: -13, vy: 0, paddles: [80, 80] }
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
    .getByRole('button', { name: 'Weights', exact: true }).click()
  await expect(page.getByRole('group', { name: 'Choose a weight explainer' })).toBeVisible()
}

async function openCounter(page: Page) {
  await openWeights(page)
  await page.getByRole('button', { name: 'Building behavior · Counter97', exact: true }).click()
  await expect(page.getByLabel('Counter97 chosen move', { exact: true })).toBeVisible({ timeout: 30_000 })
}

async function experiment(page: Page, name: 'Detect a wall' | 'Switch behavior' | 'Aim a return') {
  await page.getByRole('group', { name: 'Choose an experiment' }).getByRole('button', { name: new RegExp(name) }).click()
  await expect(page.getByLabel('Counter97 chosen move', { exact: true })).toBeVisible()
}

async function inspectWeights(page: Page) {
  const summary = page.locator('summary').filter({ hasText: 'Inspect weights and formulas' })
  const details = summary.locator('..')
  if (!await details.evaluate(element => (element as HTMLDetailsElement).open)) await summary.click()
  const inspector = page.getByRole('region', { name: 'Counter97 weight inspection', exact: true })
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
  const card = page.getByRole('article', { name: 'Counter97 decision', exact: true })
  await expect(card.getByLabel('Counter97 chosen move', { exact: true })).toHaveText(moves[choice(scores)])
  // The logits encode signed target error; keep the displayed target unclamped.
  const target = frame.paddles[0] + 80 * (scores[2] - scores[0]) / 2
  await expect(card.getByLabel('Counter97 paddle target', { exact: true })).toHaveText(target.toFixed(2))
  await inspectWeights(page)
  for (let action = 0; action < moves.length; action++) {
    await expect.poll(async () => Number(await page.getByLabel(`Counter97 ${moves[action]} score`, { exact: true }).textContent()))
      .toBeCloseTo(scores[action], 5)
  }
  return scores
}

test('two explainers lead from Follow’s shared neuron to Counter97’s three experiments', async ({ page }) => {
  await openWeights(page)
  const picker = page.getByRole('group', { name: 'Choose a weight explainer' })
  await expect(picker.getByRole('button')).toHaveCount(2)
  await expect(picker.getByRole('button', { name: 'Basics · Follow', exact: true })).toHaveAttribute('aria-pressed', 'true')
  await expect(picker.getByRole('button', { name: /Attack75/ })).toHaveCount(0)
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
  await page.getByRole('button', { name: 'Play against Follow', exact: true }).click()
  await expect(page.getByRole('combobox', { name: 'Opponent', exact: true })).toHaveValue('follow')
})

test('the wall experiment crosses a real neuron threshold while preserving the original ONNX decisions', async ({ page }) => {
  await openCounter(page)
  const experiments = page.getByRole('group', { name: 'Choose an experiment' })
  await expect(experiments.getByRole('button')).toHaveCount(3)
  await expect(experiments.getByRole('button', { name: /Detect a wall/ })).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByRole('combobox', { name: 'Neuron', exact: true })).toBeHidden()
  await expect(page.getByLabel('Wall neuron activation', { exact: true })).toHaveText('0.000000')
  await expect(page.getByRole('slider', { name: 'Ball position', exact: true })).toHaveValue('80')
  const inspector = await inspectWeights(page)
  await inspector.getByRole('combobox', { name: 'Neuron', exact: true }).selectOption('2:1')
  const session = await ort.InferenceSession.create(readFileSync('public/models/counter97.onnx'), { executionProviders: ['wasm'] })
  try {
    for (const y of [40, 150]) {
      await setRange(page.getByRole('slider', { name: 'Ball position', exact: true }), y)
      await expectOriginalDecision(page, session, { ...wallFrame, y })
      const activation = inspector.getByLabel('Neuron activation', { exact: true })
      if (y === 40) await expect(activation).toHaveText('0.000000')
      else await expect.poll(async () => Number(await activation.textContent())).toBeGreaterThan(0)
    }
  } finally { await session.release() }
  await expect(page.getByRole('link', { name: 'Download Counter97', exact: true }))
    .toHaveAttribute('href', /\/models\/counter97\.onnx$/)
  await expect(page.getByRole('link', { name: 'Read Counter97’s constructor', exact: true }))
    .toHaveAttribute('download', 'counter97.py')
})

test('muting the outgoing detector changes recovery but leaves incoming decisions and the original model intact', async ({ page }) => {
  await openCounter(page)
  await experiment(page, 'Switch behavior')
  const session = await ort.InferenceSession.create(readFileSync('public/models/counter97.onnx'), { executionProviders: ['wasm'] })
  try {
    const original = await expectOriginalDecision(page, session, recoveryFrame)
    await expect(page.getByLabel('Counter97 paddle target', { exact: true })).toHaveText('80.00')
    const toggle = page.getByRole('button', { name: 'Mute outgoing detector', exact: true })
    await toggle.click()
    await expect(toggle).toHaveAttribute('aria-pressed', 'true')
    const muted = page.getByRole('article', { name: 'Muted detector decision', exact: true })
    await expect(muted).toBeVisible()
    await expect(muted.getByLabel('Muted detector chosen move', { exact: true })).toHaveText('Right')
    await expect.poll(async () => Number(await muted.getByLabel('Muted detector paddle target', { exact: true }).textContent())).not.toBe(80)
    await expectOriginalDecision(page, session, recoveryFrame)
    await page.getByRole('button', { name: 'Reverse ball direction', exact: true }).click()
    const incoming = await expectOriginalDecision(page, session, { ...recoveryFrame, vx: -13 })
    expect(incoming).not.toEqual(original)
    await expect(muted.getByLabel('Muted detector chosen move', { exact: true })).toHaveText(moves[choice(incoming)])
    const incomingTarget = recoveryFrame.paddles[0] + 80 * (incoming[2] - incoming[0]) / 2
    await expect(muted.getByLabel('Muted detector paddle target', { exact: true })).toHaveText(incomingTarget.toFixed(2))
    await toggle.click()
    await expect(toggle).toHaveAttribute('aria-pressed', 'false')
    await expect(muted).toHaveCount(0)
  } finally { await session.release() }
  await page.getByRole('button', { name: 'Play against Counter97', exact: true }).click()
  await expect(page.getByRole('combobox', { name: 'Opponent', exact: true })).toHaveValue('counter97')
})

test('aiming responds to the opponent and switching experiments restores each focused starting state', async ({ page }) => {
  await openCounter(page)
  await experiment(page, 'Aim a return')
  const opponent = page.getByRole('slider', { name: 'Opponent position', exact: true })
  const session = await ort.InferenceSession.create(readFileSync('public/models/counter97.onnx'), { executionProviders: ['wasm'] })
  try {
    for (const [position, move] of [[60, 'Left'], [100, 'Right']] as const) {
      await setRange(opponent, position)
      await expectOriginalDecision(page, session, { ...aimFrame, paddles: [80, position] })
      await expect(page.getByLabel('Counter97 chosen move', { exact: true })).toHaveText(move)
    }
    await experiment(page, 'Switch behavior')
    await page.getByRole('button', { name: 'Mute outgoing detector', exact: true }).click()
    await page.getByRole('button', { name: 'Reverse ball direction', exact: true }).click()
    await experiment(page, 'Aim a return')
    await expect(opponent).toHaveValue('80')
    await expect(page.getByRole('article', { name: 'Muted detector decision', exact: true })).toHaveCount(0)
    await expectOriginalDecision(page, session, aimFrame)
    await experiment(page, 'Switch behavior')
    await expect(page.getByRole('button', { name: 'Mute outgoing detector', exact: true })).toHaveAttribute('aria-pressed', 'false')
    await expectOriginalDecision(page, session, recoveryFrame)
    await experiment(page, 'Detect a wall')
    await expect(page.getByRole('slider', { name: 'Ball position', exact: true })).toHaveValue('80')
  } finally { await session.release() }
})

test('advanced neuron contributions expose the real direction weight and ReLU cutoff', async ({ page }) => {
  const graph = pb.onnx.ModelProto.decode(readFileSync('public/models/counter97.onnx')).graph!
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
  await experiment(page, 'Switch behavior')
  const inspector = await inspectWeights(page)
  await inspector.getByRole('combobox', { name: 'Neuron', exact: true }).selectOption('0:4')
  for (const vx of [13, -13]) {
    if (vx < 0) await page.getByRole('button', { name: 'Reverse ball direction', exact: true }).click()
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

test('mobile and reduced-motion users can run all experiments using labelled keyboard controls', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await openWeights(page)
  await expect(page.getByRole('img', { name: /Follow chooses Right/ })).toBeVisible({ timeout: 30_000 })
  await expect(page.getByRole('button', { name: 'Animate', exact: true })).toBeDisabled()
  const counter = page.getByRole('button', { name: 'Building behavior · Counter97', exact: true })
  await counter.focus()
  await counter.press('Enter')
  await expect(counter).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByLabel('Counter97 chosen move', { exact: true })).toBeVisible()
  const ball = page.getByRole('slider', { name: 'Ball position', exact: true })
  await ball.focus()
  await ball.press('ArrowRight')
  await expect(ball).toHaveValue('81')
  const recovery = page.getByRole('group', { name: 'Choose an experiment' }).getByRole('button', { name: /Switch behavior/ })
  await recovery.focus()
  await recovery.press('Enter')
  const mute = page.getByRole('button', { name: 'Mute outgoing detector', exact: true })
  await mute.focus()
  await mute.press('Enter')
  await expect(mute).toHaveAttribute('aria-pressed', 'true')
  const inspector = await inspectWeights(page)
  const neuron = inspector.getByRole('combobox', { name: 'Neuron', exact: true })
  await neuron.focus()
  await expect(neuron).toBeFocused()
  await neuron.selectOption('3:2')
  await expect(inspector).toContainText('Linear output')
  for (const name of ['Counter97', 'Muted detector']) {
    const card = page.getByRole('article', { name: `${name} decision`, exact: true })
    await card.scrollIntoViewIfNeeded()
    const bounds = await card.boundingBox()
    expect(bounds).not.toBeNull()
    expect(bounds!.x).toBeGreaterThanOrEqual(0)
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(390)
  }
  await experiment(page, 'Aim a return')
  const opponent = page.getByRole('slider', { name: 'Opponent position', exact: true })
  await opponent.focus()
  await opponent.press('ArrowRight')
  await expect(opponent).toHaveValue('81')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
})

import { test, expect, type Locator, type Page } from '@playwright/test'
import { readFileSync } from 'node:fs'
import pb from 'onnx-proto'
import * as ort from 'onnxruntime-web/wasm'
import { choice, observation, serve, type Frame, type Scores } from '../src/game'

const names = { attack75: 'Attack75', counter97: 'Counter97' } as const
const moves = ['Left', 'Stay', 'Right']
// Keep these physical situations independent of the production explanation.
const situations: Record<string, Pick<Frame, 'x' | 'y' | 'vx' | 'vy' | 'paddles'>> = {
  outgoing: { x: 184, y: 120, vx: 13, vy: 0, paddles: [100, 80] },
  long: { x: 240, y: 80, vx: -13, vy: 3, paddles: [130, 80] },
  bounce: { x: 240, y: 130, vx: -13, vy: 3, paddles: [140, 80] },
  corner: { x: 16, y: 80, vx: -13, vy: 0, paddles: [80, 80] },
}
const browserErrors = new WeakMap<Page, string[]>()

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

async function selectModel(page: Page, name: 'Attack75' | 'Counter97') {
  await page.getByRole('group', { name: 'Choose a weight explainer' })
    .getByRole('button', { name: `${name} · ${name === 'Attack75' ? 75 : 97} parameters`, exact: true }).click()
  await expect(page.getByLabel(`${name} chosen move`, { exact: true })).toBeVisible({ timeout: 30_000 })
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

test('Follow remains the default explainer with its live shared neuron and playable model', async ({ page }) => {
  await openWeights(page)
  await expect(page.getByRole('button', { name: 'Follow · 13 parameters', exact: true }))
    .toHaveAttribute('aria-pressed', 'true')
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

test('both model cards expose the original ONNX decisions for recovery, anticipation, reflection and aiming', async ({ page }) => {
  await openWeights(page)
  await selectModel(page, 'Attack75')
  ort.env.wasm.numThreads = 1
  for (const id of ['attack75', 'counter97'] as const) {
    const session = await ort.InferenceSession.create(readFileSync(`public/models/${id}.onnx`), { executionProviders: ['wasm'] })
    try {
      for (const [situation, frame] of Object.entries(situations)) {
        await page.getByRole('combobox', { name: 'Situation', exact: true }).selectOption(situation)
        const result = await session.run({ observation: new ort.Tensor('float32', observation({ ...serve(1), ...frame }, 0), [1, 6]) })
        const scores = Array.from(result.action_scores.data as Float32Array) as Scores
        const card = page.getByRole('article', { name: `${names[id]} decision`, exact: true })
        await expect(card.getByLabel(`${names[id]} chosen move`, { exact: true })).toHaveText(moves[choice(scores)])
        // The linear logits encode signed target error; do not clamp the reported target to paddle reach.
        const target = frame.paddles[0] + 80 * (scores[2] - scores[0]) / 2
        await expect(card.getByLabel(`${names[id]} paddle target`, { exact: true })).toHaveText(target.toFixed(2))
        for (let action = 0; action < moves.length; action++) {
          const output = card.getByLabel(`${names[id]} ${moves[action]} score`, { exact: true })
          expect(Number(await output.textContent())).toBeCloseTo(scores[action], 5)
        }
      }
    } finally { await session.release() }
  }
  for (const [id, name] of Object.entries(names)) {
    await selectModel(page, name)
    await expect(page.getByRole('region', { name: `${name} weight inspection`, exact: true })).toBeVisible()
    await expect(page.getByRole('link', { name: `Download ${name}`, exact: true }))
      .toHaveAttribute('href', new RegExp(`/models/${id}\\.onnx$`))
    await expect(page.getByRole('link', { name: `Read ${name}’s constructor`, exact: true }))
      .toHaveAttribute('download', `${id}.py`)
  }
  await page.getByRole('group', { name: 'Decision stages' }).getByRole('button', { name: /Move$/ }).click()
  await expect(page.getByRole('combobox', { name: 'Neuron', exact: true })).toHaveValue('3:2')
  await selectModel(page, 'Attack75')
  await expect(page.getByRole('combobox', { name: 'Neuron', exact: true })).toHaveValue('2:2')
  await expect(page.getByLabel('Attack75 chosen move', { exact: true })).toHaveText('Stay')
  await selectModel(page, 'Counter97')
  await page.getByRole('button', { name: 'Play against Counter97', exact: true }).click()
  await expect(page.getByRole('combobox', { name: 'Opponent', exact: true })).toHaveValue('counter97')
})

test('moving the opponent reverses the corner choice while Counter97 keeps outgoing recovery centered', async ({ page }) => {
  await openWeights(page)
  await selectModel(page, 'Attack75')
  const situation = page.getByRole('combobox', { name: 'Situation', exact: true })
  const opponent = page.getByRole('slider', { name: 'Opponent position', exact: true })
  await situation.selectOption('corner')
  for (const [position, move] of [[60, 'Left'], [100, 'Right']] as const) {
    await setRange(opponent, position)
    for (const name of Object.values(names)) {
      await expect(page.getByLabel(`${name} chosen move`, { exact: true })).toHaveText(move)
    }
  }
  await situation.selectOption('outgoing')
  await expect(opponent).toHaveValue('80')
  await expect(page.getByLabel('Attack75 chosen move', { exact: true })).toHaveText('Right')
  await expect(page.getByLabel('Counter97 chosen move', { exact: true })).toHaveText('Left')
  await setRange(page.getByRole('slider', { name: 'Your paddle position', exact: true }), 80)
  for (const position of [15, 145]) {
    await setRange(opponent, position)
    await expect(page.getByLabel('Counter97 chosen move', { exact: true })).toHaveText('Stay')
    await expect(page.getByLabel('Counter97 paddle target', { exact: true })).toHaveText('80.00')
    await expect(page.getByLabel('Attack75 chosen move', { exact: true })).toHaveText('Right')
  }
})

test('neuron contributions reveal the stored direction weight and the incoming ReLU cutoff', async ({ page }) => {
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
  await openWeights(page)
  await selectModel(page, 'Counter97')
  for (const situation of ['outgoing', 'corner']) {
    await page.getByRole('combobox', { name: 'Situation', exact: true }).selectOption(situation)
    const inspector = page.getByRole('region', { name: 'Counter97 weight inspection', exact: true })
    await inspector.getByRole('combobox', { name: 'Neuron', exact: true }).selectOption('0:4')
    const input = observation({ ...serve(1), ...situations[situation] }, 0)
    const rows = inspector.getByRole('table', { name: 'Neuron weight contributions' }).locator('tbody tr')
    await expect(rows).toHaveCount(6)
    for (let index = 0; index < input.length; index++) {
      const cells = rows.nth(index).getByRole('cell')
      await expect(cells).toHaveCount(3)
      const expected = [input[index], weights[index], input[index] * weights[index]]
      for (let column = 0; column < expected.length; column++) {
        expect(Number(await cells.nth(column).textContent())).toBeCloseTo(expected[column], 5)
      }
    }
    const sum = input.reduce((total, value, index) => total + value * weights[index], bias)
    await expect(inspector.getByLabel('Neuron bias', { exact: true })).toHaveText(bias.toFixed(6))
    await expect(inspector.getByLabel('Neuron weighted sum', { exact: true })).toHaveText(sum.toFixed(6))
    await expect(inspector.getByLabel('Neuron activation', { exact: true })).toHaveText(Math.max(0, sum).toFixed(6))
  }
})

test('mobile and reduced-motion users can inspect every model using labelled keyboard controls', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await openWeights(page)
  await expect(page.getByRole('img', { name: /Follow chooses Right/ })).toBeVisible({ timeout: 30_000 })
  await expect(page.getByRole('button', { name: 'Animate', exact: true })).toBeDisabled()
  const counter = page.getByRole('button', { name: 'Counter97 · 97 parameters', exact: true })
  await counter.focus()
  await expect(counter).toBeFocused()
  await counter.press('Enter')
  await expect(counter).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByLabel('Counter97 chosen move', { exact: true })).toBeVisible()
  const opponent = page.getByRole('slider', { name: 'Opponent position', exact: true })
  await opponent.focus()
  await opponent.press('ArrowRight')
  await expect(opponent).toHaveValue('81')
  const stages = page.getByRole('group', { name: 'Decision stages' })
  await stages.getByRole('button', { name: /Recovery$/ }).focus()
  await stages.getByRole('button', { name: /Recovery$/ }).press('Enter')
  await expect(stages.getByRole('button', { name: /Recovery$/ })).toHaveAttribute('aria-pressed', 'true')
  const neuron = page.getByRole('combobox', { name: 'Neuron', exact: true })
  await neuron.focus()
  await expect(neuron).toBeFocused()
  await page.keyboard.press('r')
  await page.keyboard.press('Tab')
  await expect(neuron).toHaveValue('3:2')
  await expect(page.getByRole('region', { name: 'Counter97 weight inspection' })).toContainText('Linear output')
  for (const name of Object.values(names)) {
    const card = page.getByRole('article', { name: `${name} decision`, exact: true })
    await card.scrollIntoViewIfNeeded()
    const bounds = await card.boundingBox()
    expect(bounds).not.toBeNull()
    expect(bounds!.x).toBeGreaterThanOrEqual(0)
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(390)
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
})

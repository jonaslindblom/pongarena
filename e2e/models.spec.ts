import { test, expect, type Page } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { presets } from '../src/roster'
import { validateArenaModel } from '../src/validate'
import { encode, network } from '../tests/modelFactory'

const modelErrors = new WeakMap<Page, string[]>()

test.beforeEach(async ({ page }) => {
  const errors: string[] = []
  modelErrors.set(page, errors)
  page.on('pageerror', error => errors.push(error.message))
})

test.afterEach(async ({ page }) => {
  expect(modelErrors.get(page), 'the browser should not have uncaught runtime errors').toEqual([])
})

export async function openModels(page: Page) {
  await page.goto('/')
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('button', { name: 'Models', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Model library', exact: true })).toBeVisible()
  await expect(page.locator('article.library-card')).toHaveCount(presets.length, { timeout: 30_000 })
  await expect(page.getByRole('alert')).toHaveCount(0)
}

test('bundled cards show their actual ONNX layouts, counts and linked submitters', async ({ page }) => {
  await openModels(page)
  for (const model of presets) {
    const bytes = readFileSync(`public/models/${model.id}.onnx`)
    const stats = validateArenaModel(bytes)
    const card = page.getByRole('article', { name: `${model.name} model`, exact: true })
    await expect(card.getByLabel(`${model.name} layer layout`)).toHaveText(`Layers ${stats.architecture.widths.join(' → ')}`)
    const submitter = card.getByRole('link', { name: `@${model.submittedBy}`, exact: true })
    await expect(submitter).toBeVisible()
    await expect(submitter).toHaveAttribute('href', `https://github.com/${model.submittedBy}`)
    await expect(submitter).toHaveAttribute('rel', /noopener/)
    await expect(submitter).toHaveAttribute('rel', /noreferrer/)
    await card.getByText('About this model', { exact: true }).click()
    await expect(card.getByText('Parameters', { exact: true }).locator('xpath=following-sibling::dd[1]')).toHaveText(stats.params.toLocaleString('en-US'))
    await expect(card.getByText('MACs / tick', { exact: true }).locator('xpath=following-sibling::dd[1]')).toHaveText(stats.macs.toLocaleString('en-US'))
    await card.getByText('Inspect ONNX graph', { exact: true }).click()
    await expect(card.getByRole('list', { name: `${model.name} ONNX operators` }).locator('li')).toHaveCount(stats.nodes)
  }
})

test('submitter links are keyboard accessible and open the submitted user profile', async ({ page, context }) => {
  // Exercise the actual new-tab link while keeping the test independent of GitHub.
  await context.route('https://github.com/**', route => route.fulfill({
    contentType: 'text/html', body: '<title>GitHub profile</title><h1>Submitted user profile</h1>',
  }))
  await openModels(page)
  const model = presets[0]
  const link = page.getByRole('article', { name: `${model.name} model`, exact: true })
    .getByRole('link', { name: `@${model.submittedBy}`, exact: true })
  await link.focus()
  await expect(link).toBeFocused()
  const popupPromise = page.waitForEvent('popup')
  await link.press('Enter')
  const popup = await popupPromise
  await popup.waitForLoadState('domcontentloaded')
  await expect(popup).toHaveURL(`https://github.com/${model.submittedBy}`)
  await expect(popup.getByRole('heading', { name: 'Submitted user profile' })).toBeVisible()
  await popup.close()
})

test('card play and watch actions run the selected weights in the arena', async ({ page }) => {
  await openModels(page)
  const model = presets.find(model => model.id === 'quill')!
  await page.getByRole('article', { name: `${model.name} model`, exact: true })
    .getByRole('button', { name: 'Play against', exact: true }).click()
  await expect(page.getByRole('combobox', { name: 'Opponent', exact: true })).toHaveValue(model.id)
  await expect(page.getByRole('button', { name: 'You play', exact: true })).toHaveAttribute('aria-pressed', 'true')
  await expect.poll(async () => Number((await page.locator('.tick-readout').getAttribute('aria-label'))?.replace('Simulation tick ', ''))).toBeGreaterThan(3)
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('button', { name: 'Models', exact: true }).click()
  await page.getByRole('article', { name: `${model.name} model`, exact: true })
    .getByRole('button', { name: 'Watch', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Watch models', exact: true })).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByRole('combobox', { name: 'Bottom controller', exact: true })).toHaveValue(model.id)
  await expect.poll(async () => Number((await page.locator('.tick-readout').getAttribute('aria-label'))?.replace('Simulation tick ', ''))).toBeGreaterThan(3)
  await expect(page.getByRole('alert')).toHaveCount(0)
})

test('model download returns the exact bundled ONNX bytes', async ({ page }) => {
  await openModels(page)
  const model = presets[0]
  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('article', { name: `${model.name} model`, exact: true })
    .getByRole('link', { name: `Download ${model.name}`, exact: true }).click()
  const download = await downloadPromise
  expect(download.suggestedFilename()).toBe(`${model.id}.onnx`)
  const stream = await download.createReadStream()
  const chunks: Buffer[] = []
  for await (const chunk of stream) chunks.push(Buffer.from(chunk))
  expect(Buffer.concat(chunks)).toEqual(readFileSync(`public/models/${model.id}.onnx`))
})

test('invalid input shape is rejected and a local import persists without a submitter claim', async ({ page }) => {
  await openModels(page)
  await page.getByRole('button', { name: 'Import model', exact: true }).click()
  const invalid = network([5, 4, 3])
  invalid.graph!.input![0].type!.tensorType!.shape!.dim![1].dimValue = 5
  await page.getByLabel('ONNX model file').setInputFiles({
    name: 'wrong-input.onnx', mimeType: 'application/octet-stream', buffer: Buffer.from(encode(invalid)),
  })
  await expect(page.getByRole('dialog').getByRole('alert')).toContainText('observation as float32[1, 6]')
  await expect(page.locator('article.library-card')).toHaveCount(presets.length)
  const bytes = readFileSync('public/models/quill.onnx')
  await page.getByLabel('ONNX model file').setInputFiles({
    name: 'My local controller.onnx', mimeType: 'application/octet-stream', buffer: bytes,
  })
  await expect(page.getByRole('dialog')).not.toBeVisible()
  await expect(page.getByRole('combobox', { name: 'Opponent', exact: true }))
    .toHaveValue(`local-${createHash('sha256').update(bytes).digest('hex')}`)
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('button', { name: 'Models', exact: true }).click()
  const local = page.getByRole('article', { name: 'My local controller model', exact: true })
  await expect(local).toBeVisible()
  await expect(local.getByText('Imported model', { exact: true })).toBeVisible()
  await expect(local.getByRole('link', { name: /^@/ })).toHaveCount(0)
  await expect(local.getByLabel('My local controller layer layout')).toHaveText('Layers 6 → 12 → 4 → 3 → 3')
  await page.reload()
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('button', { name: 'Models', exact: true }).click()
  await expect(local).toBeVisible({ timeout: 30_000 })
  await expect(local.getByRole('link', { name: /^@/ })).toHaveCount(0)
})

test('model cells and attribution remain usable on a narrow mobile screen', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await openModels(page)
  for (const model of presets) {
    const card = page.getByRole('article', { name: `${model.name} model`, exact: true })
    await card.scrollIntoViewIfNeeded()
    await expect(card.getByRole('link', { name: `@${model.submittedBy}`, exact: true })).toBeVisible()
    const bounds = await card.boundingBox()
    expect(bounds).not.toBeNull()
    expect(bounds!.x).toBeGreaterThanOrEqual(0)
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(390)
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
})

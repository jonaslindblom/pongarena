import { test, expect } from '@playwright/test'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'

test('Predict75 is attributed, downloadable and playable as a bundled controller', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('/')
  const models = page.getByRole('navigation', { name: 'Main navigation' }).getByRole('button', { name: 'Models', exact: true })
  await models.click()
  const card = page.getByRole('article', { name: 'Predict75 model', exact: true })
  await expect(card).toBeVisible({ timeout: 30_000 })
  await expect(card.getByLabel('Predict75 layer layout')).toHaveText('Layers 6 → 6 → 3 → 3')
  await expect(card.getByRole('link', { name: '@jonaslindblom', exact: true }))
    .toHaveAttribute('href', 'https://github.com/jonaslindblom')
  await expect(card.getByText('75 params', { exact: true })).toBeVisible()

  const downloading = page.waitForEvent('download')
  await card.getByRole('link', { name: 'Download Predict75', exact: true }).click()
  const download = await downloading
  expect(download.suggestedFilename()).toBe('predict75.onnx')
  const chunks: Buffer[] = []
  for await (const chunk of await download.createReadStream()) chunks.push(Buffer.from(chunk))
  const downloaded = Buffer.concat(chunks)
  expect(downloaded).toEqual(readFileSync('public/models/predict75.onnx'))
  expect(createHash('sha256').update(downloaded).digest('hex'))
    .toBe('0d405fe22a0a5f568c1d23d76356893e12aae1ef1c9a7f26150e8f35520b3cad')

  await card.getByRole('button', { name: 'Play against', exact: true }).click()
  await expect(page.getByRole('combobox', { name: 'Opponent', exact: true })).toHaveValue('predict75')
  await expect.poll(async () => Number((await page.locator('.tick-readout').getAttribute('aria-label'))
    ?.replace('Simulation tick ', ''))).toBeGreaterThan(15)
  await models.click()
  await card.getByRole('button', { name: 'Watch', exact: true }).click()
  await expect(page.getByRole('combobox', { name: 'Bottom controller', exact: true })).toHaveValue('predict75')
  await expect.poll(async () => Number((await page.locator('.tick-readout').getAttribute('aria-label'))
    ?.replace('Simulation tick ', ''))).toBeGreaterThan(15)
  await expect(page.getByRole('alert')).toHaveCount(0)
  expect(errors).toEqual([])
})

import { test, expect } from '@playwright/test'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { validateArenaModel } from '../src/validate'

for (const [id, name, hash] of [
  ['predict', 'Predict', '2253a9dc4b461771f94e6592109c3a2113aa773059e3f46ab8bb1230f5db2bf6'],
  ['attack', 'Attack', '32eaa54eec39f754b6bbd16395d1895c236b5be8692d620e71b82c12c74e55f1'],
  ['counter', 'Counter', 'db7809ba7dc5f8f5eb035c2863b8c1353c11780d2cf0004e6166ee916a5213ed'],
]) test(`${name} is attributed, downloadable and playable as a bundled controller`, async ({ page }) => {
  const bytes = readFileSync(`public/models/${id}.onnx`)
  const stats = validateArenaModel(bytes)
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('/')
  const models = page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Models', exact: true })
  await models.click()
  const card = page.getByRole('article', { name: `${name} model`, exact: true })
  await expect(card).toBeVisible({ timeout: 30_000 })
  await expect(card.getByLabel(`${name} layer layout`)).toHaveText(`Layers ${stats.architecture.widths.join(' → ')}`)
  await expect(card.getByRole('link', { name: '@jonaslindblom', exact: true }))
    .toHaveAttribute('href', 'https://github.com/jonaslindblom')
  await expect(card.getByText(`${stats.params} params`, { exact: true })).toBeVisible()

  const downloading = page.waitForEvent('download')
  await card.getByRole('link', { name: `Download ${name}`, exact: true }).click()
  const download = await downloading
  expect(download.suggestedFilename()).toBe(`${name.toLowerCase()}.onnx`)
  const chunks: Buffer[] = []
  for await (const chunk of await download.createReadStream()) chunks.push(Buffer.from(chunk))
  const downloaded = Buffer.concat(chunks)
  expect(downloaded).toEqual(bytes)
  expect(createHash('sha256').update(downloaded).digest('hex'))
    .toBe(hash)

  await card.getByRole('button', { name: 'Play against', exact: true }).click()
  await expect(page.getByRole('combobox', { name: 'Opponent', exact: true })).toHaveValue(id)
  await expect.poll(async () => Number((await page.locator('.tick-readout').getAttribute('aria-label'))
    ?.replace('Simulation tick ', ''))).toBeGreaterThan(15)
  await models.click()
  await card.getByRole('button', { name: 'Watch', exact: true }).click()
  await expect(page.getByRole('combobox', { name: 'Bottom controller', exact: true })).toHaveValue(id)
  await expect.poll(async () => Number((await page.locator('.tick-readout').getAttribute('aria-label'))
    ?.replace('Simulation tick ', ''))).toBeGreaterThan(15)
  await expect(page.getByRole('alert')).toHaveCount(0)
  expect(errors).toEqual([])
})

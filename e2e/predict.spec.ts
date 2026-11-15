import { test, expect } from '@playwright/test'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { validateArenaModel } from '../src/validate'

for (const [id, name, hash] of [
  ['predict', 'Predict', 'a9411c30b370b90f0e419d62b592e5365c94126835b7706907f30faacbd9103f'],
  ['attack', 'Attack', '798c396e31ddc54e269f1f9e614317b829f0944ae3692976a561a705b07a0a59'],
  ['counter', 'Counter', 'f522f0bab90f9b269cfa93aaadf79f59c47bcca922fcbb89961cd441400e975d'],
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

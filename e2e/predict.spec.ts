import { test, expect } from '@playwright/test'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { validateArenaModel } from '../src/validate'

for (const [id, name, hash] of [
  ['predict', 'Predict', '0d405fe22a0a5f568c1d23d76356893e12aae1ef1c9a7f26150e8f35520b3cad'],
  ['attack', 'Attack', '646d1a1365e3e307bd4431fc73930c5efa19de16b7cc22bbb9200608c2b720f8'],
  ['counter', 'Counter', '5f7a480b98c982cf403f2f8849fac1e5e02503ff12e6b0b5d1e377856534e35c'],
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

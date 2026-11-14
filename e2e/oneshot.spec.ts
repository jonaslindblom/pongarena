import { test, expect } from '@playwright/test'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const experiment = 'experiments/one-prompt-weights'
const followUp = experiment + '/follow-up'
const opponents = ['follow', 'small', 'predict', 'attack', 'counter', 'quill', 'large']

test('OneShot plays the frozen experiment weights and shows the recorded prompt comparison', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 844 })
  await page.goto('/#models')
  const catalog = page.getByRole('region', { name: 'Prompt experiments models' })
  const card = catalog.getByRole('article', { name: 'OneShot model', exact: true })
  await expect(card.getByText('163 params', { exact: true })).toBeVisible({ timeout: 30_000 })
  await expect(card.getByRole('img', { name: /Reference league rank \d/ })).toBeVisible()

  const downloading = page.waitForEvent('download')
  await card.getByRole('link', { name: 'Download OneShot', exact: true }).click()
  const download = await downloading
  expect(download.suggestedFilename()).toBe('oneshot.onnx')
  const chunks: Buffer[] = []
  for await (const chunk of await download.createReadStream()) chunks.push(Buffer.from(chunk))
  // Prove the downloaded policy is still the frozen numeric reply, without
  // keeping a second compiled copy of the catalog model in the experiment.
  const directory = mkdtempSync(join(tmpdir(), 'oneshot-replay-'))
  try {
    const compiled = join(directory, 'tactical-163-a.onnx')
    execFileSync(process.execPath, ['--import', 'tsx', experiment + '/evaluate.mjs',
      'compile', '6,12,4,3,3', 'Relu', followUp + '/attempts/tactical-163-a.json', compiled])
    expect(Buffer.concat(chunks)).toEqual(readFileSync(compiled))
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }

  await card.getByRole('link', { name: 'See the prompt' }).click()
  await expect(page).toHaveURL(/#how-it-thinks\/oneshot$/)
  const oneShot = page.getByLabel('OneShot chosen move', { exact: true })
  const rulesOnly = page.getByLabel('Rules only chosen move', { exact: true })
  await expect(oneShot).toHaveText('Left')
  await expect(rulesOnly).toHaveText('Left')
  await page.getByRole('button', { name: 'Opponent right', exact: true }).click()
  await expect(oneShot).toHaveText('Right')
  await expect(rulesOnly).toHaveText('Left')

  const table = page.getByRole('table', { name: 'OneShot experiment results' })
  const rows = table.locator('tbody tr')
  const attempts = ['rules-163-a', 'rules-163-b', 'tactical-163-a', 'tactical-163-b']
  await expect(rows).toHaveCount(attempts.length)
  for (const [index, attempt] of attempts.entries()) {
    const { matchups } = JSON.parse(readFileSync(followUp + '/results/' + attempt + '.json', 'utf8'))
    const points = opponents.map(id => matchups.find((row: { opponent: { id: string } }) => row.opponent.id === id).record.leaguePoints + ' / 64')
    const rallies = matchups.find((row: { opponent: { id: string } }) => row.opponent.id === 'large').record.rallyPointsFor
    await expect(rows.nth(index).getByRole('cell')).toHaveText(points)
    if (attempt === 'tactical-163-a') {
      await expect(page.getByText(rallies + ' of 640 individual rallies', { exact: true })).toBeVisible()
    }
  }

  const rules = readFileSync(experiment + '/prompt.txt', 'utf8').replace('{{WIDTHS}}', '6 → 12 → 4 → 3 → 3')
  const tactical = rules + '\n' + readFileSync(followUp + '/tactical-brief.txt', 'utf8')
  for (const [label, prompt] of [['Rules-only prompt', rules], ['OneShot prompt · rules + tactical brief', tactical]]) {
    const summary = page.locator('summary').filter({ hasText: label })
    await summary.click()
    const content = summary.locator('..').locator('pre')
    await expect(content).toBeVisible()
    expect(await content.textContent()).toBe(prompt)
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)

  await page.getByRole('button', { name: 'Play against OneShot', exact: true }).click()
  await expect(page.getByRole('combobox', { name: 'Opponent', exact: true })).toHaveValue('oneshot')
  await expect(page).toHaveURL(/model=oneshot(?:[&#]|$)/)
  await page.reload()
  await expect(page.getByRole('combobox', { name: 'Opponent', exact: true })).toHaveValue('oneshot', { timeout: 30_000 })
  await page.getByRole('navigation').getByRole('link', { name: 'League', exact: true }).click()
  const reference = page.getByRole('region', { name: 'Reference league', exact: true })
  const row = reference.getByRole('row').filter({ hasText: 'OneShot' })
  await expect(row).toContainText('Prompted')
  await expect(row.getByRole('button', { name: 'Play against OneShot', exact: true })).toBeEnabled()
})

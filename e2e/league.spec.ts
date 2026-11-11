import { test, expect } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { benchmarkIdentity, type CompetitionRun } from '../src/competition'

test('a League model link opens its catalog card and can be shared', async ({ page }) => {
  await page.goto('/#league')
  const reference = page.getByRole('region', { name: 'Reference league', exact: true })
  const modelLink = reference.getByRole('link', { name: 'OneShot', exact: true })
  await expect(modelLink).toHaveAttribute('href', '#models/oneshot')
  await modelLink.click()

  const card = page.getByRole('article', { name: 'OneShot model', exact: true })
  await expect(page).toHaveURL(/#models\/oneshot$/)
  await expect(card).toBeFocused()
  await expect(card).toBeInViewport()

  await page.reload()
  await expect(card).toBeFocused()
  await expect(card).toBeInViewport()

  await page.goBack()
  await expect(page).toHaveURL(/#league$/)
  await expect(modelLink).toBeVisible()
})

for (const width of [390, 1280]) {
  test(`512-set standings show score uncertainty at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 })
    // Expand a real, complete development schedule into a synthetic 512-set
    // fixture, keeping each pair of side-swapped games intact.
    const source: CompetitionRun = JSON.parse(readFileSync('public/league.json', 'utf8'))
    const byPair = new Map<string, CompetitionRun['contests']>()
    for (const contest of source.contests) {
      const key = contest.a + contest.b
      if (!byPair.has(key)) byPair.set(key, [])
      byPair.get(key)!.push(contest)
    }
    const spec = { ...source.spec, seeds: Array.from({ length: 512 }, (_, index) => index) }
    const contests = [...byPair.values()].flatMap(pair => spec.seeds.map(seed =>
      ({ ...pair[seed % pair.length], seed })))
    const run: CompetitionRun = { ...source, spec, contests, benchmark: await benchmarkIdentity(spec),
      completedMatches: contests.length * 2, totalMatches: contests.length * 2 }
    await page.route('**/league.json', route => route.fulfill({ json: run }))
    const errors: string[] = []
    page.on('pageerror', error => errors.push(error.message))
    await page.goto('/')
    await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'League', exact: true }).click()
    const reference = page.getByRole('region', { name: 'Reference league' })
    await expect(reference.locator('tbody tr')).toHaveCount(spec.entrants.length)
    await expect(reference).toContainText('512 paired serve sets per opponent')
    await expect(reference).toContainText(`${run.totalMatches.toLocaleString('en-US')} games`)
    for (const entrant of spec.entrants) {
      await expect(page.getByLabel(entrant.name + ' score uncertainty', { exact: true }))
        .toHaveText(/95% CI \d+\.\d%–\d+\.\d%|No variation observed/)
    }
    await expect(reference.locator('.ranking-uncertainty-note')).toContainText('both sides and all opponents together')
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    await expect(page.getByRole('alert')).toHaveCount(0)
    expect(errors).toEqual([])
  })
}

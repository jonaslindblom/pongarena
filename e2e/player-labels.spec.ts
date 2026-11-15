import { test, expect, type Page } from '@playwright/test'

const clockStart = new Date('2026-01-01T12:00:00Z')
const browserErrors = new WeakMap<Page, string[]>()

test.beforeEach(async ({ page }) => {
  const errors: string[] = []
  browserErrors.set(page, errors)
  page.on('pageerror', error => errors.push(error.message))
})

test.afterEach(async ({ page }) => {
  expect(browserErrors.get(page), 'inspecting either paddle must not cause browser errors').toEqual([])
  await expect(page.getByRole('alert')).toHaveCount(0)
})

for (const match of [
  { opponent: 'follow', top: 'Follow', bottom: 'Counter',
    defaultImport: 'Player 2 · Bottom' },
  { opponent: 'keyboard', top: 'Counter', bottom: 'Human',
    defaultImport: 'Player 1 · Top' },
]) {
  test(`P1 stays top and P2 bottom with ${match.top} above ${match.bottom}`, async ({ page }) => {
    await page.clock.install({ time: clockStart })
    await page.goto(`/?mode=watch&model=counter&opponent=${match.opponent}`)
    await expect(page.locator('.court .status')).toHaveText(match.opponent === 'keyboard' ? 'Ready' : 'Live', { timeout: 30_000 })
    await page.clock.pauseAt(new Date(clockStart.getTime() + 60_000))

    // Expectations describe the visible court, independently of simulation indices.
    const picks = page.locator('.matchup .pick')
    await expect(picks.locator('.pick-label')).toHaveText([/PLAYER 01\s*\/\s*TOP/, /PLAYER 02\s*\/\s*BOTTOM/])
    await expect(picks.locator('.pick-name b')).toHaveText([match.top, match.bottom])
    await expect(picks.nth(0).getByRole('combobox')).toHaveAccessibleName('Top controller')
    await expect(picks.nth(1).getByRole('combobox')).toHaveAccessibleName('Bottom controller')
    // A linked human has a visible label but is not a selectable weight model.
    await expect(page.getByRole('combobox', { name: 'Top controller', exact: true }))
      .toHaveValue(match.opponent === 'keyboard' ? 'counter' : 'follow')
    await expect(page.getByRole('combobox', { name: 'Bottom controller', exact: true }))
      .toHaveValue(match.opponent === 'keyboard' ? 'keyboard' : 'counter')
    await expect(page.locator('.score-player small')).toHaveText(['P1 · TOP', 'P2 · BOTTOM'])
    await expect(page.locator('.score-player strong')).toHaveText([match.top, match.bottom])
    const points = (await page.locator('.big-score b').allTextContents()).map(Number)
    await expect(page.locator('.big-score')).toHaveAccessibleName(
      `Score: player 1, top, ${match.top}: ${points[0]}; player 2, bottom, ${match.bottom}: ${points[1]}`,
    )

    const brain = page.getByRole('region', { name: 'Model brain', exact: true })
    const topBrain = brain.getByRole('button', { name: 'Show brain for player 1', exact: true })
    const bottomBrain = brain.getByRole('button', { name: 'Show brain for player 2', exact: true })
    await expect(brain.locator('.segmented button')).toHaveText(['P1 · Top', 'P2 · Bottom'])
    await expect(topBrain).toHaveAttribute('aria-pressed', 'true')
    await expect(brain.locator('.brain-source span').first()).toHaveText(match.top)
    await bottomBrain.click()
    await expect(bottomBrain).toHaveAttribute('aria-pressed', 'true')
    if (match.bottom === 'Human') {
      await expect(brain.getByText('Human has no neural network.', { exact: false })).toBeVisible()
      await expect(brain.locator('.brain-source')).toHaveCount(0)
    } else {
      await expect(brain.locator('.brain-source span').first()).toHaveText(match.bottom)
    }

    await page.getByRole('navigation', { name: 'Main navigation' })
      .getByRole('link', { name: 'Models', exact: true }).click()
    await page.getByRole('button', { name: 'Import model', exact: true }).click()
    const dialog = page.getByRole('dialog')
    const targets = dialog.getByRole('group', { name: 'Assign to paddle', exact: true })
    await expect(targets.getByRole('button')).toHaveText(['Player 1 · Top', 'Player 2 · Bottom'])
    await expect(targets.getByRole('button', { name: match.defaultImport, exact: true }))
      .toHaveAttribute('aria-pressed', 'true')
    for (const name of ['Player 1 · Top', 'Player 2 · Bottom']) {
      await targets.getByRole('button', { name, exact: true }).click()
      await expect(targets.getByRole('button', { name, exact: true })).toHaveAttribute('aria-pressed', 'true')
    }
    await dialog.getByRole('button', { name: 'Close import', exact: true }).click()
  })
}

test('the brain keeps its layout while waiting for the first decision', async ({ page }) => {
  await page.goto('/?mode=watch&model=counter&opponent=keyboard')
  const brain = page.getByRole('region', { name: 'Model brain', exact: true })
  const graph = brain.getByRole('img', { name: 'Neural network layout; awaiting decision', exact: true })
  await expect(graph).toBeVisible({ timeout: 30_000 })
  await expect(brain.getByRole('button', { name: 'Show brain for player 1', exact: true }))
    .toHaveAttribute('aria-pressed', 'true')
  expect(await graph.locator('circle').count()).toBeGreaterThan(0)
  await expect(graph.locator('circle:not([fill-opacity="0.08"])')).toHaveCount(0)
  await expect(graph.locator('.brain-selected')).toHaveCount(0)
  await expect(brain.getByText('Start the match to see real neuron activations.')).toHaveCount(0)
})

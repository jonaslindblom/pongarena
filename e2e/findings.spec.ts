import { test, expect } from '@playwright/test'
import { findingsData } from '../src/findingsData'

test('findings connect frozen evidence to the current Small model and playable matchups', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/#findings')
  await expect(page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link')).toHaveText([
    'Play', 'How it thinks', 'League', 'Models', 'Findings',
  ])
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('The architecture can do it.Can training find the weights?')
  for (const opponent of ['quill', 'counter'] as const) {
    const name = opponent === 'quill' ? 'Quill' : 'Counter'
    await page.getByRole('button', { name: 'vs ' + name, exact: true }).click()
    const chart = page.getByRole('list', { name: 'Rally scores against ' + name })
    for (const result of findingsData.matchups.filter(row => row.opponent === opponent)) {
      const model = findingsData.models.find(model => model.id === result.model)!
      const row = chart.getByRole('listitem').filter({ has: page.getByText(model.name, { exact: true }) })
      await expect(row.locator('b')).toHaveText((result.rallyScore * 100).toFixed(1) + '%')
    }
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.getByRole('button', { name: 'Watch Small vs Counter', exact: true }).click()
  await expect(page.getByRole('combobox', { name: 'Bottom controller', exact: true })).toHaveValue('small')
  await expect(page.getByRole('combobox', { name: 'Top controller', exact: true })).toHaveValue('counter')
  await expect(page.locator('.court .status')).toHaveText('Live')
  await expect(page.locator('.court-top')).toBeInViewport()
  await page.getByRole('navigation').getByRole('link', { name: 'Models', exact: true }).click()
  const small = page.getByRole('article', { name: 'Small model', exact: true })
  await expect(small.getByText('163 params', { exact: true })).toBeVisible()
  await small.getByRole('link', { name: 'Training findings', exact: true }).click()
  await page.getByRole('button', { name: 'Play against Small', exact: true }).click()
  await expect(page.getByRole('combobox', { name: 'Opponent', exact: true })).toHaveValue('small')
  await expect(page.locator('.court-top')).toBeInViewport()
})

test('old lesson and guide links still reach useful content in the new structure', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/#how-it-thinks/oneshot')
  await expect(page.getByRole('heading', { name: 'One prompt. One Pong player.', exact: true })).toBeVisible()
  await expect(page.getByRole('navigation').getByRole('link', { name: 'Findings', exact: true })).toHaveAttribute('aria-current', 'page')
  await page.getByRole('link', { name: '1 · Training tiny networks', exact: true }).click()
  await expect(page).toHaveURL(/#findings$/)
  await page.goBack()
  await expect(page.getByRole('heading', { name: 'One prompt. One Pong player.', exact: true })).toBeVisible()
  await page.goto('/#guide')
  await expect(page.getByRole('heading', { name: 'You play from the bottom.', exact: true })).toBeVisible()
  await page.getByRole('navigation').getByRole('link', { name: 'How it thinks', exact: true }).click()
  await expect(page.getByRole('group', { name: 'Choose a lesson' }).getByRole('link')).toHaveCount(2)
  await page.getByRole('link', { name: 'Rules & controls', exact: true }).click()
  await expect(page).toHaveURL(/#rules$/)
  await expect(page.getByRole('heading', { name: 'Rules', exact: true })).toBeInViewport()
})

for (const width of [320, 390, 1280]) test(`the reading path connects mechanisms, prompts and training at ${width}px`, async ({ page }) => {
  await page.setViewportSize({ width, height: 844 })
  await page.goto('/#how-it-thinks')
  const lessons = page.getByRole('group', { name: 'Choose a lesson' })
  for (const title of ['1 · Follow the ball', '2 · Know when to return']) {
    await expect(lessons.getByRole('link', { name: title, exact: true })).toBeInViewport({ ratio: 1 })
    const bounds = await lessons.getByRole('link', { name: title, exact: true }).boundingBox()
    expect(bounds!.width).toBeGreaterThanOrEqual(44)
    expect(bounds!.height).toBeGreaterThanOrEqual(44)
  }
  await page.getByRole('link', { name: /NEXT LESSON/ }).click()
  await expect(page).toHaveURL(/#how-it-thinks\/counter$/)
  await expect(page.getByRole('heading', { level: 1 })).toBeInViewport()
  await page.getByRole('link', { name: /From a mechanism to a prompt/ }).click()
  await expect(page).toHaveURL(/#findings\/oneshot$/)
  await expect(page.getByRole('heading', { level: 1 })).toBeInViewport()
  const findings = page.getByRole('group', { name: 'Choose a finding' })
  await expect(findings.getByRole('link', { name: '2 · Prompting the weights', exact: true })).toHaveAttribute('aria-current', 'page')
  await expect(page.getByRole('heading', { name: 'Can a design idea become weights?', exact: true })).toBeVisible()
  await expect(page.getByRole('table', { name: 'Saved input probe results', exact: true })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.getByRole('link', { name: 'Counter’s recovery lesson', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Know when to stop chasing.', exact: true })).toBeInViewport()
  await page.goBack()
  await page.getByRole('link', { name: /ALSO IN FINDINGS/ }).click()
  await expect(page).toHaveURL(/#findings$/)
  await page.getByRole('link', { name: /NEXT FINDING/ }).click()
  await expect(page.getByRole('heading', { name: 'One prompt. One Pong player.', exact: true })).toBeInViewport()
})

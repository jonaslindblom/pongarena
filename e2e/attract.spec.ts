import { test, expect, type Page } from '@playwright/test'

const clockStart = new Date('2026-01-01T12:00:00Z')
const browserErrors = new WeakMap<Page, string[]>()

test.beforeEach(async ({ page }) => {
  const errors: string[] = []
  browserErrors.set(page, errors)
  page.on('pageerror', error => errors.push(error.message))
})

test.afterEach(async ({ page }) => {
  expect(browserErrors.get(page), 'demo and takeover must not cause uncaught browser errors').toEqual([])
  await expect(page.getByRole('alert')).toHaveCount(0)
})

async function openDemo(page: Page) {
  await page.clock.install({ time: clockStart })
  await page.goto('/')
  await expect(page.locator('.court .status')).toHaveText('Live', { timeout: 30_000 })
  await page.clock.pauseAt(new Date(clockStart.getTime() + 60_000))
}

async function finishDemo(page: Page) {
  // Run the real policies and physics. One-second steps leave time to inspect
  // the result before the automatic rematch's result display has elapsed.
  for (let elapsed = 0; elapsed < 300_000; elapsed += 1000) {
    await page.clock.runFor(1000)
    if (await page.locator('.match-over').isVisible()) return
  }
  await expect(page.locator('.match-over')).toBeVisible()
}

async function setHidden(page: Page, hidden: boolean) {
  await page.evaluate(value => {
    Object.defineProperty(document, 'hidden', { configurable: true, value })
    document.dispatchEvent(new Event('visibilitychange'))
  }, hidden)
}

test('a fresh landing plays Quill against Counter and stays a demo on reload', async ({ page }) => {
  await openDemo(page)
  await expect(page.getByRole('combobox', { name: 'Bottom controller' })).toHaveValue('quill')
  await expect(page.getByRole('combobox', { name: 'Top controller' })).toHaveValue('counter')
  await expect(page.getByRole('button', { name: 'Take over & play', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Pause', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Restart match', exact: true })).toBeVisible()
  await expect(page.locator('.serve-countdown')).toHaveCount(0)
  const tick = await page.locator('.tick-readout').getAttribute('aria-label')
  await page.clock.runFor(1000)
  await expect(page.locator('.tick-readout')).not.toHaveAttribute('aria-label', tick!)
  expect(new URL(page.url()).search).toBe('')
  await page.reload()
  await expect(page.locator('.court .status')).toHaveText('Live', { timeout: 30_000 })
  await expect(page.getByRole('combobox', { name: 'Bottom controller' })).toHaveValue('quill')
  await expect(page.getByRole('combobox', { name: 'Top controller' })).toHaveValue('counter')
})

test('reloading the watch URL saved after interaction starts the demo again', async ({ page }) => {
  await openDemo(page)
  await page.locator('.score-player strong').first().click()
  await expect(page).toHaveURL(/model=quill&mode=watch&opponent=counter/)
  await page.reload()
  await expect(page.locator('.court .status')).toHaveText('Live', { timeout: 30_000 })
  await expect(page.getByRole('combobox', { name: 'Bottom controller' })).toHaveValue('quill')
  await expect(page.getByRole('combobox', { name: 'Top controller' })).toHaveValue('counter')
  await expect(page.getByRole('button', { name: 'Pause', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Restart match', exact: true })).toBeVisible()
  const tick = await page.locator('.tick-readout').getAttribute('aria-label')
  await page.clock.runFor(1000)
  await expect(page.locator('.tick-readout')).not.toHaveAttribute('aria-label', tick!)
})

test('an untouched finished demo starts a new match with fresh serves', async ({ page }) => {
  test.slow()
  await page.addInitScript(() => {
    // Observe seed creation without exposing or modifying application state.
    const original = crypto.getRandomValues.bind(crypto)
    const seeds: number[] = []
    Object.assign(window, { matchSeeds: seeds })
    crypto.getRandomValues = array => {
      const result = original(array)
      if (array instanceof Uint32Array && array.length === 1) seeds.push(array[0])
      return result
    }
  })
  await openDemo(page)
  const seeds = () => page.evaluate(() => (window as unknown as { matchSeeds: number[] }).matchSeeds)
  const originalSeeds = await seeds()
  await finishDemo(page)
  await expect(page.locator('.court .status')).toHaveText('Game over')
  await page.clock.runFor(1500)
  await expect(page.locator('.match-over')).toBeVisible()
  await page.clock.runFor(2200)
  await expect(page.locator('.match-over')).toHaveCount(0)
  await expect(page.locator('.court .status')).toHaveText('Live')
  await expect(page.locator('.court-foot')).toContainText('RALLY 01')
  await expect(page.locator('.big-score')).toHaveAttribute('aria-label', 'Score 0 to 0')
  expect((await seeds()).length).toBeGreaterThan(originalSeeds.length)
})

test('interacting with the result cancels its pending automatic rematch', async ({ page }) => {
  test.slow()
  await openDemo(page)
  await finishDemo(page)
  await page.locator('.match-over h2').click()
  const score = await page.locator('.big-score').getAttribute('aria-label')
  const tick = await page.locator('.tick-readout').getAttribute('aria-label')
  await page.clock.runFor(6000)
  await expect(page.locator('.court .status')).toHaveText('Game over')
  await expect(page.locator('.big-score')).toHaveAttribute('aria-label', score!)
  await expect(page.locator('.tick-readout')).toHaveAttribute('aria-label', tick!)
})

test('keyboard takeover starts a fresh human match with time to get ready', async ({ page }) => {
  await openDemo(page)
  await page.clock.runFor(1000)
  const takeover = page.getByRole('button', { name: 'Take over & play', exact: true })
  await takeover.focus()
  await takeover.press('Enter')
  await expect(page.locator('.score-player strong').first()).toHaveText('Human')
  await expect(page.getByRole('combobox', { name: 'Opponent', exact: true })).toHaveValue('counter')
  await expect(page.locator('.big-score')).toHaveAttribute('aria-label', 'Score 0 to 0')
  await expect(page.locator('.tick-readout')).toHaveAttribute('aria-label', 'Simulation tick 0')
  await expect(page.locator('.serve-countdown strong')).toHaveText('3')
  await expect(takeover).toHaveCount(0)
  await page.clock.runFor(3100)
  await page.keyboard.down('ArrowLeft')
  await page.clock.runFor(250)
  await page.keyboard.up('ArrowLeft')
  await expect.poll(async () => Number(await page.locator('.paddle.left').getAttribute('y'))).toBeLessThan(68)
  await page.keyboard.press('p')
  const tick = await page.locator('.tick-readout').getAttribute('aria-label')
  await page.clock.runFor(6000)
  await expect(page.locator('.court .status')).toHaveText('Paused')
  await expect(page.locator('.tick-readout')).toHaveAttribute('aria-label', tick!)
})

test('a hidden untouched demo resumes, while a deliberate pause stays paused', async ({ page }) => {
  await openDemo(page)
  await page.clock.runFor(1000)
  await setHidden(page, true)
  await expect(page.locator('.court .status')).toHaveText('Paused')
  const tick = await page.locator('.tick-readout').getAttribute('aria-label')
  await page.clock.runFor(6000)
  await expect(page.locator('.tick-readout')).toHaveAttribute('aria-label', tick!)
  await setHidden(page, false)
  await expect(page.locator('.court .status')).toHaveText('Live')
  await page.clock.runFor(1000)
  await expect(page.locator('.tick-readout')).not.toHaveAttribute('aria-label', tick!)
  await page.keyboard.press('p')
  await expect(page.locator('.court .status')).toHaveText('Paused')
  const pausedTick = await page.locator('.tick-readout').getAttribute('aria-label')
  await setHidden(page, true)
  await setHidden(page, false)
  await page.clock.runFor(6000)
  await expect(page.locator('.court .status')).toHaveText('Paused')
  await expect(page.locator('.tick-readout')).toHaveAttribute('aria-label', pausedTick!)
})

test('choosing players leaves the viewer in control of when their match starts', async ({ page }) => {
  await openDemo(page)
  await page.getByRole('combobox', { name: 'Bottom controller' }).selectOption('follow')
  await page.clock.runFor(6000)
  await expect(page.locator('.court .status')).toHaveText('Ready')
  await expect(page.locator('.tick-readout')).toHaveAttribute('aria-label', 'Simulation tick 0')
  await expect(page.getByRole('combobox', { name: 'Bottom controller' })).toHaveValue('follow')
})

test.describe('mobile invitation', () => {
  test.use({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true })

  test('the invitation is reachable on mobile and one tap starts human play', async ({ page }) => {
    await openDemo(page)
    const takeover = page.getByRole('button', { name: 'Take over & play', exact: true })
    await takeover.scrollIntoViewIfNeeded()
    await expect(takeover).toBeInViewport({ ratio: 1 })
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    await takeover.tap()
    await expect(page.locator('.score-player strong').first()).toHaveText('Human')
    await expect(page.locator('.serve-countdown strong')).toHaveText('3')
    await expect(page.locator('.court')).toHaveClass(/human-controlled/)
  })
})

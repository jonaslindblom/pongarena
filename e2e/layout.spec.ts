import { test, expect, type Page } from '@playwright/test'

const clockStart = new Date('2026-01-01T12:00:00Z')
const browserErrors = new WeakMap<Page, string[]>()

test.beforeEach(async ({ page }) => {
  const errors: string[] = []
  browserErrors.set(page, errors)
  page.on('pageerror', error => errors.push(error.message))
})

test.afterEach(async ({ page }) => {
  expect(browserErrors.get(page), 'layout changes must not cause browser or resize-observer errors').toEqual([])
})

async function openHumanCourt(page: Page) {
  await page.clock.install({ time: clockStart })
  await page.goto('/?model=counter')
  await expect(page.locator('.court .status')).toHaveText('Ready', { timeout: 30_000 })
  await page.clock.pauseAt(new Date(clockStart.getTime() + 60_000))
}

async function fieldSize(page: Page) {
  const bounds = await page.locator('.field').boundingBox()
  expect(bounds).not.toBeNull()
  return { width: bounds!.width, height: bounds!.height }
}

async function expectFieldSize(page: Page, size: { width: number; height: number }) {
  await expect.poll(async () => Math.abs((await fieldSize(page)).height - size.height)).toBeLessThan(1)
  await expect.poll(async () => Math.abs((await fieldSize(page)).width - size.width)).toBeLessThan(1)
}

async function expectFullscreenCourt(page: Page) {
  const viewport = await page.evaluate(() => ({ width: innerWidth, height: innerHeight }))
  for (const selector of ['.court', '.field', '.paddle.side-0', '.paddle.side-1', '.court-control.play-btn', '.court-foot']) {
    const element = page.locator(selector)
    await expect(element, `${selector} must remain visible`).toBeVisible()
    const bounds = await element.boundingBox()
    expect(bounds, `${selector} must have screen bounds`).not.toBeNull()
    expect(bounds!.width, `${selector} must have positive width`).toBeGreaterThan(0)
    expect(bounds!.height, `${selector} must have positive height`).toBeGreaterThan(0)
    expect(bounds!.x, `${selector} must not extend past the viewport's left edge`).toBeGreaterThanOrEqual(-1)
    expect(bounds!.y, `${selector} must not extend above the viewport`).toBeGreaterThanOrEqual(-1)
    expect(bounds!.x + bounds!.width, `${selector} must fit within the viewport width`).toBeLessThanOrEqual(viewport.width + 1)
    expect(bounds!.y + bounds!.height, `${selector} must fit within the viewport height`).toBeLessThanOrEqual(viewport.height + 1)
  }
  await expect(page.locator('.field')).toBeInViewport({ ratio: 1 })
  const field = await fieldSize(page)
  expect(field.height).toBeGreaterThan(viewport.height / 2)
  expect(field.width).toBeGreaterThan(100)
}

async function checkFullscreenRoundTrip(page: Page) {
  const normalSize = await fieldSize(page)
  await page.getByRole('button', { name: 'Enter full screen', exact: true }).click()
  await expect(page.locator('.court-fullscreen')).toBeVisible()
  await expectFullscreenCourt(page)
  await page.getByRole('button', { name: 'Exit full screen', exact: true }).click()
  await expect(page.locator('.court-fullscreen')).toHaveCount(0)
  await expectFieldSize(page, normalSize)
}

for (const viewport of [{ width: 1280, height: 900 }, { width: 2239, height: 1414 }]) {
  test(`the centered desktop arena preserves its playing surface at ${viewport.width}×${viewport.height}`, async ({ page }) => {
    await page.setViewportSize(viewport)
    await openHumanCourt(page)
    const shell = await page.locator('.shell-arena').boundingBox()
    expect(shell).not.toBeNull()
    expect(Math.abs(shell!.x + shell!.width / 2 - viewport.width / 2)).toBeLessThan(1)
    expect(shell!.width).toBeGreaterThan(viewport.width / 2)
    const humanSize = await fieldSize(page)
    expect(humanSize.height).toBeGreaterThan(viewport.height / 2)
    expect(humanSize.width).toBeGreaterThan(300)
    await page.getByRole('button', { name: 'Play', exact: true }).click()
    await expect(page.locator('.serve-countdown strong')).toHaveText('3')
    await expectFieldSize(page, humanSize)
    await checkFullscreenRoundTrip(page)

    await page.goto('/')
    await expect(page.locator('.court .status')).toHaveText('Live', { timeout: 30_000 })
    // The invitation adds content around the existing court; it must not shrink
    // the playing surface or make it jump in size when a person takes over.
    await expectFieldSize(page, humanSize)
    await page.getByRole('button', { name: 'Take over & play', exact: true }).click()
    await expect(page.locator('.score-player strong').first()).toHaveText('Human')
    await expect(page.locator('.serve-countdown strong')).toHaveText('3')
    await expectFieldSize(page, humanSize)
    await checkFullscreenRoundTrip(page)
  })
}

test.describe('mobile court layout', () => {
  test.use({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true })

  test('mobile shows the court before player selection and brings it into view on takeover', async ({ page }) => {
    // Exercise the viewport fullscreen used on phones without the native API.
    await page.addInitScript(() => {
      Object.defineProperty(Element.prototype, 'requestFullscreen', { configurable: true, value: undefined })
    })
    await page.clock.install({ time: clockStart })
    await page.goto('/')
    await expect(page.locator('.court .status')).toHaveText('Live', { timeout: 30_000 })
    await page.clock.pauseAt(new Date(clockStart.getTime() + 60_000))
    const matchup = await page.locator('.matchup').boundingBox()
    const court = await page.locator('.court').boundingBox()
    expect(matchup).not.toBeNull()
    expect(court).not.toBeNull()
    expect(court!.y + court!.height).toBeLessThanOrEqual(matchup!.y)
    const originalSize = await fieldSize(page)
    const takeover = page.getByRole('button', { name: 'Take over & play', exact: true })
    // Visitors should see the game and invitation without scrolling past selectors.
    await expect(takeover).toBeInViewport({ ratio: 1 })
    await expect(page.locator('.field')).toBeInViewport({ ratio: 0.25 })
    await takeover.tap()
    await expect(page.locator('.score-player strong').first()).toHaveText('Human')
    await expect(page.locator('.serve-countdown strong')).toHaveText('3')
    await expect(page.locator('.court-top')).toBeInViewport({ ratio: 1 })
    await expect.poll(async () => Math.abs((await page.locator('.court').boundingBox())!.y)).toBeLessThan(2)
    await expectFieldSize(page, originalSize)
    await checkFullscreenRoundTrip(page)
  })
})

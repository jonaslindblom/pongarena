import { test, expect, type Page } from '@playwright/test'

test.use({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true })

async function openCourt(page: Page) {
  // Exercise the viewport fullscreen used by iPhones without native fullscreen.
  await page.addInitScript(() => {
    Object.defineProperty(Element.prototype, 'requestFullscreen', { configurable: true, value: undefined })
  })
  await page.goto('/')
  await expect(page.locator('.court-control.play-btn')).toBeEnabled({ timeout: 30_000 })
}

async function expectSelectionGuard(page: Page) {
  const court = page.getByRole('region', { name: 'Pong court' })
  for (const target of ['.score-player strong', '.court-modes button', '.court-foot span', '.field']) {
    const result = await court.locator(target).first().evaluate(element => ({
      userSelect: getComputedStyle(element).getPropertyValue('-webkit-user-select'),
      selectPrevented: !element.dispatchEvent(new Event('selectstart', { bubbles: true, cancelable: true })),
      menuPrevented: !element.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true })),
    }))
    expect(result).toEqual({ userSelect: 'none', selectPrevented: true, menuPrevented: true })
  }
}

test('court selection and long-press menus stay disabled while ready, playing and paused', async ({ page }) => {
  await openCourt(page)
  await expectSelectionGuard(page)
  await expect(page.locator('.court')).toHaveCSS('touch-action', 'auto')
  await page.locator('.court-control.play-btn').tap()
  await expect(page.locator('.court .status')).toHaveText('Live')
  await expectSelectionGuard(page)
  await page.locator('.court-control.play-btn').tap()
  await expect(page.locator('.court .status')).toHaveText('Paused')
  await expectSelectionGuard(page)
  await expect(page.locator('.court')).toHaveCSS('touch-action', 'auto')
  // The guard belongs to the game, not the surrounding page or its links.
  expect(await page.locator('.brand').evaluate(element =>
    element.dispatchEvent(new Event('selectstart', { bubbles: true, cancelable: true })))).toBe(true)
  expect(await page.locator('meta[name="viewport"]').getAttribute('content')).not.toMatch(/user-scalable|maximum-scale/)
})

test('holding and dragging steers without selecting text or blocking native control taps', async ({ page, context, browserName }) => {
  test.skip(browserName !== 'chromium', 'Real multi-touch input uses Chromium DevTools Protocol')
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()) })
  await openCourt(page)
  await page.locator('.court-control.play-btn').tap()
  await expect(page.locator('.court .status')).toHaveText('Live')
  await page.evaluate(() => {
    document.addEventListener('touchstart', event => {
      document.documentElement.dataset.lastTouchPrevented = String(event.defaultPrevented)
    })
  })
  const cdp = await context.newCDPSession(page)
  const label = await page.locator('.score-player strong').first().boundingBox()
  const court = await page.locator('.court').boundingBox()
  expect(label).not.toBeNull()
  expect(court).not.toBeNull()
  const point = { id: 1, x: label!.x + label!.width / 2, y: label!.y + label!.height / 2 }
  // A long press directly on a text label used to invite the iOS magnifier.
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [point] })
  await expect(page.locator('.touch-zone.left')).toHaveAttribute('data-held', 'true')
  await expect(page.locator('html')).toHaveAttribute('data-last-touch-prevented', 'true')
  await expect.poll(async () => Number(await page.locator('.paddle.left').getAttribute('y'))).toBeLessThan(55)
  await page.waitForTimeout(700)
  expect(await page.evaluate(() => getSelection()?.toString())).toBe('')
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ ...point, x: court!.x + court!.width - 20 }] })
  await expect(page.locator('.touch-zone.right')).toHaveAttribute('data-held', 'true')
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] })
  await expect(page.locator('.touch-zone[data-held]')).toHaveCount(0)
  await page.locator('.court-control.play-btn').tap()
  await expect(page.locator('.court .status')).toHaveText('Paused')
  await expect(page.locator('html')).toHaveAttribute('data-last-touch-prevented', 'false')
  await page.getByRole('button', { name: 'Restart match' }).tap()
  await expect(page.locator('.court .status')).toHaveText('Live')
  await page.getByRole('button', { name: 'Enter full screen' }).tap()
  await expect(page.locator('.court-fullscreen')).toBeVisible()
  await page.getByRole('button', { name: 'Exit full screen' }).tap()
  await expect(page.locator('.court-fullscreen')).toHaveCount(0)
  await page.locator('.court-control.play-btn').tap()
  await expect(page.locator('.court .status')).toHaveText('Paused')
  await page.getByRole('button', { name: 'Watch models', exact: true }).tap()
  await expectSelectionGuard(page)
  await expect(page.getByRole('alert')).toHaveCount(0)
  expect(errors).toEqual([])
})

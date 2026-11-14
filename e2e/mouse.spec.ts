import { test, expect, type Page } from '@playwright/test'

const clockStart = new Date('2026-01-01T12:00:00Z')
const browserErrors = new WeakMap<Page, string[]>()

test.beforeEach(async ({ page }) => {
  const errors: string[] = []
  browserErrors.set(page, errors)
  page.on('pageerror', error => errors.push(error.message))
})

test.afterEach(async ({ page }) => {
  expect(browserErrors.get(page), 'changing input methods must not cause browser errors').toEqual([])
  await expect(page.getByRole('alert')).toHaveCount(0)
})

async function openHumanCourt(page: Page) {
  await page.clock.install({ time: clockStart })
  await page.goto('/?model=counter')
  await expect(page.locator('.court .status')).toHaveText('Ready', { timeout: 30_000 })
  await page.clock.pauseAt(new Date(clockStart.getTime() + 60_000))
  await page.getByRole('button', { name: 'Play', exact: true }).click()
  await expect(page.locator('.serve-countdown strong')).toHaveText('3')
}

async function fieldPoint(page: Page, target: number) {
  const bounds = await page.locator('.field').boundingBox()
  expect(bounds).not.toBeNull()
  return { x: bounds!.x + bounds!.width * target / 160, y: bounds!.y + bounds!.height * 0.75 }
}

async function clickTarget(page: Page, target: number) {
  const point = await fieldPoint(page, target)
  await page.mouse.click(point.x, point.y)
}

async function moveToTarget(page: Page, target: number) {
  const point = await fieldPoint(page, target)
  await page.mouse.move(point.x, point.y)
}

async function paddleCenter(page: Page) {
  return Number(await page.locator('.paddle.left').getAttribute('y')) + 12
}

test('one click steers toward a stationary cursor at normal speed without holding the button', async ({ page }) => {
  await openHumanCourt(page)
  const hint = page.locator('.court-mouse-hint')
  await expect(hint).toContainText('Click to steer')
  await moveToTarget(page, 112)
  await expect(page.locator('.court-mouse-status')).toHaveCount(0)
  await clickTarget(page, 112)
  await expect(hint).toHaveText('Move left and right')
  await expect(page.locator('.court-mouse-status')).toHaveText('Mouse control on')
  // Clicking during the countdown arms steering but cannot teleport the paddle.
  expect(await paddleCenter(page)).toBe(80)
  await page.clock.runFor(3100)
  const earlyCenter = await paddleCenter(page)
  const tick = Number((await page.locator('.tick-readout').getAttribute('aria-label'))!.split(' ').at(-1))
  expect(earlyCenter).toBeGreaterThan(80)
  expect(earlyCenter).toBeLessThanOrEqual(80 + tick * 2)
  expect(earlyCenter).toBeLessThan(112)
  // No more pointer events: the target must still be reevaluated each tick.
  await page.clock.runFor(550)
  expect(Math.abs(await paddleCenter(page) - 112)).toBeLessThanOrEqual(1)
  await page.clock.runFor(100)
  expect(Math.abs(await paddleCenter(page) - 112)).toBeLessThanOrEqual(1)
  await moveToTarget(page, 96)
  await page.clock.runFor(300)
  expect(Math.abs(await paddleCenter(page) - 96)).toBeLessThanOrEqual(1)
  await expect(hint).toHaveCSS('opacity', '0')
})

test('steering stays active beyond both field edges and on re-entry without another click', async ({ page }) => {
  await openHumanCourt(page)
  await clickTarget(page, 80)
  await page.clock.runFor(3100)
  const bounds = (await page.locator('.court-surface').boundingBox())!
  // Pointer travel beyond either edge still asks for that edge, at normal speed.
  await page.mouse.move(bounds.x - 20, bounds.y + bounds.height / 2)
  await expect(page.locator('.court-mouse-status')).toHaveText('Mouse control on')
  await page.clock.runFor(150)
  const leftward = await paddleCenter(page)
  expect(leftward).toBeLessThan(80)
  expect(leftward).toBeGreaterThanOrEqual(68)
  await page.mouse.move(bounds.x + bounds.width + 20, bounds.y + bounds.height / 2)
  await expect(page.locator('.court-mouse-status')).toHaveText('Mouse control on')
  await page.clock.runFor(150)
  const rightward = await paddleCenter(page)
  expect(rightward).toBeGreaterThan(leftward)
  expect(rightward - leftward).toBeLessThanOrEqual(12)
  // Coming back over the field must not require a second activation click.
  await moveToTarget(page, 100)
  await page.clock.runFor(250)
  expect(await paddleCenter(page)).toBeGreaterThan(rightward)
  await expect(page.locator('.court-mouse-status')).toHaveText('Mouse control on')
})

test('an outside click releases steering while labels, controls and selectors remain usable', async ({ page }) => {
  await openHumanCourt(page)
  await clickTarget(page, 120)
  await page.clock.runFor(3300)
  // Clicking outside the field, even inside the card, intentionally releases it.
  await page.locator('.score-player strong').first().click()
  await expect(page.locator('.court-mouse-status')).toHaveCount(0)
  await expect(page.locator('.court-mouse-hint')).toContainText('Click to steer')
  await page.clock.runFor(50)
  const stopped = await paddleCenter(page)
  await moveToTarget(page, 140)
  await page.clock.runFor(200)
  expect(await paddleCenter(page)).toBe(stopped)
  await clickTarget(page, 100)
  await page.getByRole('button', { name: 'Pause', exact: true }).click()
  await expect(page.locator('.court .status')).toHaveText('Paused')
  await expect(page.locator('.court-mouse-status')).toHaveCount(0)
  await page.keyboard.press('p')
  await clickTarget(page, 100)
  const opponent = page.getByRole('combobox', { name: 'Opponent', exact: true })
  await opponent.click()
  await expect(page.locator('.court-mouse-status')).toHaveCount(0)
  await expect(opponent).toBeFocused()
  await page.keyboard.press('Escape')
  await opponent.selectOption('follow')
  await expect(opponent).toHaveValue('follow')
  await expect(page.locator('.court .status')).toHaveText('Ready')
})

test('keyboard movement takes over without snapping back to the old mouse target', async ({ page }) => {
  await openHumanCourt(page)
  // Clicking the field must release selector focus so movement keys steer the game.
  await page.getByRole('combobox', { name: 'Opponent', exact: true }).focus()
  await clickTarget(page, 100)
  await expect(page.locator('.court-surface')).toBeFocused()
  await page.clock.runFor(3450)
  const beforeKey = await paddleCenter(page)
  expect(Math.abs(beforeKey - 100)).toBeLessThanOrEqual(1)
  await page.keyboard.down('ArrowLeft')
  await expect(page.locator('.court-mouse-hint')).toBeHidden()
  await page.clock.runFor(150)
  await page.keyboard.up('ArrowLeft')
  await expect(page.locator('.court-mouse-status')).toHaveCount(0)
  await expect(page.locator('.court-mouse-hint')).toBeHidden()
  await page.clock.runFor(50)
  const stopped = await paddleCenter(page)
  expect(stopped).toBeLessThan(beforeKey)
  await page.clock.runFor(200)
  expect(await paddleCenter(page)).toBe(stopped)
})

for (const key of ['ArrowRight', 'a']) {
  test(`${key} dismisses the steering hint immediately without a prior mouse click`, async ({ page }) => {
    await openHumanCourt(page)
    const hint = page.locator('.court-mouse-hint')
    await expect(hint).toBeVisible()
    await page.keyboard.press('Shift')
    await expect(hint).toBeVisible()
    // Movement keys used in a selector are form input, not a choice of game controls.
    const opponent = page.getByRole('combobox', { name: 'Opponent', exact: true })
    await opponent.focus()
    await opponent.dispatchEvent('keydown', { key: 'ArrowRight', code: 'ArrowRight' })
    await expect(hint).toBeVisible()
    await page.locator('.court-surface').focus()
    await page.keyboard.down(key)
    await expect(hint).toBeHidden()
    await page.keyboard.up(key)
    await expect(hint).toBeHidden()
    await page.keyboard.press('p')
    await expect(page.locator('.court .status')).toHaveText('Paused')
    await page.keyboard.press('p')
    await expect(page.locator('.court .status')).toHaveText('Live')
    await expect(hint).toBeHidden()
    if (key === 'ArrowRight') {
      // Learning keyboard controls should survive the next serve countdown.
      for (let elapsed = 0; elapsed < 20_000; elapsed += 100) {
        await page.clock.runFor(100)
        if (!(await page.locator('.court-foot').textContent())?.includes('RALLY 01')) break
      }
      await expect(page.locator('.court-foot')).toContainText('RALLY 02')
      await expect(hint).toBeHidden()
    }
    await clickTarget(page, 100)
    await expect(hint).toBeVisible()
    await expect(hint).toHaveText('Move left and right')
  })
}

test('pause, restart, Escape and focus loss each require a new steering click', async ({ page }) => {
  await openHumanCourt(page)
  await clickTarget(page, 110)
  await page.keyboard.press('p')
  await expect(page.locator('.court .status')).toHaveText('Paused')
  await expect(page.locator('.court-mouse-status')).toHaveCount(0)
  await page.keyboard.press('p')
  await moveToTarget(page, 120)
  await expect(page.locator('.court-mouse-status')).toHaveCount(0)
  await clickTarget(page, 110)
  await page.keyboard.press('r')
  await expect(page.locator('.serve-countdown strong')).toHaveText('3')
  await expect(page.locator('.court-mouse-status')).toHaveCount(0)
  await clickTarget(page, 110)
  await page.keyboard.press('Escape')
  await expect(page.locator('.court-mouse-status')).toHaveCount(0)
  await expect(page.locator('.court .status')).toHaveText('Live')
  await clickTarget(page, 110)
  await page.evaluate(() => window.dispatchEvent(new Event('blur')))
  await expect(page.locator('.court .status')).toHaveText('Paused')
  await expect(page.locator('.court-mouse-status')).toHaveCount(0)
  await page.getByRole('button', { name: 'Watch models', exact: true }).click()
  await expect(page.locator('.court-mouse-hint')).toHaveCount(0)
  await expect(page.locator('.court-mouse-status')).toHaveCount(0)
})

test('model demos have no human steering hints or click-to-steer behavior', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('.court .status')).toHaveText('Live', { timeout: 30_000 })
  await expect(page.locator('.court-mouse-hint')).toHaveCount(0)
  await page.keyboard.press('ArrowLeft')
  await clickTarget(page, 120)
  await expect(page.locator('.score-player.right strong')).toHaveText('Quill')
  await expect(page.locator('.court-mouse-status')).toHaveCount(0)
  await page.getByRole('button', { name: 'Take over & play', exact: true }).click()
  await expect(page.locator('.court-mouse-hint')).toBeVisible()
})

test.describe('touch hints', () => {
  test.use({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true })

  test('phones retain hold-half hints without mouse instructions', async ({ page }) => {
    await page.goto('/?model=counter')
    await expect(page.getByRole('button', { name: 'Play', exact: true })).toBeEnabled({ timeout: 30_000 })
    await page.getByRole('button', { name: 'Play', exact: true }).tap()
    await expect(page.locator('.court .status')).toHaveText('Live')
    await expect(page.locator('.court-touch-hints')).toBeVisible()
    await expect(page.locator('.touch-zone.left')).toContainText('Hold left')
    await expect(page.locator('.court-mouse-hint')).toBeHidden()
    await expect(page.locator('.court-mouse-status')).toBeHidden()
  })
})

import { chromium, type Browser } from '@playwright/test'

let browser: Browser | undefined
try {
  browser = await chromium.launch({ executablePath: process.env.PONG_CHROMIUM_EXECUTABLE || undefined })
  const page = await browser.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1 })
  await page.goto(new URL('./social-card.html', import.meta.url).href)
  await page.evaluate(() => document.fonts.ready)
  await page.screenshot({ path: 'public/social-card.png', animations: 'disabled' })
  console.log('Updated public/social-card.png')
} finally {
  await browser?.close()
}

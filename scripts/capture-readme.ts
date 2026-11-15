import { chromium, type Browser } from '@playwright/test'
import { preview } from 'vite'

// A private preview and browser leave any developer's open app untouched.
const server = await preview({ preview: { host: '127.0.0.1', port: 0, open: false } })
let browser: Browser | undefined
try {
  browser = await chromium.launch({ executablePath: process.env.PONG_CHROMIUM_EXECUTABLE || undefined })
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 })
  await page.addInitScript(() => {
    const original = crypto.getRandomValues.bind(crypto)
    crypto.getRandomValues = values => {
      const result = original(values)
      // Fix only the match seed; the screenshot still runs the real policies.
      if (values instanceof Uint32Array && values.length === 1) values[0] = 1729
      return result
    }
  })
  const start = new Date('2026-01-01T12:00:00Z')
  await page.clock.install({ time: start })
  const url = server.resolvedUrls?.local[0]
  if (!url) throw new Error('The screenshot preview has no local URL.')
  await page.goto(url)
  await page.getByRole('button', { name: 'Take over & play', exact: true }).waitFor()
  await page.waitForFunction(() => document.querySelector('.court .status')?.textContent === 'Live')
  await page.clock.pauseAt(new Date(start.getTime() + 60_000))
  // Start from tick zero after loading, regardless of model initialization time.
  await page.getByRole('button', { name: 'Restart match', exact: true }).click()
  await page.mouse.move(0, 0)
  await page.clock.runFor(2000)
  await page.locator('.brain-graph').waitFor()
  await page.evaluate(() => document.fonts.ready)
  await page.screenshot({ path: 'docs/media/pongarena.png', fullPage: true, animations: 'disabled' })
  console.log('Updated docs/media/pongarena.png')
} finally {
  await browser?.close()
  await new Promise<void>((resolve, reject) => server.httpServer.close(error => error ? reject(error) : resolve()))
}

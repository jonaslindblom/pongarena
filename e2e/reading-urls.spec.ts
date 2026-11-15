import { test, expect } from '@playwright/test'

const pages = [
  { path: 'how-it-thinks/', title: 'How a 13-Parameter Pong Network Works — PongArena', heading: 'One neuron. Follow the ball.', text: 'Subtract two positions' },
  { path: 'how-it-thinks/counter/', title: 'Switch Off a Pong Neuron: Inside Counter — PongArena', heading: 'Know when to stop chasing.', text: 'A signal becomes a strategy.' },
  { path: 'findings/', title: 'Training vs. Programming Tiny Pong Networks — PongArena', heading: 'The architecture can do it.Can training find the weights?', text: 'More than one training run.' },
  { path: 'findings/oneshot/', title: 'One Prompt, 163 Parameters: OneShot Pong — PongArena', heading: 'One prompt. One Pong player.', text: 'Can a design idea become weights?' },
]

test.describe('indexable reading pages', () => {
  test.use({ javaScriptEnabled: false })

  for (const reading of pages) test(`${reading.path} serves its content and metadata without JavaScript`, async ({ page, request }) => {
    const response = await page.goto('/' + reading.path)
    expect(response?.status()).toBe(200)
    await expect(page).toHaveTitle(reading.title)
    await expect(page.getByRole('heading', { level: 1 })).toHaveText(reading.heading)
    await expect(page.getByRole('heading', { name: reading.text, exact: true })).toBeVisible()
    const canonical = 'https://jonaslindblom.github.io/pongarena/' + reading.path
    await expect(page.locator('link[rel="canonical"]')).toHaveAttribute('href', canonical)
    await expect(page.locator('meta[property="og:url"]')).toHaveAttribute('content', canonical)
    await expect(page.locator('meta[property="og:title"]')).toHaveAttribute('content', reading.title)
    await expect(page.locator('meta[name="twitter:title"]')).toHaveAttribute('content', reading.title)
    const description = await page.locator('meta[name="description"]').getAttribute('content')
    expect(description).toBeTruthy()
    await expect(page.locator('meta[property="og:description"]')).toHaveAttribute('content', description!)
    await expect(page.locator('meta[name="twitter:description"]')).toHaveAttribute('content', description!)
    for (const link of await page.locator('a[download]').all()) {
      const url = await link.evaluate(element => (element as HTMLAnchorElement).href)
      if (url.startsWith('data:')) continue
      const download = await request.get(url)
      expect(download.status(), url).toBe(200)
      expect(download.headers()['content-type'], url).not.toContain('text/html')
    }
  })

  test('static reading links resolve from a nested page and appear in the sitemap', async ({ page, request }) => {
    await page.goto('/how-it-thinks/counter/')
    await page.getByRole('link', { name: /From a mechanism to a prompt/ }).click()
    await expect(page).toHaveURL(/\/findings\/oneshot\/$/)
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('One prompt. One Pong player.')
    const sitemap = await request.get('/sitemap.xml')
    expect(sitemap.status()).toBe(200)
    for (const reading of pages)
      expect(await sitemap.text()).toContain('<loc>https://jonaslindblom.github.io/pongarena/' + reading.path + '</loc>')
  })

  test('generated links stay inside a GitHub Pages repository prefix', async ({ page }) => {
    await page.route('**/pongarena/**', async route => {
      const url = new URL(route.request().url())
      url.pathname = url.pathname.slice('/pongarena'.length)
      await route.fulfill({ response: await route.fetch({ url: url.href }) })
    })
    await page.goto('/pongarena/how-it-thinks/counter/')
    const links = await page.locator('a[href]').evaluateAll(elements => elements
      .map(element => new URL((element as HTMLAnchorElement).href))
      .filter(url => url.origin === window.location.origin)
      .map(url => url.pathname))
    expect(links.length).toBeGreaterThan(0)
    for (const path of links) expect(path).toMatch(/^\/pongarena\//)
    await page.getByRole('link', { name: /From a mechanism to a prompt/ }).click()
    await expect(page).toHaveURL(/\/pongarena\/findings\/oneshot\/$/)
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('One prompt. One Pong player.')
  })
})

test('old shared links migrate without losing queries and metadata follows navigation history', async ({ page }) => {
  await page.goto('/?source=linkedin#how-it-thinks/counter')
  await expect(page).toHaveURL(/\/how-it-thinks\/counter\/\?source=linkedin$/)
  await expect(page.getByLabel('Counter chosen move', { exact: true })).toBeVisible()
  await expect(page).toHaveTitle('Switch Off a Pong Neuron: Inside Counter — PongArena')
  await page.getByRole('link', { name: /From a mechanism to a prompt/ }).click()
  await expect(page).toHaveTitle('One Prompt, 163 Parameters: OneShot Pong — PongArena')
  await page.goBack()
  await expect(page).toHaveTitle('Switch Off a Pong Neuron: Inside Counter — PongArena')
  await page.reload()
  await expect(page.getByLabel('Counter chosen move', { exact: true })).toBeVisible()
  await page.getByRole('navigation').getByRole('link', { name: 'Play', exact: true }).click()
  await expect(page).toHaveTitle('PongArena — Play Pong Against Tiny Neural Networks')
  await expect(page.locator('link[rel="canonical"]')).toHaveAttribute('href', 'https://jonaslindblom.github.io/pongarena/')
  await expect(page.getByRole('alert')).toHaveCount(0)
})

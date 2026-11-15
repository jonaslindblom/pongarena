import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { createElement } from 'react'
import { renderToString } from 'react-dom/server'
import { build } from 'vite'
import { readingPages, siteUrl } from '../src/routes'

const template = await readFile('dist/index.html', 'utf8')
const escape = (text: string) => text.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
// Vite resolves the same CSS, raw text and model imports used by the browser app.
// Rendering the existing components keeps the indexed text and interactive pages in sync.
await mkdir('node_modules/.cache', { recursive: true })
const directory = await mkdtemp('node_modules/.cache/pongarena-prerender-')
try {
  await build({ build: { ssr: 'src/App.tsx', outDir: directory,
    rollupOptions: { output: { entryFileNames: 'app.mjs' } } } })
  const { default: App } = await import(pathToFileURL(resolve(directory, 'app.mjs')).href)
  for (const [path, metadata] of Object.entries(readingPages)) {
    const url = new URL(path, siteUrl)
    const content = renderToString(createElement(App, { location: url, base: new URL(siteUrl) }))
    const html = template
      .replace('<base href="./" />', `<base href="${'../'.repeat(path.split('/').length - 1)}" />`)
      .replace(/<title>[^<]*<\/title>/, `<title>${escape(metadata.title)}</title>`)
      .replace(/(<meta\s+(?:name|property)="(?:description|og:description|twitter:description)"\s+content=")[^"]*/g, (_, prefix) => prefix + escape(metadata.description))
      .replace(/(<meta (?:name|property)="(?:og:title|twitter:title)" content=")[^"]*/g, (_, prefix) => prefix + escape(metadata.title))
      .replace(/(<link rel="canonical" href=")[^"]*/, (_, prefix) => prefix + url.href)
      .replace(/(<meta property="og:url" content=")[^"]*/, (_, prefix) => prefix + url.href)
      .replace('<div id="root"></div>', () => `<div id="root">${content}</div>`)
    await mkdir('dist/' + path, { recursive: true })
    await writeFile('dist/' + path + 'index.html', html)
    console.log('Rendered ' + path)
  }
  const urls = ['', ...Object.keys(readingPages)].map(path => `<url><loc>${new URL(path, siteUrl).href}</loc></url>`)
  await writeFile('dist/sitemap.xml', '<?xml version="1.0" encoding="UTF-8"?>\n' +
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + urls.join('\n') + '\n</urlset>\n')
} finally {
  await rm(directory, { recursive: true, force: true })
}

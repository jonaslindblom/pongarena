import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { createHash } from 'node:crypto'
import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { pages } from './src/routes'

function sourceFiles(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name)
    return entry.isDirectory() ? sourceFiles(path) : [path]
  })
}
const sourceHash = createHash('sha256')
for (const path of [...sourceFiles('src'), 'index.html', 'package-lock.json', 'vite.config.ts'].sort())
  sourceHash.update(path).update('\0').update(readFileSync(path)).update('\0')

export default defineConfig({
  base: './',
  plugins: [react(), {
    name: 'page-base',
    transformIndexHtml: {
      order: 'pre',
      handler(html, context) {
        const pathname = (context.originalUrl ?? context.path).split('?')[0]
        const path = Object.keys(pages).filter(Boolean).find(path =>
          pathname.endsWith('/' + path) || pathname.endsWith('/' + path.slice(0, -1)))
        if (!path) return html
        const depth = path.split('/').length - (pathname.endsWith('/') ? 1 : 2)
        const base = '../'.repeat(depth) || './'
        return html.replace('<base href="./" />', `<base href="${base}" />`)
      },
    },
  }],
  define: { 'import.meta.env.VITE_SOURCE_ID': JSON.stringify(sourceHash.digest('hex').slice(0, 16)) },
})

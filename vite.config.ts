import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { createHash } from 'node:crypto'
import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'

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
  plugins: [react()],
  define: { 'import.meta.env.VITE_SOURCE_ID': JSON.stringify(sourceHash.digest('hex').slice(0, 16)) },
})

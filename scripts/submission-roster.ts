import { lstat, readFile } from 'node:fs/promises'
import ts from 'typescript'
import { isGitHubUsername } from '../src/submitter'

export interface SubmissionPreset {
  id: string
  name: string
  submittedBy: string
  kind?: string
  handcrafted?: boolean
  description?: string
}

/** Read submitted metadata as literal data; never import or execute PR code. */
export function parseSubmissionRoster(source: string): SubmissionPreset[] {
  if (Buffer.byteLength(source) > 128 * 1024) throw new Error('Submitted roster exceeds 128 KiB.')
  const file = ts.createSourceFile('roster.ts', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
  const fail = (): never => { throw new Error('Submit export const presets = [...] using only literal model metadata.') }
  if (file.statements.length !== 1) return fail()
  const statement = file.statements[0]
  if (!ts.isVariableStatement(statement) ||
    statement.modifiers?.length !== 1 || statement.modifiers[0].kind !== ts.SyntaxKind.ExportKeyword ||
    !(statement.declarationList.flags & ts.NodeFlags.Const) || statement.declarationList.declarations.length !== 1) return fail()
  const declaration = statement.declarationList.declarations[0]
  if (!ts.isIdentifier(declaration.name) || declaration.name.text !== 'presets' || !declaration.initializer ||
    !ts.isArrayLiteralExpression(declaration.initializer)) return fail()
  const ids = new Set<string>()
  return declaration.initializer.elements.map(element => {
    if (!ts.isObjectLiteralExpression(element)) return fail()
    const values: Record<string, string | boolean> = Object.create(null)
    for (const property of element.properties) {
      if (!ts.isPropertyAssignment(property) ||
        !(ts.isIdentifier(property.name) || ts.isStringLiteral(property.name))) return fail()
      const key = property.name.text
      if (!['id', 'name', 'submittedBy', 'kind', 'description', 'handcrafted'].includes(key) || key in values) return fail()
      const value = property.initializer
      if (key === 'handcrafted') {
        if (value.kind !== ts.SyntaxKind.TrueKeyword && value.kind !== ts.SyntaxKind.FalseKeyword) return fail()
        values[key] = value.kind === ts.SyntaxKind.TrueKeyword
      } else {
        if (!ts.isStringLiteral(value) && !ts.isNoSubstitutionTemplateLiteral(value)) return fail()
        values[key] = value.text
      }
    }
    if (typeof values.id !== 'string' || !/^[A-Za-z0-9_-]+$/.test(values.id) || ids.has(values.id) ||
      typeof values.name !== 'string' || !values.name.trim()) return fail()
    if (!isGitHubUsername(values.submittedBy))
      throw new Error('Each submitted model needs submittedBy as a valid GitHub username, not a URL.')
    ids.add(values.id)
    return { ...values } as unknown as SubmissionPreset
  })
}

export async function readSubmissionRoster(path: string | URL): Promise<SubmissionPreset[]> {
  const stats = await lstat(path)
  if (!stats.isFile() || stats.size > 128 * 1024) throw new Error('Submitted roster must be a regular file up to 128 KiB.')
  return parseSubmissionRoster(await readFile(path, 'utf8'))
}

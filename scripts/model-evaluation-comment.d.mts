export interface EvaluationReport {
  artifact: string
  pr: number
  runId: number
  headSha: string
  identity: string
}

type Request = (path: string, options?: { method: 'POST' | 'PATCH'; body: string }) => Promise<unknown>

export function selectEvaluation(request: Request, repository: string, id: unknown): Promise<EvaluationReport | undefined>
export function verifyProvenance(provenance: unknown, repository: string, report: EvaluationReport): void
export function standingsComment(value: unknown, report: EvaluationReport, runUrl: string): Promise<string>
export function upsertComment(request: Request, repository: string, report: EvaluationReport, body: string): Promise<'posted' | 'updated' | 'unchanged' | 'outdated'>

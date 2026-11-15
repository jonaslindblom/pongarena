import { useEffect, useState } from 'react'
import { assetPath } from './assets'
import { restoreCompetition, standings } from './competition'
import { PONG } from './rules'

export function useReferenceRanks(enabled: boolean) {
  const [requested, setRequested] = useState(false)
  const [result, setResult] = useState(() => ({ ranks: new Map<string, number>(), error: '', loading: true }))

  useEffect(() => { if (enabled) setRequested(true) }, [enabled])
  useEffect(() => {
    if (!requested) return
    const request = new AbortController()
    void (async () => {
      try {
        const response = await fetch(assetPath('league.json'), { signal: request.signal })
        if (!response.ok) throw new Error('Reference league ranks could not be loaded.')
        const run = await restoreCompetition(await response.json())
        if (run.spec.rulesId !== PONG.id || run.spec.contractId !== PONG.contractId || run.spec.division !== 'all')
          throw new Error('The reference league uses different game rules or classes.')
        const ranks = new Map<string, number>()
        for (const row of standings(run)) if (row.rank !== null) ranks.set(row.entrant.sha256, row.rank)
        if (!request.signal.aborted) setResult({ ranks, error: '', loading: false })
      } catch (error) {
        if (!request.signal.aborted) setResult({ ranks: new Map(), loading: false,
          error: error instanceof Error ? error.message : 'Reference league ranks could not be loaded.' })
      }
    })()
    return () => request.abort()
  }, [requested])

  return { ...result, loading: (enabled || requested) && result.loading }
}

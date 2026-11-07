import { ArrowDownToLine, Gamepad2, Play, RotateCcw, Shuffle, Square } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import {
  DIVISIONS, fitsCompetition, restoreCompetition, runCompetition, smallestDivision, standings,
  type CompetitionClass, type CompetitionEntrant, type CompetitionRun, type CompetitionSpec,
  type PlayCompetitionMatch,
} from './competition'
import { assetPath } from './assets'
import './Ranking.css'

const STORAGE_KEY = 'pongarena-last-league'
const numbers = (n: number) => n.toLocaleString()
function freshSeeds() {
  const seeds = new Set<number>()
  while (seeds.size < 16) seeds.add(crypto.getRandomValues(new Uint32Array(1))[0])
  return [...seeds]
}

function StandingsTable({ run, entrants, handcraftedHashes, onPlay }: {
  run: CompetitionRun
  entrants: CompetitionEntrant[]
  handcraftedHashes: readonly string[]
  onPlay: (id: string) => void
}) {
  return <div className="ranking-table-wrap"><table className="ranking-table">
    <thead><tr><th>Rank</th><th>Model</th><th>Points</th><th>W / D / L</th><th><span className="visually-hidden">Play</span></th></tr></thead>
    <tbody>{standings(run).map((row) => {
      const entrant = row.entrant
      const loaded = entrants.find((current) => current.sha256 === entrant.sha256)
      return <tr key={entrant.sha256}>
        <td>{row.rank ?? '—'}</td>
        <td><strong>{entrant.name}</strong><small>
          <span className="ranking-class">{smallestDivision(entrant)}</span>
          {' · '}{numbers(entrant.params)} params{handcraftedHashes.includes(entrant.sha256) ? ' · Handwritten' : ''}
        </small></td>
        <td>{numbers(row.leaguePoints)}<small>of {row.played}</small></td>
        <td>{row.wins} / {row.draws} / {row.losses}</td>
        <td><button className="secondary-btn ranking-play" disabled={!loaded} onClick={() => loaded && onPlay(loaded.id)} aria-label={'Play against ' + entrant.name}><Gamepad2 size={14} /><span>Play</span></button></td>
      </tr>
    })}</tbody>
  </table></div>
}

export function Ranking({ entrants, handcraftedHashes, rulesId, contractId, playMatch, onStart, onPlay }: {
  entrants: CompetitionEntrant[]
  handcraftedHashes: readonly string[]
  rulesId: string
  contractId: string
  playMatch: PlayCompetitionMatch
  onStart?: () => void
  onPlay: (id: string) => void
}) {
  const [reference, setReference] = useState<CompetitionRun | null>(null)
  const [referenceError, setReferenceError] = useState('')
  const [division, setDivision] = useState<CompetitionClass>('all')
  const [selected, setSelected] = useState<string[] | null>(null)
  const [seeds, setSeeds] = useState(freshSeeds)
  const [seedCount, setSeedCount] = useState(4)
  const [run, setRun] = useState<CompetitionRun | null>(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const abort = useRef<AbortController | null>(null)
  const mounted = useRef(true)
  const revision = useRef(0)
  const unique = entrants.filter((e, i) => entrants.findIndex((other) => other.sha256 === e.sha256) === i)
  const eligible = unique.filter((e) => fitsCompetition(e, division))
  const chosen = selected === null ? eligible : eligible.filter((e) => selected.includes(e.id))
  const limit = DIVISIONS.find((d) => d.id === division)
  const total = chosen.length * (chosen.length - 1) * seedCount
  const canRepeat = run?.spec.entrants.every((e) => unique.some((current) => current.sha256 === e.sha256)) ?? false

  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false; revision.current++; abort.current?.abort() }
  }, [])
  useEffect(() => {
    const request = new AbortController()
    setReference(null)
    setReferenceError('')
    void (async () => {
      try {
        const response = await fetch(assetPath('league.json'), { signal: request.signal })
        if (!response.ok) throw new Error('The reference league could not be loaded.')
        const saved = await response.json()
        if (saved?.spec?.rulesId !== rulesId || saved?.spec?.contractId !== contractId)
          throw new Error('The reference league uses different game rules.')
        const restored = await restoreCompetition(saved)
        if (!request.signal.aborted) setReference(restored)
      } catch (error) {
        if (!request.signal.aborted) setReferenceError(error instanceof Error ? error.message : 'The reference league could not be loaded.')
      }
    })()
    void (async () => {
      try {
        const value = localStorage.getItem(STORAGE_KEY)
        if (!value) return
        const saved = JSON.parse(value)
        if (saved?.spec?.rulesId !== rulesId || saved?.spec?.contractId !== contractId) return
        const restored = await restoreCompetition(saved)
        if (!request.signal.aborted) setRun(restored)
      } catch {
        if (!request.signal.aborted) setMessage('The previous local result could not be restored under the current league rules.')
      }
    })()
    return () => request.abort()
  }, [rulesId, contractId])

  async function start(repeat?: CompetitionSpec) {
    if (busy || abort.current) return
    onStart?.()
    const version = ++revision.current
    const controller = new AbortController()
    abort.current = controller
    setBusy(true)
    setMessage('')
    try {
      const spec = repeat ? { ...repeat, entrants: repeat.entrants.map((e) =>
        unique.find((current) => current.sha256 === e.sha256)!) } : {
        rulesId, contractId, division, seeds: seeds.slice(0, seedCount), entrants: chosen,
      }
      const result = await runCompetition({ ...spec, playMatch, signal: controller.signal,
        onProgress: (progress) => { if (mounted.current && revision.current === version) setRun(progress) },
      })
      if (result.status === 'complete' && mounted.current && revision.current === version) {
        try { localStorage.setItem(STORAGE_KEY, JSON.stringify(result)) }
        catch { setMessage('This browser could not save the result. Export it to keep a copy.') }
      }
    } catch (error) {
      if (mounted.current && revision.current === version) setMessage(error instanceof Error ? error.message : String(error))
    } finally {
      if (mounted.current && revision.current === version) setBusy(false)
      if (abort.current === controller) abort.current = null
    }
  }
  function toggle(id: string) {
    const ids = chosen.map((e) => e.id)
    setSelected(ids.includes(id) ? ids.filter((key) => key !== id) : [...ids, id])
  }
  function download() {
    if (!run) return
    const url = URL.createObjectURL(new Blob([JSON.stringify(run, null, 2) + '\n'], { type: 'application/json' }))
    const link = document.createElement('a')
    link.href = url
    link.download = 'pongarena-' + run.spec.division + '-' + run.benchmark.slice(0, 12) + '.json'
    link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  return <div className="ranking-page">
    <section className="panel ranking-results" aria-label="Reference league">
      <div className="ranking-result-heading"><h2>Reference standings</h2><span className="mini-tag">ALL MODELS</span></div>
      <p className="ranking-note">Bundled models, evaluated together on fixed serves. Pick an opponent and play.</p>
      {reference ? <>
        <StandingsTable run={reference} entrants={unique} handcraftedHashes={handcraftedHashes} onPlay={onPlay} />
        <p className="ranking-note">{numbers(reference.totalMatches)} games · {reference.spec.seeds.length} paired serve sets per opponent. A contest win earns 1 point; a tie earns ½. Equal points share rank.</p>
        <div className="ranking-actions">
          <a className="secondary-btn" href={assetPath('league.json')} download><ArrowDownToLine size={15} /> Results JSON</a>
          <a className="secondary-btn" href={assetPath('league.md')} download>League report</a>
        </div>
        <details className="ranking-seeds"><summary>Benchmark details</summary>
          <p>Generated from the repository’s models and rules. Model pull requests regenerate this snapshot and CI checks it. Scores apply to this field and serve set.</p>
          <code>{reference.spec.rulesId} · {reference.benchmark}</code>
          <code>{reference.spec.seeds.join(', ')}</code>
        </details>
      </> : <p className="ranking-note" role="status">{referenceError || 'Loading the reference standings…'}</p>}
    </section>
    <details className="panel ranking-config">
      <summary className="ranking-custom-title">Run your own league</summary>
      <p className="ranking-note">Compare imported models or choose a different field. Local results stay separate from the reference league.</p>
      <fieldset disabled={busy}>
        <div className="ranking-fields">
          <label>League<select value={division} onChange={(e) => { setDivision(e.target.value as CompetitionClass); setSelected(null) }}>
            <option value="all">All models</option>
            {DIVISIONS.map((d) => <option key={d.id} value={d.id}>{d.name} models</option>)}
          </select><small>{limit ? <>Up to {numbers(limit.maxParams)} parameters / {numbers(limit.maxMacs)} MACs</> : 'Every eligible model, ranked together.'}</small></label>
          <label>Serve sets per pairing<select value={seedCount} onChange={(e) => setSeedCount(Number(e.target.value))}>
            <option value={4}>4 · Quick</option><option value={8}>8 · Standard</option><option value={16}>16 · Broader</option>
          </select><small>Five rallies per game, then swap sides on the same serves.</small></label>
        </div>
        <div className="ranking-roster-heading"><h3>Choose the field</h3><span>{chosen.length} selected</span></div>
        <div className="ranking-roster">{eligible.map((e) => <label className="ranking-entry" key={e.sha256}>
          <input type="checkbox" checked={chosen.some((c) => c.id === e.id)} onChange={() => toggle(e.id)} aria-label={'Include ' + e.name} />
          <span><strong>{e.name}</strong><small>{`${smallestDivision(e)} · ${numbers(e.params)} params`}</small></span>
        </label>)}</div>
        {!eligible.length && <p className="ranking-note">No loaded models fit this league yet.</p>}
        <div className="ranking-actions">
          <button className="import-btn" disabled={chosen.length < 2} onClick={() => void start()}><Play size={16} /> Run league</button>
          <button className="secondary-btn" onClick={() => setSeeds(freshSeeds())}><Shuffle size={16} /> New serves</button>
          <span>{numbers(total)} games</span>
        </div>
      </fieldset>
      {busy && <button className="secondary-btn ranking-cancel" onClick={() => abort.current?.abort()}><Square size={15} /> Cancel league</button>}
      {message && <p className="message" role="status">{message}</p>}
      {run && <div className="ranking-local-results">
        <h3>{run.status === 'complete' ? 'Your local result' : run.status === 'running' ? 'League in progress' : 'Incomplete league'}</h3>
        <p className="ranking-note">{run.spec.division === 'all' ? 'All models' : run.spec.division} · {run.spec.seeds.length} paired serve sets</p>
        <div className="ranking-progress"><progress value={run.completedMatches} max={run.totalMatches} aria-label="Completed league games" /><span>{numbers(run.completedMatches)} / {numbers(run.totalMatches)} games</span></div>
        {run.status !== 'complete' && <p className="ranking-note">Completed side pairs contribute points. Ranks appear when every game finishes.</p>}
        {run.error && <p className="message error" role="alert">{run.error}</p>}
        <StandingsTable run={run} entrants={unique} handcraftedHashes={handcraftedHashes} onPlay={onPlay} />
        <div className="ranking-actions">
          <button className="secondary-btn" disabled={busy} onClick={download}><ArrowDownToLine size={16} /> Export</button>
          <button className="secondary-btn" disabled={busy || !canRepeat} onClick={() => void start(run.spec)}><RotateCcw size={16} /> Repeat benchmark</button>
        </div>
        {!canRepeat && <p className="ranking-note">Load the same model files to repeat this result.</p>}
        <details className="ranking-seeds"><summary>Benchmark details</summary><code>{run.benchmark}</code><code>{run.spec.seeds.join(', ')}</code></details>
      </div>}
    </details>
  </div>
}

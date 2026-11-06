import { ArrowDownToLine, Cpu, Play, RotateCcw, Shuffle, Square, Trophy } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import {
  DIVISIONS, fitsCompetition, restoreCompetition, runCompetition, smallestDivision, standings,
  type CompetitionClass, type CompetitionEntrant, type CompetitionRun, type CompetitionSpec,
  type PlayCompetitionMatch,
} from './competition'
import './Ranking.css'

// Keep the existing storage namespace so saved data survives the product rename.
const STORAGE_KEY = 'rally-lab-last-league-v1'
const numbers = (n: number) => n.toLocaleString()
function freshSeeds() {
  const seeds = new Set<number>()
  while (seeds.size < 16) seeds.add(crypto.getRandomValues(new Uint32Array(1))[0])
  return [...seeds]
}

export function Ranking({ entrants, handcraftedHashes, rulesId, contractId, playMatch, onStart }: {
  entrants: CompetitionEntrant[]
  handcraftedHashes: readonly string[]
  rulesId: string
  contractId: string
  playMatch: PlayCompetitionMatch
  onStart?: () => void
}) {
  const [division, setDivision] = useState<CompetitionClass>('standard')
  const [selected, setSelected] = useState<string[] | null>(null)
  const [seeds, setSeeds] = useState(freshSeeds)
  const [seedCount, setSeedCount] = useState(8)
  const [run, setRun] = useState<CompetitionRun | null>(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const abort = useRef<AbortController | null>(null)
  const mounted = useRef(true)
  const revision = useRef(0)
  const unique = entrants.filter((e, i) => entrants.findIndex((other) => other.sha256 === e.sha256) === i)
  const eligible = unique.filter((e) => fitsCompetition(e, division))
  const chosen = selected === null ? eligible.slice(0, 6) : eligible.filter((e) => selected.includes(e.id))
  const limit = DIVISIONS.find((d) => d.id === division)
  const total = chosen.length * (chosen.length - 1) * seedCount
  const rows = run ? standings(run) : []
  const currentHashes = new Set(unique.map((e) => e.sha256))
  const canRepeat = run?.spec.entrants.every((e) => currentHashes.has(e.sha256)) ?? false

  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false; revision.current++; abort.current?.abort() }
  }, [])
  useEffect(() => {
    let active = true
    const version = ++revision.current
    abort.current?.abort()
    setSelected(null)
    setRun(null)
    setBusy(false)
    void (async () => {
      try {
        const value = localStorage.getItem(STORAGE_KEY)
        if (!value) return
        const saved = JSON.parse(value)
        if (saved?.spec?.rulesId !== rulesId || saved?.spec?.contractId !== contractId) return
        const restored = await restoreCompetition(saved)
        if (active && revision.current === version) setRun(restored)
      } catch {
        if (active && revision.current === version) setMessage('A previous local result could not be restored. You can run a new league.')
      }
    })()
    return () => { active = false; abort.current?.abort() }
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
        catch { setMessage('The completed result is available here, but this browser could not save it. Export it to keep a copy.') }
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
  function handcraftedLabel(hash: string) {
    if (!handcraftedHashes.includes(hash)) return null
    return <span className="ranking-handcrafted"><Cpu size={10} aria-hidden="true" />Handcrafted · No training</span>
  }
  function classLabel(entrant: CompetitionEntrant) {
    return <span className="ranking-class" data-controller-type={entrant.type}>
      {entrant.type === 'script' ? 'Script' : smallestDivision(entrant)}
    </span>
  }

  return (
    <div className="ranking-page">
      <section className="panel ranking-intro">
        <div className="ranking-heading"><Trophy size={20} /><h2>Make the controllers compete</h2><span className="mini-tag">LOCAL LEAGUE</span></div>
        <p>Every controller faces the same opponents and serves, playing both sides. Training method does not affect eligibility or score.</p>
        <p className="ranking-note">All controllers ranks models and scripts together. Nano, Small, Standard and Open are weights-only classes.</p>
        <p className="ranking-note">Size limits are provisional. Smaller models can enter larger classes. Each league has its own roster and results.</p>
      </section>
      <section className="panel ranking-config">
        <fieldset disabled={busy}>
          <div className="ranking-fields">
            <label>League
              <select value={division} onChange={(e) => { setDivision(e.target.value as CompetitionClass); setSelected(null) }}>
                <option value="all">All controllers</option>
                {DIVISIONS.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
              </select>
              <small>{limit ? <>Weights only · Up to {numbers(limit.maxParams)} parameters · {numbers(limit.maxMacs)} MACs / decision</>
                : 'Models of every size and built-in scripts compete for the same ranks.'}</small>
            </label>
            <label>Serve sets per pairing
              <select value={seedCount} onChange={(e) => setSeedCount(Number(e.target.value))}>
                <option value={4}>4 · quick comparison</option>
                <option value={8}>8 · standard comparison</option>
                <option value={16}>16 · broader comparison</option>
              </select>
              <small>Five rallies per game, then swap sides on the same serve set.</small>
            </label>
          </div>
          <div className="ranking-roster-heading"><h3>Choose the field</h3><span>{chosen.length} selected · {eligible.length} eligible</span></div>
          <div className="ranking-roster">
            {eligible.map((e) => <label className="ranking-entry" key={e.sha256}>
              <input type="checkbox" checked={chosen.some((c) => c.id === e.id)} onChange={() => toggle(e.id)} aria-label={'Include ' + e.name} />
              <span><strong>{e.name}</strong>{handcraftedLabel(e.sha256)}<small>{e.type === 'script'
                ? 'Deterministic script' : <>{numbers(e.params!)} parameters · {numbers(e.macs!)} MACs</>}</small></span>
              {classLabel(e)}
            </label>)}
          </div>
          {!eligible.length && <p className="ranking-note">No loaded controllers are eligible for this league yet. Import a model or choose another league.</p>}
          <div className="ranking-actions">
            <button className="import-btn" disabled={chosen.length < 2} onClick={() => void start()}><Play size={16} /> Run league</button>
            <button className="secondary-btn" onClick={() => setSeeds(freshSeeds())}><Shuffle size={16} /> New serves</button>
            <span>{numbers(total)} games · {numbers(total * 5)} rallies</span>
          </div>
          <details className="ranking-seeds"><summary>Current serve seeds</summary><code>{seeds.slice(0, seedCount).join(', ')}</code></details>
        </fieldset>
        {busy && <button className="secondary-btn ranking-cancel" onClick={() => abort.current?.abort()}><Square size={15} /> Cancel league</button>}
      </section>
      {message && <p className="message" role="status">{message}</p>}
      {run && <section className="panel ranking-results">
        <div className="ranking-result-heading">
          <div><div className="section-code">{run.spec.division === 'all' ? 'ALL CONTROLLERS' : run.spec.division.toUpperCase()} / {run.spec.rulesId}</div><h2>{run.status === 'complete' ? 'League standings' : run.status === 'running' ? 'League in progress' : 'Incomplete league'}</h2></div>
          <span className="mini-tag">{run.status === 'complete' ? 'COMPLETE' : run.status.toUpperCase()}</span>
        </div>
        <div className="ranking-progress"><progress value={run.completedMatches} max={run.totalMatches} aria-label="Completed league games" /><span>{numbers(run.completedMatches)} / {numbers(run.totalMatches)} games</span></div>
        {run.status !== 'complete' && <p className="ranking-note">Only completed side pairs contribute below. Ranks appear after every scheduled game finishes.</p>}
        {run.error && <p className="message error" role="alert">{run.error}</p>}
        <div className="ranking-table-wrap">
          <table className="ranking-table">
            <thead><tr><th>Rank</th><th>Controller</th><th>League points</th><th>W / D / L</th><th>Rally points</th><th>Parameters</th></tr></thead>
            <tbody>{rows.map((row) => <tr key={row.entrant.sha256}>
              <td>{row.rank ?? '—'}</td><td><strong>{row.entrant.name}</strong>{handcraftedLabel(row.entrant.sha256)}<small>{row.entrant.sha256.slice(0, 10)} · {classLabel(row.entrant)}</small></td>
              <td>{row.leaguePoints.toLocaleString()}<small>of {row.played} contests</small></td>
              <td>{row.wins} / {row.draws} / {row.losses}</td><td>{row.rallyPointsFor} : {row.rallyPointsAgainst}<small>{row.rallyDraws} drawn rallies</small></td>
              <td>{row.entrant.type === 'script' ? '—' : numbers(row.entrant.params!)}</td>
            </tr>)}</tbody>
          </table>
        </div>
        <p className="ranking-note">A contest combines both side assignments: ten rallies on the same seed. More rally points wins one league point; a tied contest gives half a point each. Equal league points share rank. Rally length is not rewarded.</p>
        <div className="ranking-actions">
          <button className="secondary-btn" disabled={busy} onClick={download}><ArrowDownToLine size={16} /> Export results</button>
          <button className="secondary-btn" disabled={busy || !canRepeat} onClick={() => void start(run.spec)}><RotateCcw size={16} /> Repeat benchmark</button>
          <code className="ranking-fingerprint">Benchmark {run.benchmark.slice(0, 16)}</code>
        </div>
        {!canRepeat && <p className="ranking-note">The same model files and script versions must be available to repeat this saved benchmark.</p>}
        <details className="ranking-seeds"><summary>Recorded serve seeds</summary><code>{run.spec.seeds.join(', ')}</code></details>
      </section>}
      <p className="ranking-footnote">These are local benchmark results, not a verified public leaderboard. Compare scores only within the same benchmark. New opponents, rules, classes or serves create a new benchmark; public rankings will require independent evaluation.</p>
    </div>
  )
}

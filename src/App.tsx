import { useEffect, useRef, useState, type MouseEvent } from 'react'
import {
  ArrowDownToLine,
  ArrowRight,
  Check,
  CircuitBoard,
  FlaskConical,
  Gamepad2,
  Github,
  Layers3,
  Network,
  Plus,
  Upload,
  Trophy,
  X,
} from 'lucide-react'
import { Arena, ModelIcon } from './Arena'
import { keyboard, loadImportedModel, loadPresets, type Controller } from './models'
import type { Pair, Side } from './game'
import { MAX_BYTES } from './validate'
import { savedModels, saveModel } from './storage'
import { useMatch } from './useMatch'
import { Architecture, architectureLabel } from './Architecture'
import { ModelSubmitter } from './ModelSubmitter'
import { ModelOrigin } from './ModelOrigin'
import { ControllerBadge } from './ControllerBadge'
import { PONG } from './rules'
import { Ranking } from './Ranking'
import { RankBadge } from './RankBadge'
import { useReferenceRanks } from './useReferenceRanks'
import { WeightExplainers, type ExplainerId } from './WeightExplainers'
import { browserCompetitionMatch } from './runCompetition'
import { sha256, smallestDivision, type CompetitionEntrant } from './competition'
import { DeviceTest } from './DeviceTest'
import { finishStartup, getStartupSnapshot, markArenaReady, recordStartupFailure, recordStartupTiming } from './startup'
import { assetPath } from './assets'
import { DEFAULT_MODEL, DEFAULT_WATCH_IDS, readMatchLink, writeMatchLink } from './matchLink'
import { bottomSide, playerNumber, topSide } from './courtPlayers'
import { Findings, type FindingId } from './Findings'
import { QuickRules } from './QuickRules'
import { normalizeReadingUrl, pageMetadata, pagePath, readPage, type Page } from './routes'
import './App.css'

const tabs: { page: Page; label: string; icon: typeof Gamepad2 }[] = [
  { page: 'Arena', label: 'Play', icon: Gamepad2 },
  { page: 'How it thinks', label: 'How it thinks', icon: Network },
  { page: 'Rankings', label: 'League', icon: Trophy },
  { page: 'Model library', label: 'Models', icon: Layers3 },
  { page: 'Findings', label: 'Findings', icon: FlaskConical },
]
const descriptions = {
  Rankings: 'Programmed, trained, and prompted models, tested on the same serves from both sides.',
  'Model library': 'Programmed, trained and prompted describe how the weights were made. Every player is a neural network.',
  Rules: 'A quick guide to playing, watching, and bringing your own model.',
  'Device test': 'Measure this device with a repeatable workload, then copy the report to compare.',
}
const rules = PONG
export default function App({ location = new URL(window.location.href), base = new URL(document.baseURI) }: {
  location?: URL; base?: URL
}) {
  const [{ page, explainer, finding, catalogId }, setRoute] = useState(() => readPage(location, base))
  const referenceRanks = useReferenceRanks(page === 'Model library')
  const [initialMatch] = useState(() => readMatchLink(location.search))
  const [mode, setMode] = useState(initialMatch.mode)
  const [demo, setDemo] = useState(initialMatch.demo && page === 'Arena')
  const [startup, setStartup] = useState(getStartupSnapshot)
  const [loadingComplete, setLoadingComplete] = useState(false)
  const [models, setModels] = useState<Controller[]>([keyboard])
  const currentModels = useRef(models)
  const [ids, setIds] = useState<Pair<string>>(initialMatch.ids)
  const [error, setError] = useState(''),
    [notice, setNotice] = useState('')
  const [importSide, setImportSide] = useState<Side>(0),
    [uploading, setUploading] = useState(false),
    [fileError, setFileError] = useState(''),
    [drag, setDrag] = useState(false)
  const dialog = useRef<HTMLDialogElement>(null),
    fileInput = useRef<HTMLInputElement>(null)
  const eligibleModels = models.filter((model) => model.id === 'keyboard' ||
    model.inputSize === rules.inputSize && model.outputSize === rules.outputSize)
  const linkedModel = page === 'Model library'
    ? eligibleModels.find(model => model.id === catalogId && model.type === 'weights')
    : undefined
  useEffect(() => {
    if (!linkedModel) return
    const card = document.getElementById('catalog-model-' + linkedModel.id)
    card?.focus({ preventScroll: true })
    card?.scrollIntoView({ block: 'start' })
  }, [linkedModel])
  const ready = ids.every((id) => eligibleModels.some((model) => model.id === id))
  useEffect(() => {
    if (ready) { markArenaReady(); setStartup(getStartupSnapshot()) }
  }, [ready])
  useEffect(() => {
    if (loadingComplete) { finishStartup(ready); setStartup(getStartupSnapshot()) }
  }, [loadingComplete, ready])
  const players = ids.map((id) => models.find((p) => p.id === id)) as Pair<Controller | undefined>
  const bottom = bottomSide(ids)
  const m = useMatch(players, setError, page === 'Arena', rules, undefined, true)
  useEffect(() => {
    if (!loadingComplete || !models.some((model) => model.id === DEFAULT_MODEL)) return
    const fallback: Pair<string> = mode === 'play' ? ['keyboard', DEFAULT_MODEL] : [DEFAULT_MODEL, 'quill']
    if (ids.some((id) => !eligibleModels.some((model) => model.id === id))) {
      setIds(ids.map((id, side) => eligibleModels.some((model) => model.id === id) ? id : fallback[side]) as Pair<string>)
      setNotice('A linked model is unavailable in this browser. Using a bundled model instead.')
      return
    }
    if (page === 'Arena' && !demo) {
      const url = writeMatchLink(new URL(window.location.href), mode, ids)
      window.history.replaceState(null, '', url)
    }
  }, [loadingComplete, mode, ids, models, page, demo])
  const currentMatch = useRef(m)
  currentMatch.current = m
  useEffect(() => {
    const metadata = pageMetadata({ page, explainer, finding })
    document.title = metadata.title
    for (const selector of ['meta[name="description"]', 'meta[property="og:description"]', 'meta[name="twitter:description"]'])
      document.querySelector<HTMLMetaElement>(selector)!.content = metadata.description
    for (const selector of ['meta[property="og:title"]', 'meta[name="twitter:title"]'])
      document.querySelector<HTMLMetaElement>(selector)!.content = metadata.title
    document.querySelector<HTMLLinkElement>('link[rel="canonical"]')!.href = metadata.url
    document.querySelector<HTMLMetaElement>('meta[property="og:url"]')!.content = metadata.url
  }, [page, explainer, finding])
  useEffect(() => {
    if (!demo) return
    const stopDemo = () => setDemo(false)
    const events = ['pointerdown', 'click', 'keydown', 'wheel'] as const
    for (const event of events) window.addEventListener(event, stopDemo, { capture: true, passive: true })
    return () => {
      for (const event of events) window.removeEventListener(event, stopDemo, true)
    }
  }, [demo])
  const finished = m.finished
  useEffect(() => {
    if (!demo || !ready || page !== 'Arena' || error) return
    let timer: ReturnType<typeof setTimeout>
    function continueDemo() {
      clearTimeout(timer)
      if (document.hidden || currentMatch.current.playing) return
      if (currentMatch.current.finished) {
        timer = setTimeout(() => {
          if (!document.hidden) currentMatch.current.play()
        }, 3500)
      } else currentMatch.current.play()
    }
    continueDemo()
    document.addEventListener('visibilitychange', continueDemo)
    return () => {
      clearTimeout(timer)
      document.removeEventListener('visibilitychange', continueDemo)
    }
  }, [demo, ready, page, error, m.playing, finished])
  useEffect(() => {
    function restoreRoute() {
      const url = normalizeReadingUrl(new URL(window.location.href), new URL(document.baseURI))
      if (url.href !== window.location.href) window.history.replaceState(null, '', url)
      const route = readPage(url, new URL(document.baseURI))
      setDemo(false)
      currentMatch.current.pause()
      if (route.page === 'Arena') {
        const match = readMatchLink(window.location.search)
        setMode(match.mode)
        setIds(match.ids)
      }
      setRoute(route)
    }
    window.addEventListener('popstate', restoreRoute)
    return () => window.removeEventListener('popstate', restoreRoute)
  }, [])
  function appendModels(entries: Controller[]) {
    const next = [...currentModels.current]
    for (const entry of entries) {
      if (next.some((model) => model.id === entry.id)) {
        void entry.session?.release().catch(() => setError('An unused model session could not be released.'))
      } else next.push(entry)
    }
    currentModels.current = next
    setModels(next)
  }
  useEffect(() => {
    let active = true
    const bundledStartedAt = performance.now()
    loadPresets()
      .then(async (builtins) => {
        if (!active) return
        recordStartupTiming('Bundled controller loading', bundledStartedAt)
        appendModels(builtins)
        const restoreStartedAt = performance.now()
        try {
          const restored: Controller[] = []
          for (const item of await savedModels()) {
            try {
              restored.push(
                await loadImportedModel(item.bytes, item.id, item.name),
              )
            } catch {
              if (active) {
                recordStartupFailure('A previously saved model could not be loaded.')
                setError('A previously saved model could not be loaded.')
              }
            }
          }
          if (active && restored.length) appendModels(restored)
        } catch {
          if (active) {
            recordStartupFailure('Browser storage was unavailable.')
            setNotice('Browser storage is unavailable. Models can still play in this session.')
          }
        } finally {
          if (active) recordStartupTiming('Saved controller restoration', restoreStartedAt)
        }
      })
      .catch((e) => {
        if (active) {
          recordStartupFailure(e instanceof Error ? e.message : String(e))
          setError(
            'Could not initialize the arena. ' + (e instanceof Error ? e.message : String(e)),
          )
        }
      })
      .finally(() => { if (active) setLoadingComplete(true) })
    return () => {
      active = false
    }
  }, [])
  function pageUrl(next: Page, lesson: ExplainerId | FindingId = 'follow') {
    const url = new URL(pagePath(next, lesson), base)
    url.search = location.search
    if (next === 'Arena') writeMatchLink(url, mode, ids)
    else for (const param of ['model', 'mode', 'opponent', 'fullscreen']) url.searchParams.delete(param)
    return assetPath(url.pathname.slice(base.pathname.length)) + url.search + url.hash
  }
  function navigate(next: Page, lesson: ExplainerId | FindingId = 'follow') {
    setDemo(false)
    if (next !== page) m.pause()
    const url = new URL(pageUrl(next, lesson), base)
    if (url.href !== window.location.href) {
      window.history.pushState(null, '', url)
      window.scrollTo({ top: 0 })
    }
    setRoute(readPage(url, base))
  }
  function followTab(event: MouseEvent<HTMLAnchorElement>, next: Page, lesson: ExplainerId | FindingId = 'follow') {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
    event.preventDefault()
    navigate(next, lesson)
  }
  function select(index: Side, id: string) {
    const keepPlaying = mode === 'watch' && (currentMatch.current.playing || currentMatch.current.finished)
    setDemo(false)
    currentMatch.current.reset()
    setIds((prev) => (index === 0 ? [id, prev[1]] : [prev[0], id]))
    setError('')
    if (keepPlaying) currentMatch.current.play()
  }
  function openImport() {
    setDemo(false)
    m.pause()
    setFileError('')
    setImportSide(mode === 'play' ? 1 : 0)
    dialog.current?.showModal()
  }
  function playAgainst(opponent: string) {
    m.reset()
    setIds(['keyboard', opponent])
    setMode('play')
    navigate('Arena')
    setError('')
    m.play()
  }
  function watchModels(first: string, second: string) {
    m.reset()
    setMode('watch')
    setIds([first, second])
    navigate('Arena')
    setError('')
    m.play()
  }
  function changeMode(next: 'play' | 'watch') {
    setDemo(false)
    if (next === mode) {
      if (next === 'watch') m.play()
      return
    }
    m.reset()
    setMode(next)
    const opponent = ids[topSide(bottom)]
    setIds(next === 'play' ? ['keyboard', opponent === 'keyboard' ? DEFAULT_MODEL : opponent] : DEFAULT_WATCH_IDS)
    setError('')
    m.play()
  }
  async function importFile(file: File | undefined) {
    if (!file || uploading) return
    setUploading(true)
    setFileError('')
    try {
      if (file.size > MAX_BYTES) throw new Error('Choose a file of 256 KiB or less.')
      const bytes = new Uint8Array(await file.arrayBuffer())
      const digest = await sha256(bytes)
      const id = 'local-' + digest
      let entry = currentModels.current.find((c) => c.id === id)
      let storageNotice = 'Available in this browser.'
      if (!entry) {
        const name = file.name.replace(/\.onnx$/i, '').trim() || 'Untitled model'
        entry = await loadImportedModel(bytes, id, name)
        try {
          await saveModel({ id, name, bytes })
          storageNotice = 'Saved in this browser.'
        } catch {
          storageNotice = 'Available for this session.'
        }
        appendModels([entry])
        entry = currentModels.current.find((c) => c.id === id)!
      }
      select(importSide, id)
      navigate('Arena')
      const nextIds: Pair<string> = importSide === 0 ? [id, ids[1]] : [ids[0], id]
      setNotice(entry.name + ' assigned to player 0' + playerNumber(importSide, bottomSide(nextIds)) + '. ' + storageNotice)
      dialog.current?.close()
    } catch (e) {
      setFileError(e instanceof Error ? e.message : 'Could not read this model.')
    } finally {
      setUploading(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }
  return (
    <>
      <header className="topbar">
        <div className="topinner">
          <a
            className="brand"
            aria-label="PongArena home"
            href={pageUrl('Arena')}
            onClick={event => followTab(event, 'Arena')}
          >
            <img src={assetPath('mark.svg')} alt="" />
            <span>
              Pong<span className="brand-light">Arena</span>
            </span>
          </a>
          <nav aria-label="Main navigation">
            {tabs.map(({ page: t, label, icon: Icon }) => (
              <a
                key={t}
                aria-current={page === t ? 'page' : undefined}
                href={pageUrl(t)}
                onClick={event => followTab(event, t)}
              >
                <Icon size={16} />
                {label}
              </a>
            ))}
          </nav>
          <a className="github-link" href="https://github.com/jonaslindblom/pongarena" target="_blank" rel="noopener noreferrer" aria-label="Source on GitHub" title="Source on GitHub">
            <Github size={18} aria-hidden="true" /><span>GitHub</span>
          </a>
        </div>
      </header>
      <main className={'shell' + (page === 'Arena' ? ' shell-arena' : '')}>
        {page === 'Arena' && (
          <header className="arena-intro">
            <h1>How small can a neural network be and still play Pong?</h1>
            <p>Play against programmed, trained, and prompted networks.{' '}
              <a href={pageUrl('Findings')} onClick={event => followTab(event, 'Findings')}>See what we learned</a>,{' '}
              <a href={pageUrl('How it thinks')} onClick={event => followTab(event, 'How it thinks')}>look inside a decision</a>, or{' '}
              <a href="https://github.com/jonaslindblom/pongarena/blob/main/CONTRIBUTING.md">build and test your own model</a>.
            </p>
          </header>
        )}
        {page !== 'Arena' && page !== 'How it thinks' && page !== 'Findings' && <div className="page-head">
          <div>
            <h1>{page === 'Rankings' ? 'League' : page}</h1>
            <p>{descriptions[page]}</p>
          </div>
          <div className="page-actions">
            {page === 'Model library' && <button className="import-btn" disabled={!ready} onClick={openImport}>
              <Plus size={17} /> Import model
            </button>}
          </div>
        </div>}
        {error && (
          <div className="message error" role="alert">
            {error}
            <button aria-label="Dismiss error" onClick={() => setError('')}>
              <X size={15} />
            </button>
          </div>
        )}
        {notice && (
          <div role="status" className="message">
            <Check size={16} />
            {notice}
            <button aria-label="Dismiss message" onClick={() => setNotice('')}>
              <X size={15} />
            </button>
          </div>
        )}
        {page === 'Arena' && (
          <Arena
            mode={mode}
            demo={demo}
            onModeChange={changeMode}
            models={eligibleModels}
            ids={ids}
            pick={select}
            swap={() => {
              setDemo(false)
              m.reset()
              setIds([ids[1], ids[0]])
            }}
            m={m}
            ready={ready}
          />
        )}
        {page === 'Findings' && <Findings models={models} selected={finding ?? 'training'} onSelect={(event, story) => followTab(event, 'Findings', story)} onLesson={event => followTab(event, 'How it thinks', 'counter')} onPlay={playAgainst} onWatch={watchModels} />}
        {page === 'Device test' && <DeviceTest models={eligibleModels} startup={startup} loadingComplete={loadingComplete} />}
        {page === 'How it thinks' && <WeightExplainers models={models} selected={explainer} onSelect={(event, lesson) => followTab(event, 'How it thinks', lesson)} onFinding={event => followTab(event, 'Findings', 'oneshot')} onPlay={playAgainst} />}
        {page === 'Rankings' && (
          <Ranking key={rules.id} entrants={eligibleModels.filter((model) => model.type !== 'human' && model.sha256).map((model) => ({
            id: model.id, name: model.name, sha256: model.sha256!, params: model.params, macs: model.macs,
          } satisfies CompetitionEntrant))} rulesId={rules.id} contractId={rules.contractId}
            modelDetails={eligibleModels.filter((model) => model.type === 'weights' && model.sha256).map((model) => ({
              sha256: model.sha256!, origin: model.origin, method: model.method,
            }))}
            playMatch={browserCompetitionMatch(eligibleModels, rules)} onStart={m.pause} onPlay={playAgainst} />
        )}
        {page === 'Model library' && (
          <>
          {catalogId && loadingComplete && !linkedModel && (
            <p className="library-rank-note" role="status">The linked model is unavailable in this browser.</p>
          )}
          <p className="library-rank-note" role="status">{referenceRanks.error
            ? 'Reference ranks unavailable: ' + referenceRanks.error
            : referenceRanks.loading ? 'Loading reference league ranks…' : 'Ranks from the reference league, matched to the exact model weights.'}</p>
          {[
            { title: 'Programmed', description: 'Weights calculated from deliberately designed formulas.', models: eligibleModels.filter(model => model.method === 'programmed' && model.origin !== 'local').sort((a, b) => a.params - b.params) },
            { title: 'Trained', description: 'Weights learned through training or automated optimization.', models: eligibleModels.filter(model => model.method === 'trained' && model.origin !== 'local').sort((a, b) => a.params - b.params) },
            { title: 'Prompted', description: 'Weights generated directly in response to a prompt.', models: eligibleModels.filter(model => model.method === 'prompted' && model.origin !== 'local').sort((a, b) => a.params - b.params) },
            { title: 'Mixed methods', description: 'Weights created by combining methods.', models: eligibleModels.filter(model => model.method === 'mixed' && model.origin !== 'local').sort((a, b) => a.params - b.params) },
            { title: 'Your models', description: 'Models imported into this browser.', models: eligibleModels.filter(model => model.origin === 'local') },
          ].filter(group => group.models.length > 0).map(group => (
          <section className="library-group" key={group.title} aria-label={group.title === 'Your models' ? group.title : `${group.title} models`}>
          <div className="library-group-heading"><h2>{group.title}</h2><p>{group.description}</p></div>
          <div className="library">
            {group.models.map((c) => (
                <article className="panel library-card" key={c.id} aria-label={`${c.name} model`}
                  id={'catalog-model-' + c.id} tabIndex={c.id === catalogId ? -1 : undefined}
                  data-model-id={c.id}
                  data-weight-origin={c.handcrafted ? 'handcrafted' : undefined}>
                  <div className="library-top">
                    <span className="library-icon">
                      <ModelIcon id={c.id} size={27} />
                    </span>
                    {c.sha256 && referenceRanks.ranks.has(c.sha256) &&
                      <RankBadge rank={referenceRanks.ranks.get(c.sha256)!} label="Reference league rank" />}
                  </div>
                  <small>{c.kind}</small>
                  <div className="library-name"><h2>{c.name}</h2><ModelOrigin origin={c.origin} /></div>
                  <ControllerBadge model={c} />
                  <ModelSubmitter handle={c.submittedBy} />
                  {c.architecture && <p className="library-layout" aria-label={`${c.name} layer layout`}>
                    <span>Layers</span> <b>{architectureLabel(c)}</b>
                  </p>}
                  <details className="model-details"><summary>About this model</summary>
                  {c.origin === 'community' && <small className="model-method-note">Method reported by the contributor.</small>}
                  <p>{c.description}</p>
                  {c.sourceUrl && <p><a className="model-source" href={c.sourceUrl} target="_blank" rel="noopener noreferrer">Source / recipe</a></p>}
                  <Architecture model={c} />
                  <dl>
                    <dt>Parameters</dt>
                    <dd>{c.params.toLocaleString()}</dd>
                    <dt>MACs / tick</dt>
                    <dd>{c.macs.toLocaleString()}</dd>
                    <dt>Smallest class</dt>
                    <dd>{smallestDivision(c) ?? '—'}</dd>
                  </dl>
                  </details>
                  <div className="library-actions">
                    {c.id === 'small' && <a className="secondary-btn" href={pageUrl('Findings')} onClick={event => followTab(event, 'Findings')}><FlaskConical size={14} /> Training findings</a>}
                    {c.id === 'oneshot' && <a className="secondary-btn"
                      href={pageUrl('Findings', 'oneshot')}
                      onClick={event => followTab(event, 'Findings', 'oneshot')}
                    ><Network size={14} /> See the prompt</a>}
                    {(c.id === 'follow' || c.id === 'counter') && <button className="secondary-btn" onClick={() => {
                      navigate('How it thinks', c.id as ExplainerId)
                    }}><Network size={14} /> How it thinks</button>}
                    <button className="secondary-btn" onClick={() => playAgainst(c.id)}>
                      <Gamepad2 size={14} /> Play against
                    </button>
                    <button
                      className="secondary-btn"
                      onClick={() => watchModels(c.id, c.id === 'quill' ? 'large' : 'quill')}
                    >
                      Watch <ArrowRight size={14} />
                    </button>
                    {(c.origin === 'core' || c.origin === 'community') && (
                      <a
                        aria-label={'Download ' + c.name}
                        title="Download ONNX"
                        className="icon-btn"
                        href={assetPath('models/' + c.id + '.onnx')}
                        download={c.id + '.onnx'}
                      >
                        <ArrowDownToLine size={16} />
                      </a>
                    )}
                  </div>
                </article>
              ))}
          </div>
          </section>
          ))}
          </>
        )}
        {page === 'Rules' && <QuickRules />}
        <footer>
          <span>
            <FlaskConical size={13} /> PongArena <b>/</b> Neural network playground
          </span>
          <div className="footer-links">
            <a href={pageUrl('Rules')} onClick={event => followTab(event, 'Rules')}>Rules & controls</a>
            <a href="https://github.com/jonaslindblom/pongarena/blob/main/CONTRIBUTING.md">Contribute a model</a>
            <a href={pageUrl('Device test')} onClick={event => followTab(event, 'Device test')}>Device test</a>
          </div>
        </footer>
      </main>
      <dialog
        ref={dialog}
        onCancel={(e) => {
          if (uploading) e.preventDefault()
        }}
        className="import-modal"
      >
        <div className="modal-title">
          <span className="library-icon">
            <CircuitBoard size={25} />
          </span>
          <button
            aria-label="Close import"
            disabled={uploading}
            onClick={() => dialog.current?.close()}
            className="icon-btn"
          >
            <X size={20} />
          </button>
        </div>
        <div className="section-code">{rules.name.toUpperCase()} / LOCAL IMPORT</div>
        <h2>Import a model</h2>
        <p>Choose a self-contained ONNX file with six observations and three action scores: left, stay, right. Your weights stay in this browser. <a href="https://github.com/jonaslindblom/pongarena/blob/main/CONTRIBUTING.md" target="_blank" rel="noopener noreferrer">Model requirements</a></p>
        {mode === 'watch' && <fieldset disabled={uploading}>
          <legend>Assign to paddle</legend>
          <div className="side-options">
            {([topSide(bottom), bottom] as const).map((i) => (
              <button key={i} aria-pressed={i === importSide} onClick={() => setImportSide(i)}>
                Player {playerNumber(i, bottom)} · {i === bottom ? 'Bottom' : 'Top'}
              </button>
            ))}
          </div>
        </fieldset>}
        <button
          className={'dropzone ' + (drag ? 'drag' : '')}
          disabled={uploading}
          onClick={() => fileInput.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setDrag(true)
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDrag(false)
            void importFile(e.dataTransfer.files[0])
          }}
        >
          <Upload size={26} />
          <b>{uploading ? 'Checking model…' : 'Choose or drop a file'}</b>
          <small>.onnx · up to 256 KiB</small>
        </button>
        <input
          aria-label="ONNX model file"
          ref={fileInput}
          type="file"
          accept=".onnx"
          hidden
          onChange={(e) => void importFile(e.target.files?.[0])}
        />
        {fileError && (
          <p role="alert" className="modal-error">
            {fileError}
          </p>
        )}
        <a className="starter-link" href={assetPath('models/quill.onnx')} download>
          <ArrowDownToLine size={15} /> Download Quill
        </a>
      </dialog>
    </>
  )
}

import { useEffect, useRef, useState, type MouseEvent } from 'react'
import {
  ArrowDownToLine,
  ArrowRight,
  BookOpen,
  Check,
  CircuitBoard,
  FlaskConical,
  Gamepad2,
  Github,
  Layers3,
  Monitor,
  Network,
  Smartphone,
  Plus,
  Upload,
  Trophy,
  X,
} from 'lucide-react'
import { Arena, ModelIcon } from './Arena'
import { keyboard, loadModel, loadPresets, presets, type Controller } from './models'
import type { Pair, Side } from './game'
import { MAX_BYTES, MAX_DENSE_HIDDEN_LAYERS } from './validate'
import { savedModels, saveModel } from './storage'
import { useMatch } from './useMatch'
import { Architecture, architectureLabel } from './Architecture'
import { ModelSubmitter } from './ModelSubmitter'
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
import { DEFAULT_MODEL, readMatchLink, writeMatchLink } from './matchLink'
import quillSourceUrl from '../handwritten/quill.py?url'
import './App.css'
type Page = 'Arena' | 'Rankings' | 'Model library' | 'How it thinks' | 'Field guide' | 'Device test'
const pageHashes: Record<Page, string> = {
  Arena: '#play',
  Rankings: '#league',
  'Model library': '#models',
  'How it thinks': '#how-it-thinks',
  'Field guide': '#guide',
  'Device test': '#device-test',
}
const counterLessonHash = '#how-it-thinks/counter'

function readPage(hash: string): { page: Page; explainer: ExplainerId } {
  if (hash === counterLessonHash) return { page: 'How it thinks', explainer: 'counter' }
  const page = (Object.keys(pageHashes) as Page[]).find(page => pageHashes[page] === hash) ?? 'Arena'
  return { page, explainer: 'follow' }
}

const tabs: { page: Page; label: string; icon: typeof Gamepad2 }[] = [
  { page: 'Arena', label: 'Play', icon: Gamepad2 },
  { page: 'Rankings', label: 'League', icon: Trophy },
  { page: 'Model library', label: 'Models', icon: Layers3 },
  { page: 'How it thinks', label: 'How it thinks', icon: Network },
  { page: 'Field guide', label: 'Guide', icon: BookOpen },
]
const descriptions = {
  Rankings: 'Handwritten and trained models, tested on the same serves from both sides.',
  'Model library': 'Play against a model, inspect its brain, or bring your own weights.',
  'Field guide': 'The game rules, observations, actions, and submission contract.',
  'Device test': 'Measure this device with a repeatable workload, then copy the report to compare.',
}
const rules = PONG
const inputs = [
  ['Ball X', 'Centered at 0. Negative toward your end, positive toward your opponent.'],
  ['Ball Y', 'Simulation Y, centered at 0. On screen, negative is left and positive is right.'],
  ['Ball velocity X', 'An approaching serve moves 5 units toward your paddle per tick, so the model receives −5/6 (about −0.83). Moving away is positive.'],
  ['Ball velocity Y', 'Moving 3 units right per tick on screen gives the model +0.5. Moving left is negative.'],
  ['Your paddle Y', 'Position of your paddle across the screen, using the same scale as ball Y.'],
  ['Opponent paddle Y', 'Position of the opposing paddle across the screen, using the same scale as ball Y.'],
]
function Guide({ onDeviceTest }: { onDeviceTest: () => void }) {
  return (
    <div className="guide-grid">
      <article className="panel guide">
        <div className="section-code">01 / GAME RULES</div>
        <h2>Play the match</h2>
        <p>
          Each match runs five seeded rallies at 30 decisions per second. A miss gives the other
          paddle one point. After 1,600 ticks a rally is drawn. The higher score after five rallies
          wins.
        </p>
        <p>
          Serves start at the center with gentle, varied angles and alternate ends.
          Human games count down three seconds before each serve.
          New games use fresh serves. Restart repeats the same serves for practice.
        </p>
        <p>
          When two models repeat the same ball and paddle state after a return, their rally is
          drawn early: the same decisions would repeat forever. Matches with a human use the
          full time limit. Drawn rallies award no points.
        </p>
        <p>
          Aim with the contact point on your paddle: edge hits send steeper returns. Every successful
          return increases speed toward the opposing paddle by one, from a gentle serve at 5 up to a cap of 13.
          Center hits keep that speed. Paddles move 2 units per tick. Both players see the
          same instant before either moves. Opening the arena picks five serves from different
          positions across the court. Each paddle receives serves moving both left and right. Restart repeats the set.
        </p>
        <p>
          Every match uses a vertical court. You play from the bottom. Hold the left
          half to move left or the right half to move right, and lift to stop. You can also use A/D,
          ←/→, or the touch controls. Press P to play, pause or resume; after game over, P plays again. Press R to restart immediately. Press F for full screen and F or Esc to exit; full screen hides the workspace and locks page scrolling.
          Touch and pointer controls steer through the same movement limit. Animation fills the space between
          decisions; pausing preserves the exact game state.
        </p>
        <p>
          Quill is a neural network with handwritten weights: its ball prediction and paddle control
          are built directly into dense matrices. Small and Large are trained in PyTorch.
          How it thinks starts with Follow’s single neuron, then shows how Counter uses ball direction to switch between interception and recovery.
          Every model receives the same observations and chooses from the same moves.
        </p>
        <div className="section-code">02 / OBSERVATION</div>
        <h2>What your model sees</h2>
        <p>
          One float32 tensor named <code>observation</code> with shape <code>[1, {rules.inputSize}]</code>.
          Models use simulation coordinates: your paddle is on the left from the
          model’s perspective, and the court is 360 by 160 units, with positions scaled to −1 and +1
          at the edges. On screen, this court is rotated: simulation X runs between the paddles,
          and simulation Y runs left to right.
        </p>
        <p>
          For each velocity input, the arena divides the ball’s movement in one tick by 6. This
          puts typical speeds on a similar scale to positions. Fast shots can still exceed 1 in
          magnitude. Multiply the input by 6 to recover court units per tick.
        </p>
        <div className="input-table">
          {inputs.map(([n, d], i) => (
            <div key={n}>
              <small>0{i}</small>
              <b>{n}</b>
              <span>{d}</span>
            </div>
          ))}
        </div>
        <div className="section-code">03 / ACTION</div>
        <h2>Choose a move</h2>
        <p>
          Return finite float32 scores in <code>action_scores</code> with shape <code>[1, {rules.outputSize}]</code>,
          ordered <strong>{rules.actions.join(', ').toLowerCase()}</strong> in simulation coordinates.
          On screen these mean <strong>left, stay, right</strong>. The highest score wins.
          Ties go to the earliest action. Ordinary moves cover two court units.
        </p>
        <p>
          The brain panel shows the model’s real neuron activations and highlights its selected action.
          Pause to inspect a decision. Human players have no neural network.
        </p>
      </article>
      <aside className="panel guide guide-side">
        <div className="section-code">FILE CONTRACT</div>
        <h2>Bring your weights</h2>
        <p>Use a plain dense neural network with embedded float32 weights and biases.</p>
        <dl>
          <dt>Operator set</dt>
          <dd>ONNX 18</dd>
          <dt>Max. file size</dt>
          <dd>256 KiB</dd>
          <dt>MACs per decision</dt>
          <dd>65,536</dd>
          <dt>Ops per graph</dt>
          <dd>{2 * MAX_DENSE_HIDDEN_LAYERS + 1}</dd>
          <dt>Intermediate elements, total</dt>
          <dd>262,144</dd>
        </dl>
        <p>
          Map the six observations directly to three scores through one to {MAX_DENSE_HIDDEN_LAYERS + 1} <code>Gemm</code>
          {' '}layers. Use zero to {MAX_DENSE_HIDDEN_LAYERS} hidden layers, with the same <code>Relu</code> or{' '}
          <code>Tanh</code> activation after every hidden layer. The output layer is linear.
        </p>
        <p>
          Each dense layer uses its own fixed matrix and bias. Branches, skip connections,
          feature calculations, input products, and output processing are rejected. Put
          model knowledge in the dense weights and biases.
        </p>
        <a className="secondary-btn" href={assetPath('models/quill.onnx')} download>
          <ArrowDownToLine size={16} /> Quill ONNX
        </a>
        <h3>Handwrite a model</h3>
        <p>
          Quill turns geometric rules into weights and biases, inspired by{' '}
          <a href="https://rikiyat.github.io/blog/weight-space-programming/" target="_blank" rel="noreferrer">
            weight-space programming
          </a>. Read <a href={quillSourceUrl} download="quill.py">Quill’s Python source</a>, change a formula, and compare
          the resulting model in the league.
        </p>
        <h3>Train a model</h3>
        <p>
          The repository’s <code>training/</code> folder contains a standalone PyTorch recipe for
          Small and Large, with scripted teachers, reinforcement learning, evaluation and ONNX export.
          Start with its README, then import your exported model here.
        </p>
        <h3>Check your device</h3>
        <p>Measure inference and animation with the same workload on different devices.</p>
        <button className="secondary-btn" onClick={onDeviceTest}><Smartphone size={16} /> Device test</button>
      </aside>
    </div>
  )
}

export default function App() {
  const [{ page, explainer }, setRoute] = useState(() => readPage(window.location.hash))
  const referenceRanks = useReferenceRanks(page === 'Model library')
  const [initialMatch] = useState(() => readMatchLink(window.location.search))
  const [mode, setMode] = useState(initialMatch.mode)
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
  const ready = ids.every((id) => eligibleModels.some((model) => model.id === id))
  useEffect(() => {
    if (ready) { markArenaReady(); setStartup(getStartupSnapshot()) }
  }, [ready])
  useEffect(() => {
    if (loadingComplete) { finishStartup(ready); setStartup(getStartupSnapshot()) }
  }, [loadingComplete, ready])
  const players = ids.map((id) => models.find((p) => p.id === id)) as Pair<Controller | undefined>
  const m = useMatch(players, setError, page === 'Arena', rules, undefined, true)
  useEffect(() => {
    if (!loadingComplete || !models.some((model) => model.id === DEFAULT_MODEL)) return
    const fallback: Pair<string> = mode === 'play' ? ['keyboard', DEFAULT_MODEL] : [DEFAULT_MODEL, 'quill']
    if (ids.some((id) => !eligibleModels.some((model) => model.id === id))) {
      setIds(ids.map((id, side) => eligibleModels.some((model) => model.id === id) ? id : fallback[side]) as Pair<string>)
      setNotice('A linked model is unavailable in this browser. Using a bundled model instead.')
      return
    }
    if (page === 'Arena') {
      const url = writeMatchLink(new URL(window.location.href), mode, ids)
      window.history.replaceState(null, '', url)
    }
  }, [loadingComplete, mode, ids, models, page])
  const currentMatch = useRef(m)
  currentMatch.current = m
  useEffect(() => {
    function restoreRoute() {
      const route = readPage(window.location.hash)
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
                await loadModel(item.bytes, {
                  id: item.id,
                  name: item.name,
                  kind: 'Imported model',
                  description: 'Saved in this browser.',
                }),
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
  function pageUrl(next: Page, lesson: ExplainerId = 'follow') {
    const url = new URL(window.location.href)
    url.hash = next === 'How it thinks' && lesson === 'counter' ? counterLessonHash : pageHashes[next]
    if (next === 'Arena') return writeMatchLink(url, mode, ids)
    for (const param of ['model', 'mode', 'opponent', 'fullscreen']) url.searchParams.delete(param)
    return url
  }
  function navigate(next: Page, lesson: ExplainerId = 'follow') {
    if (next !== page) m.pause()
    const url = pageUrl(next, lesson)
    if (url.href !== window.location.href) window.history.pushState(null, '', url)
    setRoute({ page: next, explainer: lesson })
  }
  function followTab(event: MouseEvent<HTMLAnchorElement>, next: Page) {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
    event.preventDefault()
    navigate(next)
  }
  function select(index: Side, id: string) {
    currentMatch.current.reset()
    setIds((prev) => (index === 0 ? [id, prev[1]] : [prev[0], id]))
    setError('')
  }
  function openImport() {
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
  function changeMode(next: 'play' | 'watch') {
    if (next === mode) return
    m.reset()
    setMode(next)
    setIds(next === 'play' ? ['keyboard', ids[1]] : ['large', DEFAULT_MODEL])
    setError('')
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
      if (!entry) {
        const name = file.name.replace(/\.onnx$/i, '').trim() || 'Untitled model'
        entry = await loadModel(bytes, {
          id,
          name,
          kind: 'Imported model',
          description: 'Saved in this browser.',
        })
        try {
          await saveModel({ id, name, bytes })
        } catch {
          entry.description = 'Available for this session.'
        }
        appendModels([entry])
        entry = currentModels.current.find((c) => c.id === id)!
      }
      select(importSide, id)
      navigate('Arena')
      setNotice(entry.name + ' assigned to player 0' + (importSide + 1) + '. ' + entry.description)
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
            href={pageUrl('Arena').href}
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
                href={pageUrl(t).href}
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
        {page === 'Arena' && <h1 className="visually-hidden">Pong arena</h1>}
        {page !== 'Arena' && page !== 'How it thinks' && <div className="page-head">
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
            onModeChange={changeMode}
            models={eligibleModels}
            ids={ids}
            pick={select}
            swap={() => {
              m.reset()
              setIds([ids[1], ids[0]])
            }}
            m={m}
            ready={ready}
          />
        )}
        {page === 'Device test' && <DeviceTest models={eligibleModels} startup={startup} loadingComplete={loadingComplete} />}
        {page === 'How it thinks' && <WeightExplainers models={models} selected={explainer} onSelect={lesson => navigate('How it thinks', lesson)} onPlay={playAgainst} />}
        {page === 'Rankings' && (
          <Ranking key={rules.id} entrants={eligibleModels.filter((model) => model.type !== 'human' && model.sha256).map((model) => ({
            id: model.id, name: model.name, sha256: model.sha256!, params: model.params, macs: model.macs,
          } satisfies CompetitionEntrant))} rulesId={rules.id} contractId={rules.contractId}
            handcraftedHashes={eligibleModels.filter((model) => model.handcrafted && model.sha256).map((model) => model.sha256!)}
            playMatch={browserCompetitionMatch(eligibleModels, rules)} onStart={m.pause} onPlay={playAgainst} />
        )}
        {page === 'Model library' && (
          <>
          <p className="library-rank-note" role="status">{referenceRanks.error
            ? 'Reference ranks unavailable: ' + referenceRanks.error
            : referenceRanks.loading ? 'Loading reference league ranks…' : 'Ranks from the reference league, matched to the exact model weights.'}</p>
          {[
            { title: 'Handwritten', description: 'From following the ball to prediction, aiming and recovery.', models: eligibleModels.filter(model => model.handcrafted && !model.id.startsWith('local-')).sort((a, b) => a.params - b.params) },
            { title: 'Trained', description: 'Neural networks learned in PyTorch, from small to large.', models: eligibleModels.filter(model => model.type === 'weights' && !model.handcrafted && !model.id.startsWith('local-')).sort((a, b) => a.params - b.params) },
            { title: 'Your models', description: 'Models imported and saved in this browser.', models: eligibleModels.filter(model => model.id.startsWith('local-')) },
          ].filter(group => group.models.length > 0).map(group => (
          <section className="library-group" key={group.title} aria-label={`${group.title} models`}>
          <div className="library-group-heading"><h2>{group.title}</h2><p>{group.description}</p></div>
          <div className="library">
            {group.models.map((c) => (
                <article className="panel library-card" key={c.id} aria-label={`${c.name} model`}
                  data-model-id={c.id}
                  data-weight-origin={c.handcrafted ? 'handcrafted' : undefined}>
                  <div className="library-top">
                    <span className="library-icon">
                      <ModelIcon id={c.id} size={27} />
                    </span>
                    {c.sha256 && referenceRanks.ranks.has(c.sha256)
                      ? <RankBadge rank={referenceRanks.ranks.get(c.sha256)!} label="Reference league rank" />
                      : <span className="mini-tag">{c.id.startsWith('local-') ? 'IMPORTED' : 'BUILT-IN'}</span>}
                  </div>
                  <small>{c.kind}</small>
                  <h2>{c.name}</h2>
                  <ControllerBadge model={c} />
                  <ModelSubmitter handle={c.submittedBy} />
                  {c.architecture && <p className="library-layout" aria-label={`${c.name} layer layout`}>
                    <span>Layers</span> <b>{architectureLabel(c)}</b>
                  </p>}
                  <details className="model-details"><summary>About this model</summary>
                  <p>{c.description}</p>
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
                    {(c.id === 'follow' || c.id === 'counter') && <button className="secondary-btn" onClick={() => {
                      navigate('How it thinks', c.id as ExplainerId)
                    }}><Network size={14} /> How it thinks</button>}
                    <button className="secondary-btn" onClick={() => playAgainst(c.id)}>
                      <Gamepad2 size={14} /> Play against
                    </button>
                    <button
                      className="secondary-btn"
                      onClick={() => {
                        m.reset()
                        setMode('watch')
                        setIds([c.id, c.id === 'quill' ? 'large' : 'quill'])
                        navigate('Arena')
                        setError('')
                        m.play()
                      }}
                    >
                      Watch <ArrowRight size={14} />
                    </button>
                    {presets.some((p) => p.id === c.id) && (
                      <a
                        aria-label={'Download ' + c.name}
                        title="Download ONNX"
                        className="icon-btn"
                        href={assetPath('models/' + c.id + '.onnx')}
                        download
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
        {page === 'Field guide' && <Guide onDeviceTest={() => navigate('Device test')} />}
        <footer>
          <span>
            <FlaskConical size={13} /> PongArena <b>/</b> Neural network playground
          </span>
          <span>
            <Monitor size={14} /> Runs in your browser
          </span>
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
        <p>Choose a self-contained ONNX file with six observations and three action scores: up, stay, down in simulation coordinates (left, stay, right on screen). Your weights stay in this browser.</p>
        {mode === 'watch' && <fieldset disabled={uploading}>
          <legend>Assign to paddle</legend>
          <div className="side-options">
            {([0, 1] as const).map((i) => (
              <button key={i} aria-pressed={i === importSide} onClick={() => setImportSide(i)}>
                Player {i + 1}
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

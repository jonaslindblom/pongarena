import { useEffect, useRef, useState } from 'react'
import {
  ArrowDownToLine,
  ArrowRight,
  BookOpen,
  Check,
  CircuitBoard,
  FlaskConical,
  Gamepad2,
  Layers3,
  Monitor,
  Plus,
  Upload,
  Trophy,
  X,
} from 'lucide-react'
import { Arena, ModelIcon } from './Arena'
import { keyboard, loadModel, loadPresets, presets, type Controller } from './models'
import type { Pair, Side } from './game'
import { ArchivedModelError, MAX_BYTES } from './validate'
import { savedModels, saveModel } from './storage'
import { useMatch } from './useMatch'
import { Architecture } from './Architecture'
import { ControllerBadge } from './ControllerBadge'
import { CLASSIC } from './rules'
import { Ranking } from './Ranking'
import { browserCompetitionMatch } from './runCompetition'
import { sha256, smallestDivision, type CompetitionEntrant } from './competition'
type Page = 'Arena' | 'Rankings' | 'Model library' | 'Field guide'
const tabs: { page: Page; icon: typeof Gamepad2 }[] = [
  { page: 'Arena', icon: Gamepad2 },
  { page: 'Rankings', icon: Trophy },
  { page: 'Model library', icon: Layers3 },
  { page: 'Field guide', icon: BookOpen },
]
const descriptions: Record<Page, string> = {
  Arena: 'Pit two controllers against each other and watch every decision unfold.',
  Rankings: 'Run reproducible local tournaments with equal budgets, paired serves, and your own weights.',
  'Model library': 'Choose a controller, download a baseline, or bring your own weights.',
  'Field guide': 'The game rules, observations, actions, and submission contract.',
}
const rules = CLASSIC
const inputs = [
  ['Ball X', 'Centered at 0. Negative toward your end, positive toward your opponent.'],
  ['Ball Y', 'Centered at 0. Negative at the top, positive at the bottom.'],
  ['Ball velocity X', 'An approaching serve moves 5 units toward your paddle per tick, so the model receives −5/6 (about −0.83). Moving away is positive.'],
  ['Ball velocity Y', 'Moving 3 units down per tick gives the model +0.5. Moving up is negative.'],
  ['Your paddle Y', 'Center of your paddle, using the same scale as ball Y.'],
  ['Opponent paddle Y', 'Center of the opposing paddle, using the same scale as ball Y.'],
]
function Guide() {
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
          When two models or scripts repeat the same ball and paddle state after a return, their rally is
          drawn early: the same decisions would repeat forever. Matches with a human use the
          full time limit. Drawn rallies award no points.
        </p>
        <p>
          Aim with the contact point on your paddle: edge hits send steeper returns. Every successful
          return increases horizontal speed by one, from a gentle serve at 5 up to a cap of 13.
          Center hits keep that speed. Paddles move 2 units per tick. Both players see the
          same instant before either moves. Opening the arena picks five serves from different
          heights. Each paddle receives both rising and falling serves. Restart repeats the set.
        </p>
        <p>
          Choose Human to use W/S, arrow keys, the on-screen buttons, or drag the court. Touch and
          pointer controls steer through the same movement limit. Animation fills the space between
          decisions; paused frames and replays preserve the exact game state.
        </p>
        <p>
          The amber Script badge marks a deterministic program: Attacker plans shot placement,
          Defender plans catches and recovery, and Interceptor follows the predicted landing point.
          They use the same six observations and movement limits as weight models. Pick one as
          Player 02, then press Play yourself to face it.
        </p>
        <div className="section-code">02 / OBSERVATION</div>
        <h2>What your model sees</h2>
        <p>
          One float32 tensor named <code>observation</code> with shape <code>[1, {rules.inputSize}]</code>. Your
          paddle is always on the left from the model’s perspective. The court is 360 by 160 units,
          with positions scaled to −1 and +1 at the edges.
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
          ordered <strong>{rules.actions.join(', ').toLowerCase()}</strong>. The highest score wins.
          Ties go to the earliest action. Ordinary moves cover two court units.
        </p>
        <p>
          The inspector shows the raw scores for the move that produced the displayed frame. Bars
          start at zero in the center and scale to the largest absolute score for that player and
          tick. Scripts show a 1 for their chosen action and 0 for the others; these are decisions,
          not confidence scores.
        </p>
      </article>
      <aside className="panel guide guide-side">
        <div className="section-code">FILE CONTRACT</div>
        <h2>Bring your weights</h2>
        <p>Use a single stateless, fixed-shape ONNX file with embedded float32 weights.</p>
        <dl>
          <dt>Operator set</dt>
          <dd>ONNX 18</dd>
          <dt>Max. file size</dt>
          <dd>256 KiB</dd>
          <dt>MACs per decision</dt>
          <dd>65,536</dd>
          <dt>Ops per graph</dt>
          <dd>256</dd>
          <dt>Intermediate elements, total</dt>
          <dd>262,144</dd>
        </dl>
        <p>
          Supported operations include <code>Gemm</code>, <code>MatMul</code>, <code>Add</code>,{' '}
          <code>Relu</code>, and <code>Tanh</code>.
        </p>
        <a className="secondary-btn" href="/models/tracker.onnx" download>
          <ArrowDownToLine size={16} />
          Starter ONNX
        </a>
        <h3>Train independently</h3>
        <p>
          The v4 package includes exact rules, a headless example, a Python reference, and
          deterministic test cases.
        </p>
        <a className="secondary-btn" href="/training/pong-v4.zip" download>
          <ArrowDownToLine size={16} /> Training package
        </a>
        <h3>First weight edit</h3>
        <p>
          In Tracker, change <code>w1[0, 3]</code> from 0 to 0.525. The paddle now accounts for
          vertical ball velocity. Compare it with the bundled Anticipator.
        </p>
      </aside>
    </div>
  )
}

export default function App() {
  const [page, setPage] = useState<Page>('Arena')
  const [models, setModels] = useState<Controller[]>([keyboard])
  const currentModels = useRef(models)
  const [ids, setIds] = useState<Pair<string>>(['pacer-v4', 'apprentice-v4'])
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
  const ready = models.some((p) => p.id === 'tracker')
  const players = ids.map((id) => models.find((p) => p.id === id)) as Pair<Controller | undefined>
  const m = useMatch(players, setError, page === 'Arena', rules)
  const currentMatch = useRef(m)
  currentMatch.current = m
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
    loadPresets()
      .then(async (builtins) => {
        if (!active) return
        appendModels(builtins)
        if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches && initialView.current)
          m.play()
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
            } catch (error) {
              if (active && !(error instanceof ArchivedModelError))
                setError('A previously saved model could not be loaded.')
            }
          }
          if (active && restored.length) appendModels(restored)
        } catch {
          if (active)
            setNotice('Browser storage is unavailable. Models can still play in this session.')
        }
      })
      .catch((e) => {
        if (active)
          setError(
            'Could not initialize the arena. ' + (e instanceof Error ? e.message : String(e)),
          )
      })
    return () => {
      active = false
    }
  }, [])
  const initialView = useRef(true)
  function navigate(next: Page) {
    initialView.current = false
    if (next !== page) m.pause()
    setPage(next)
  }
  function select(index: Side, id: string) {
    currentMatch.current.reset()
    setIds((prev) => (index === 0 ? [id, prev[1]] : [prev[0], id]))
    setError('')
  }
  function openImport() {
    initialView.current = false
    m.pause()
    setFileError('')
    setImportSide(0)
    dialog.current?.showModal()
  }
  function playAgainst(opponent: string) {
    initialView.current = false
    m.reset()
    m.setSpeed(1)
    setIds(['keyboard', opponent])
    setPage('Arena')
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
      setPage('Arena')
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
            href="#arena"
            onClick={(e) => {
              e.preventDefault()
              navigate('Arena')
            }}
          >
            <img src="/mark.svg" alt="" />
            <span>
              Rally<span className="brand-light">Lab</span>
            </span>
          </a>
          <nav aria-label="Main navigation">
            {tabs.map(({ page: t, icon: Icon }) => (
              <button
                key={t}
                aria-current={page === t ? 'page' : undefined}
                onClick={() => navigate(t)}
              >
                <Icon size={16} />
                {t}
              </button>
            ))}
          </nav>
          <span className="local-status">
            <i className="online-dot" /> Local workspace
          </span>
        </div>
      </header>
      <main className="shell">
        <div className="page-head">
          <div>
            <div className="eyebrow">
              <i /> NEURAL GAME LAB <span>/</span> {rules.name.toUpperCase()}
            </div>
            <h1>{page === 'Arena' ? 'Match arena' : page}</h1>
            <p>{descriptions[page]}</p>
          </div>
          <div className="page-actions">
            {page === 'Arena' && (
              <button className="secondary-btn" disabled={!ready}
                onClick={() => playAgainst(ids[1] === 'keyboard' ? 'tracker' : ids[1])}>
                <Gamepad2 size={17} /> Play yourself
              </button>
            )}
            <button className="import-btn" disabled={!ready} onClick={openImport}>
              <Plus size={17} /> Import model
            </button>
          </div>
        </div>
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
        {page === 'Rankings' && (
          <Ranking key={rules.id} entrants={eligibleModels.filter((model) => model.type === 'weights' && model.sha256).map((model) => ({
            id: model.id, name: model.name, sha256: model.sha256!, params: model.params, macs: model.macs,
          } satisfies CompetitionEntrant))} rulesId={rules.id} contractId={rules.contractId}
            playMatch={browserCompetitionMatch(eligibleModels, rules)} onStart={m.pause} />
        )}
        {page === 'Model library' && (
          <div className="library">
            {eligibleModels
              .filter((c) => c.id !== 'keyboard')
              .map((c) => (
                <article className="panel library-card" key={c.id}>
                  <div className="library-top">
                    <span className="library-icon">
                      <ModelIcon id={c.id} size={27} />
                    </span>
                    <span className="mini-tag">
                      {c.id.startsWith('local-') ? 'IMPORTED' : 'BUILT-IN'}
                    </span>
                  </div>
                  <small>{c.kind}</small>
                  <h2>{c.name}</h2>
                  <ControllerBadge model={c} />
                  <p>{c.description}</p>
                  <Architecture model={c} />
                  <dl>
                    {c.type === 'weights' ? <>
                    <dt>Parameters</dt>
                    <dd>{c.params.toLocaleString()}</dd>
                    <dt>MACs / tick</dt>
                    <dd>{c.macs.toLocaleString()}</dd>
                    <dt>Smallest class</dt>
                    <dd>{smallestDivision(c) ?? '—'}</dd>
                    </> : <>
                    <dt>Controller</dt>
                    <dd>Scripted logic</dd>
                    <dt>Weights</dt>
                    <dd>None</dd>
                    <dt>Observations / actions</dt>
                    <dd>6 / 3</dd>
                    </>}
                    <dt>Game contract</dt>
                    <dd>Classic v4</dd>
                  </dl>
                  {c.report && (
                    <a className="training-link" href={c.report} target="_blank" rel="noreferrer">
                      Learning curves & evaluation ↗
                    </a>
                  )}
                  <div className="library-actions">
                    <button className="secondary-btn" onClick={() => playAgainst(c.id)}>
                      <Gamepad2 size={14} /> Play against
                    </button>
                    <button
                      className="secondary-btn"
                      onClick={() => {
                        select(0, c.id)
                        setPage('Arena')
                      }}
                    >
                      Use as P1 <ArrowRight size={14} />
                    </button>
                    {presets.some((p) => p.id === c.id) && (
                      <a
                        aria-label={'Download ' + c.name}
                        title="Download ONNX"
                        className="icon-btn"
                        href={'/models/' + c.id + '.onnx'}
                        download
                      >
                        <ArrowDownToLine size={16} />
                      </a>
                    )}
                  </div>
                </article>
              ))}
          </div>
        )}
        {page === 'Field guide' && <Guide />}
        <footer>
          <span>
            <FlaskConical size={13} /> Rally Lab <b>/</b> Local edition
          </span>
          <span>
            <Monitor size={14} /> Browser inference
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
        <p>Choose a self-contained ONNX file with six observations and three action scores: up, stay, down. Your weights stay in this browser.</p>
        <fieldset disabled={uploading}>
          <legend>Assign to paddle</legend>
          <div className="side-options">
            {([0, 1] as const).map((i) => (
              <button key={i} aria-pressed={i === importSide} onClick={() => setImportSide(i)}>
                P{i + 1} · {i === 0 ? 'Left' : 'Right'}
              </button>
            ))}
          </div>
        </fieldset>
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
        <a className="starter-link" href="/models/tracker.onnx" download>
          <ArrowDownToLine size={15} /> Download the starter paddle
        </a>
      </dialog>
    </>
  )
}

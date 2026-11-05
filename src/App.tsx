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
  X,
} from 'lucide-react'
import { Arena, ModelIcon } from './Arena'
import { keyboard, loadModel, loadPresets, presets, type Controller } from './models'
import type { Pair, Side } from './game'
import { MAX_BYTES } from './validate'
import { savedModels, saveModel } from './storage'
import { useMatch } from './useMatch'
import { Architecture } from './Architecture'
type Page = 'Arena' | 'Model library' | 'Field guide'
const tabs: { page: Page; icon: typeof Gamepad2 }[] = [
  { page: 'Arena', icon: Gamepad2 },
  { page: 'Model library', icon: Layers3 },
  { page: 'Field guide', icon: BookOpen },
]
const descriptions: Record<Page, string> = {
  Arena: 'Pit two controllers against each other and watch every decision unfold.',
  'Model library': 'Choose a controller, download a baseline, or bring your own weights.',
  'Field guide': 'Everything your paddle needs to compete in Pong v3.',
}
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
          When two models repeat the same ball and paddle state after a return, their rally is
          drawn early: the same decisions would repeat forever. Matches with a human use the
          full time limit. Drawn rallies award no points.
        </p>
        <p>
          Hit near the edge of your paddle to send a faster, steeper return. Horizontal ball speed
          ranges from 5 to 13 units per tick; paddles move 2 units per tick. Both players see the
          same instant before either moves. Opening the arena picks five serves from different
          heights. Each paddle receives both rising and falling serves. Restart repeats the set.
        </p>
        <div className="section-code">02 / OBSERVATION</div>
        <h2>What your model sees</h2>
        <p>
          One float32 tensor named <code>observation</code> with shape <code>[1, 6]</code>. Your
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
          Return finite float32 scores in <code>action_scores</code> with shape <code>[1, 3]</code>,
          ordered <strong>up, stay, down</strong>. The highest score wins. Ties go to the earliest
          action. Each move covers two court units.
        </p>
        <p>
          The inspector shows the raw scores for the move that produced the displayed frame. Bars
          start at zero in the center and scale to the largest absolute score for that player and
          tick.
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
          The v3 package includes exact rules, a headless example, a Python reference, and
          deterministic test cases.
        </p>
        <a className="secondary-btn" href="/training/pong-v3.zip" download>
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
  const [ids, setIds] = useState<Pair<string>>(['contender-v3', 'selfplay-v2'])
  const [error, setError] = useState(''),
    [notice, setNotice] = useState('')
  const [importSide, setImportSide] = useState<Side>(0),
    [uploading, setUploading] = useState(false),
    [fileError, setFileError] = useState(''),
    [drag, setDrag] = useState(false)
  const dialog = useRef<HTMLDialogElement>(null),
    fileInput = useRef<HTMLInputElement>(null)
  const ready = models.some((p) => p.id === 'tracker')
  const players = ids.map((id) => models.find((p) => p.id === id)) as Pair<Controller | undefined>
  const m = useMatch(players, setError, page === 'Arena')
  useEffect(() => {
    let active = true
    loadPresets()
      .then(async (builtins) => {
        if (!active) return
        setModels([keyboard, ...builtins])
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
            } catch {
              if (active) setError('A previously saved model could not be loaded.')
            }
          }
          if (active && restored.length)
            setModels((old) => [...old, ...restored.filter((c) => !old.some((a) => a.id === c.id))])
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
    m.reset()
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
  async function importFile(file: File | undefined) {
    if (!file || uploading) return
    setUploading(true)
    setFileError('')
    try {
      if (file.size > MAX_BYTES) throw new Error('Choose a file of 256 KiB or less.')
      const bytes = new Uint8Array(await file.arrayBuffer())
      const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (b) =>
        b.toString(16).padStart(2, '0'),
      ).join('')
      const id = 'local-' + digest
      let entry = models.find((c) => c.id === id)
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
        setModels((prev) => [...prev, entry!])
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
              <i /> NEURAL GAME LAB <span>/</span> PONG V3
            </div>
            <h1>{page === 'Arena' ? 'Match arena' : page}</h1>
            <p>{descriptions[page]}</p>
          </div>
          <button className="import-btn" disabled={!ready} onClick={openImport}>
            <Plus size={17} /> Import model
          </button>
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
            models={models}
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
        {page === 'Model library' && (
          <div className="library">
            {models
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
                  <p>{c.description}</p>
                  <Architecture model={c} />
                  <dl>
                    <dt>Parameters</dt>
                    <dd>{c.params.toLocaleString()}</dd>
                    <dt>MACs / tick</dt>
                    <dd>{c.macs.toLocaleString()}</dd>
                  </dl>
                  {c.report && (
                    <a className="training-link" href={c.report} target="_blank" rel="noreferrer">
                      Learning curves & evaluation ↗
                    </a>
                  )}
                  <div className="library-actions">
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
        <div className="section-code">PONG V3 / LOCAL IMPORT</div>
        <h2>Import a model</h2>
        <p>Choose a self-contained ONNX file. Your weights stay in this browser.</p>
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

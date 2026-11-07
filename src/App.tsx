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
  Smartphone,
  Plus,
  Upload,
  Trophy,
  X,
} from 'lucide-react'
import { Arena, ModelIcon } from './Arena'
import { keyboard, loadModel, loadPresets, presets, type Controller } from './models'
import type { Pair, Side } from './game'
import { ArchivedModelError, DenseModelError, MAX_BYTES, MAX_DENSE_HIDDEN_LAYERS } from './validate'
import { savedModels, saveModel } from './storage'
import { useMatch } from './useMatch'
import { Architecture } from './Architecture'
import { ControllerBadge } from './ControllerBadge'
import { CLASSIC } from './rules'
import { Ranking } from './Ranking'
import { browserCompetitionMatch } from './runCompetition'
import { sha256, smallestDivision, type CompetitionEntrant } from './competition'
import { DeviceTest } from './DeviceTest'
import { finishStartup, getStartupSnapshot, markArenaReady, recordStartupFailure, recordStartupTiming } from './startup'
import { assetPath } from './assets'
import './App.css'
type Page = 'Arena' | 'Rankings' | 'Model library' | 'Field guide' | 'Device test'
const tabs: { page: Page; label: string; icon: typeof Gamepad2 }[] = [
  { page: 'Arena', label: 'Play', icon: Gamepad2 },
  { page: 'Rankings', label: 'League', icon: Trophy },
  { page: 'Model library', label: 'Models', icon: Layers3 },
  { page: 'Field guide', label: 'Guide', icon: BookOpen },
]
const descriptions: Record<Page, string> = {
  Arena: 'Move with A/D or ←/→. Press R to restart.',
  Rankings: 'One field, common serves, both sides. See how the controllers compare.',
  'Model library': 'Choose a controller, download a baseline, or bring your own weights.',
  'Field guide': 'The game rules, observations, actions, and submission contract.',
  'Device test': 'Measure this device with a repeatable workload, then copy the report to compare.',
}
const rules = CLASSIC
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
          When two models or scripts repeat the same ball and paddle state after a return, their rally is
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
          ←/→, or the on-screen buttons. Press R to restart immediately. Full screen hides the workspace and locks page scrolling.
          Touch and pointer controls steer through the same movement limit. Animation fills the space between
          decisions; pausing preserves the exact game state.
        </p>
        <p>
          The amber Script badge marks a deterministic program: Attacker plans shot placement,
          Defender plans catches and recovery, and Interceptor follows the predicted landing point.
          They use the same six observations and movement limits as weight models. Choose one
          as your opponent, or use Play against on its model card.
        </p>
        <p>
          The blue Handcrafted weights badge marks a dense neural network whose weights were
          designed manually and generated from fixed formulas, without training. Vector and
          Prism use this method. Their decisions run through the neural network.
        </p>
        <div className="section-code">02 / OBSERVATION</div>
        <h2>What your model sees</h2>
        <p>
          One float32 tensor named <code>observation</code> with shape <code>[1, {rules.inputSize}]</code>.
          Models keep the original simulation coordinates: your paddle is on the left from the
          model’s perspective, and the court is 360 by 160 units, with positions scaled to −1 and +1
          at the edges. On screen, this court is rotated: simulation X runs between the paddles,
          and simulation Y runs left to right. Existing models use the same inputs and actions.
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
          Show brain opens the model’s real neuron activations and highlights its selected action.
          Pause to inspect a decision. Scripts and human players have no neural network.
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
        <a className="secondary-btn" href={assetPath('models/tracker.onnx')} download>
          <ArrowDownToLine size={16} />
          Starter ONNX
        </a>
        <h3>Train independently</h3>
        <p>
          The v4 package includes exact rules, a headless example, a Python reference, and
          deterministic test cases. Follow the dense-network contract above when exporting;
          the frozen package retains the older general-graph export description.
        </p>
        <a className="secondary-btn" href={assetPath('training/pong-v4.zip')} download>
          <ArrowDownToLine size={16} /> Training package
        </a>
        <h3>First weight edit</h3>
        <p>
          In Tracker, change <code>w1[0, 3]</code> from 0 to 0.525. The paddle now accounts for
          ball velocity across the court (simulation Y). Import the edited file and compare it with Tracker.
        </p>
        <h3>Check your device</h3>
        <p>Measure inference and animation with the same workload on different devices.</p>
        <button className="secondary-btn" onClick={onDeviceTest}><Smartphone size={16} /> Device test</button>
      </aside>
    </div>
  )
}

export default function App() {
  const [page, setPage] = useState<Page>('Arena')
  const [mode, setMode] = useState<'play' | 'watch'>('play')
  const [startup, setStartup] = useState(getStartupSnapshot)
  const [loadingComplete, setLoadingComplete] = useState(false)
  const [models, setModels] = useState<Controller[]>([keyboard])
  const currentModels = useRef(models)
  const [ids, setIds] = useState<Pair<string>>(['keyboard', 'tracker'])
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
  useEffect(() => {
    if (ready) { markArenaReady(); setStartup(getStartupSnapshot()) }
  }, [ready])
  useEffect(() => {
    if (loadingComplete) { finishStartup(ready); setStartup(getStartupSnapshot()) }
  }, [loadingComplete, ready])
  const players = ids.map((id) => models.find((p) => p.id === id)) as Pair<Controller | undefined>
  const m = useMatch(players, setError, page === 'Arena', rules, undefined, true)
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
    const bundledStartedAt = performance.now()
    loadPresets()
      .then(async (builtins) => {
        if (!active) return
        recordStartupTiming('Bundled controller loading', bundledStartedAt)
        appendModels(builtins)
        const restoreStartedAt = performance.now()
        try {
          const restored: Controller[] = []
          let retiredModels = false
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
              if (error instanceof DenseModelError) retiredModels = true
              else if (active && !(error instanceof ArchivedModelError)) {
                recordStartupFailure('A previously saved model could not be loaded.')
                setError('A previously saved model could not be loaded.')
              }
            }
          }
          if (active && restored.length) appendModels(restored)
          if (active && retiredModels)
            setNotice('Some saved models do not meet the dense network contract and were not restored. Import a plain dense network with six inputs and three outputs.')
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
  function navigate(next: Page) {
    if (next !== page) m.pause()
    setPage(next)
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
    setPage('Arena')
    setError('')
    m.play()
  }
  function changeMode(next: 'play' | 'watch') {
    if (next === mode) return
    m.reset()
    setMode(next)
    setIds(next === 'play' ? ['keyboard', ids[1]] : ['prism-small-v4', 'quill-plus-v4'])
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
            <img src={assetPath('mark.svg')} alt="" />
            <span>
              Pong<span className="brand-light">Arena</span>
            </span>
          </a>
          <nav aria-label="Main navigation">
            {tabs.map(({ page: t, label, icon: Icon }) => (
              <button
                key={t}
                aria-current={page === t ? 'page' : undefined}
                onClick={() => navigate(t)}
              >
                <Icon size={16} />
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>
      <main className="shell">
        <div className="page-head">
          <div>
            <h1>{page === 'Arena' ? mode === 'play' ? 'Play Pong' : 'Watch models' : page === 'Rankings' ? 'League' : page}</h1>
            <p>{descriptions[page]}</p>
          </div>
          <div className="page-actions">
            {page === 'Arena' && <div className="play-mode" role="group" aria-label="Game mode">
              <button aria-pressed={mode === 'play'} onClick={() => changeMode('play')}>You play</button>
              <button aria-pressed={mode === 'watch'} onClick={() => changeMode('watch')}>Watch models</button>
            </div>}
            {page === 'Model library' && <button className="import-btn" disabled={!ready} onClick={openImport}>
              <Plus size={17} /> Import model
            </button>}
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
            mode={mode}
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
        {page === 'Rankings' && (
          <Ranking key={rules.id} entrants={eligibleModels.filter((model) => model.type !== 'human' && model.sha256).map((model) => ({
            id: model.id, name: model.name, sha256: model.sha256!, type: model.type as 'weights' | 'script',
            params: model.type === 'script' ? null : model.params, macs: model.type === 'script' ? null : model.macs,
          } satisfies CompetitionEntrant))} rulesId={rules.id} contractId={rules.contractId}
            handcraftedHashes={eligibleModels.filter((model) => model.handcrafted && model.sha256).map((model) => model.sha256!)}
            playMatch={browserCompetitionMatch(eligibleModels, rules)} onStart={m.pause} onPlay={playAgainst} />
        )}
        {page === 'Model library' && (
          <div className="library">
            {eligibleModels
              .filter((c) => c.id !== 'keyboard')
              .map((c) => (
                <article className="panel library-card" key={c.id}
                  data-weight-origin={c.handcrafted ? 'handcrafted' : undefined}>
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
                  <details className="model-details"><summary>About this controller</summary>
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
                    <a className="training-link" href={assetPath(c.report)} target="_blank" rel="noreferrer">
                      {c.handcrafted ? 'Weight design & evaluation ↗' : 'Learning curves & evaluation ↗'}
                    </a>
                  )}
                  </details>
                  <div className="library-actions">
                    <button className="secondary-btn" onClick={() => playAgainst(c.id)}>
                      <Gamepad2 size={14} /> Play against
                    </button>
                    <button
                      className="secondary-btn"
                      onClick={() => {
                        m.reset()
                        setMode('watch')
                        setIds([c.id, c.id === 'quill-plus-v4' ? 'prism-small-v4' : 'quill-plus-v4'])
                        setPage('Arena')
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
        )}
        {page === 'Field guide' && <Guide onDeviceTest={() => navigate('Device test')} />}
        <footer>
          <span>
            <FlaskConical size={13} /> PongArena <b>/</b> Classic v4
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
        <a className="starter-link" href={assetPath('models/tracker.onnx')} download>
          <ArrowDownToLine size={15} /> Download the starter paddle
        </a>
      </dialog>
    </>
  )
}

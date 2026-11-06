import { useEffect, useRef, useState } from 'react'
import { flushSync } from 'react-dom'
import { Activity, Check, Copy, Play, Smartphone, Square } from 'lucide-react'
import { env } from 'onnxruntime-web/wasm'
import { Arena } from './Arena'
import { infer, type Controller } from './models'
import { CLASSIC, type Pair } from './rules'
import { useMatch, type MatchTickTiming } from './useMatch'
import {
  benchmarkInference, DEVICE_TEST, DEVICE_TEST_METHODS, runAnimationSample,
  type AnimationMeasurement, type InferenceMeasurement,
} from './deviceBenchmark'
import type { StartupSnapshot } from './startup'
import './DeviceTest.css'

// Keep the existing storage namespace so saved data survives the product rename.
const STORAGE_KEY = 'rally-lab-device-test-v1'
const matchIds = [...DEVICE_TEST.matchIds] as Pair<string>
const viewport = () => ({ width: innerWidth, height: innerHeight, pixelRatio: devicePixelRatio })
const ms = (value: number | null | undefined) => value == null ? '—' : value.toFixed(1) + ' ms'

interface DeviceReport {
  schema: 'pongarena-device-report-v1'
  status: 'complete'
  measuredAt: string
  device: string
  visit: string
  startup: StartupSnapshot
  environment: {
    userAgent: string
    language: string
    hardwareConcurrency: number
    secureContext: boolean
    reducedMotion: boolean
    viewportAtStart: ReturnType<typeof viewport>
    viewportDuringAnimation: ReturnType<typeof viewport>
    viewportAtEnd: ReturnType<typeof viewport>
  }
  runtime: { name: string; version: string | undefined; executionProvider: string; simd: boolean; threads: number }
  workload: typeof DEVICE_TEST
  methods: typeof DEVICE_TEST_METHODS
  rules: { id: string; contract: string }
  animationModels: { id: string; name: string; sha256: string }[]
  loadedWeightModels: { id: string; name: string; sha256: string | undefined }[]
  inference: InferenceMeasurement[]
  animation: AnimationMeasurement
}

function restoreReport(startup: StartupSnapshot): DeviceReport | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(STORAGE_KEY) ?? 'null')
    return (value?.schema === 'pongarena-device-report-v1' || value?.schema === 'rally-lab-device-report-v1') && value.status === 'complete' &&
      value.workload?.version === DEVICE_TEST.version &&
      value.startup?.pageId === startup.pageId && value.startup?.build?.id === startup.build.id
      ? { ...value, schema: 'pongarena-device-report-v1' } : null
  } catch { return null }
}

function nextFrame(signal: AbortSignal) {
  signal.throwIfAborted()
  return new Promise<void>((resolve, reject) => {
    const cancel = () => { cancelAnimationFrame(raf); reject(signal.reason) }
    const raf = requestAnimationFrame(() => { signal.removeEventListener('abort', cancel); resolve() })
    signal.addEventListener('abort', cancel, { once: true })
  })
}

export function DeviceTest({ models, startup, loadingComplete }: {
  models: Controller[]
  startup: StartupSnapshot
  loadingComplete: boolean
}) {
  const [report, setReport] = useState<DeviceReport | null>(() => restoreReport(startup))
  const [device, setDevice] = useState(() => report?.device ?? '')
  const [visit, setVisit] = useState(() => report?.visit ?? 'unspecified')
  const [stage, setStage] = useState<'idle' | 'inference' | 'animation'>('idle')
  const [progress, setProgress] = useState({ completed: 0, total: 1, modelName: '' })
  const [message, setMessage] = useState('')
  const [copyMessage, setCopyMessage] = useState('')
  const [showReport, setShowReport] = useState(false)
  const abort = useRef<AbortController | null>(null)
  const mounted = useRef(true)
  const ticks = useRef<MatchTickTiming[] | null>(null)
  const animationStartedAt = useRef(0)
  const preview = useRef<HTMLFieldSetElement>(null)
  const status = useRef<HTMLDivElement>(null)
  const reportBox = useRef<HTMLTextAreaElement>(null)
  const matchPlayers = matchIds.map((id) => models.find((model) => model.id === id)) as Pair<Controller | undefined>
  const m = useMatch(matchPlayers, (error) => {
    abort.current?.abort(new Error(error))
    if (mounted.current) setMessage(error)
  }, true, CLASSIC, (tick) => ticks.current?.push(tick))
  const latestMatch = useRef(m)
  latestMatch.current = m
  const requiredIds = [...DEVICE_TEST.modelIds, ...DEVICE_TEST.matchIds]
  const ready = loadingComplete && requiredIds.every((id) => models.some((model) => model.id === id && model.session && model.sha256))
  const busy = stage !== 'idle'
  const reportText = report ? JSON.stringify(report, null, 2) : ''

  useEffect(() => {
    mounted.current = true
    const visibility = () => {
      if (document.hidden) abort.current?.abort(new Error('Test stopped because the page became hidden. Keep it visible and run again.'))
    }
    const orientation = () => abort.current?.abort(new Error('Test stopped because the device rotated. Keep the same orientation and run again.'))
    document.addEventListener('visibilitychange', visibility)
    window.addEventListener('orientationchange', orientation)
    return () => {
      mounted.current = false
      abort.current?.abort(new Error('Test stopped because you left Device test.'))
      document.removeEventListener('visibilitychange', visibility)
      window.removeEventListener('orientationchange', orientation)
    }
  }, [])

  async function start() {
    if (!ready || abort.current || document.hidden) return
    const controller = new AbortController()
    abort.current = controller
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur()
    const label = device.trim() || 'Not specified'
    const visitLabel = visit
    const startupSnapshot = structuredClone(startup)
    const viewportAtStart = viewport()
    let viewportDuringAnimation = viewportAtStart
    setReport(null)
    setShowReport(false)
    setCopyMessage('')
    setMessage('')
    setProgress({ completed: 0, total: DEVICE_TEST.batches * DEVICE_TEST.modelIds.length, modelName: 'Warming models' })
    setStage('inference')
    try { sessionStorage.removeItem(STORAGE_KEY) } catch { /* The report remains available without storage. */ }
    try {
      latestMatch.current.pause()
      const inference = await benchmarkInference(models, controller.signal, (value) => {
        if (mounted.current) setProgress(value)
      }, infer)
      controller.signal.throwIfAborted()
      latestMatch.current.reset(DEVICE_TEST.seed)
      latestMatch.current.setSpeed(1)
      flushSync(() => setStage('animation'))
      // Mount and position the real arena before starting the timed workload.
      await nextFrame(controller.signal)
      controller.signal.throwIfAborted()
      if (preview.current) preview.current.style.scrollMarginTop = (status.current?.getBoundingClientRect().height ?? 80) + 16 + 'px'
      preview.current?.scrollIntoView({ block: 'start', behavior: 'auto' })
      await nextFrame(controller.signal)
      controller.signal.throwIfAborted()
      const samples: MatchTickTiming[] = []
      ticks.current = samples
      const animation = await runAnimationSample({
        signal: controller.signal,
        ticks: samples,
        start: () => {
          viewportDuringAnimation = viewport()
          animationStartedAt.current = performance.now()
          latestMatch.current.play()
        },
        stop: () => latestMatch.current.pause(),
      })
      controller.signal.throwIfAborted()
      const completed: DeviceReport = {
        schema: 'pongarena-device-report-v1', status: 'complete', measuredAt: new Date().toISOString(),
        device: label, visit: visitLabel, startup: startupSnapshot,
        environment: {
          userAgent: navigator.userAgent, language: navigator.language,
          hardwareConcurrency: navigator.hardwareConcurrency, secureContext: window.isSecureContext,
          reducedMotion: matchMedia('(prefers-reduced-motion: reduce)').matches,
          viewportAtStart, viewportDuringAnimation, viewportAtEnd: viewport(),
        },
        runtime: { name: 'ONNX Runtime Web', version: env.versions.web, executionProvider: 'wasm', simd: true, threads: 1 },
        workload: DEVICE_TEST, methods: DEVICE_TEST_METHODS,
        rules: { id: CLASSIC.id, contract: CLASSIC.contractId },
        animationModels: matchPlayers.map((model) => ({ id: model!.id, name: model!.name, sha256: model!.sha256! })),
        loadedWeightModels: models.filter((model) => model.type === 'weights').map((model) => ({ id: model.id, name: model.name, sha256: model.sha256 })),
        inference, animation,
      }
      if (mounted.current) {
        setReport(completed)
        try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify(completed)) }
        catch { setMessage('Test complete. Copy the report before leaving this page; browser storage is unavailable.') }
      }
    } catch (error) {
      if (mounted.current) setMessage(error instanceof Error ? error.message : String(error))
    } finally {
      ticks.current = null
      latestMatch.current.pause()
      if (abort.current === controller) abort.current = null
      if (mounted.current) setStage('idle')
    }
  }

  async function copyReport() {
    if (!report) return
    setCopyMessage('')
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(reportText)
        if (mounted.current) setCopyMessage('Report copied. Paste it into this chat.')
        return
      }
    } catch { /* HTTP LAN origins and denied clipboard access use the selectable report below. */ }
    setShowReport(true)
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()))
    if (!mounted.current) return
    reportBox.current?.focus()
    reportBox.current?.select()
    reportBox.current?.setSelectionRange(0, reportText.length)
    let copied = false
    try { copied = document.execCommand('copy') } catch { /* Manual copy remains available. */ }
    setCopyMessage(copied ? 'Report copied. Paste it into this chat.' : 'Automatic copy is unavailable. The report is selected below; use your browser’s Copy command.')
  }

  const animationSeconds = stage === 'animation' && animationStartedAt.current
    ? Math.max(0, Math.ceil((DEVICE_TEST.animationMs - performance.now() + animationStartedAt.current) / 1000)) : DEVICE_TEST.animationMs / 1000
  return (
    <div className="device-test-page">
      <section className="panel device-intro">
        <div className="device-heading"><Smartphone size={21} /><h2>A baseline for this device</h2><span className="mini-tag">LOCAL TEST</span></div>
        <p>Measure this page’s startup, Tracker, Pacer v4, Prism Small v4, and a 15-second animated match. The same inputs and serve seed run on every device. Your report stays in this browser until you copy it.</p>
        <p className="device-note">Keep the page visible and the phone in the same orientation. During animated play, leave the court in view without scrolling. Repeat with the same power-saving settings when comparing results.</p>
        {startup.build.mode !== 'production' && <p className="device-warning">Development build: useful for checking the test. Use a production preview for the phone baseline.</p>}
      </section>
      <section className="panel device-config">
        <fieldset disabled={busy}>
          <div className="device-fields">
            <label>Phone or computer name
              <input value={device} onChange={(event) => setDevice(event.target.value)} placeholder="e.g. iPhone 15 · Safari" maxLength={100} autoComplete="off" />
              <small>Browsers cannot reliably identify the exact phone model.</small>
            </label>
            <label>This page visit
              <select value={visit} onChange={(event) => setVisit(event.target.value)}>
                <option value="unspecified">Not specified</option>
                <option value="first-visit">First visit</option>
                <option value="repeat-load">Repeat load</option>
              </select>
              <small>Your label, not a verified cache state. Reload to measure startup again.</small>
            </label>
          </div>
          <div className="device-actions">
            <button className="import-btn" disabled={!ready} onClick={() => void start()}><Play size={16} /> Run device test</button>
            <span>{ready ? 'About 20 seconds · no settings change' : loadingComplete ? 'A required model could not be loaded.' : 'Waiting for model loading to finish…'}</span>
          </div>
        </fieldset>
        <div className="device-startup">
          <span>Playable after <b>{ms(startup.arenaReadyMs)}</b></span>
          <span>Library loaded after <b>{ms(startup.libraryReadyMs)}</b></span>
        </div>
        <p className="device-note">Startup is measured from navigation on this page load. It is reused if you run the test again without reloading. Copy your report before reloading for the next visit.</p>
        {startup.hiddenDuringStartup && <p className="device-warning">This page was hidden during startup. Reload while keeping it visible for a comparable loading measurement.</p>}
        {startup.failures.length > 0 && <p className="device-warning">Loading reported an error. The report will include it; reload for a clean startup comparison.</p>}
      </section>
      {busy && <div ref={status} className="device-progress panel" role="status">
        <Activity size={18} />
        <div><strong>{stage === 'animation' ? `Animated match · ${animationSeconds}s remaining` : `Inference · ${progress.modelName}`}</strong>
          <small>{stage === 'animation' ? 'Keep the court visible. Controls are fixed for this test.' : `${progress.completed} / ${progress.total} batches · warmed-up model execution`}</small></div>
        <button className="secondary-btn" onClick={() => abort.current?.abort(new Error('Test cancelled. No partial report was saved.'))}><Square size={14} /> Cancel</button>
      </div>}
      {stage === 'animation' && <fieldset ref={preview} className="device-preview" disabled aria-label="Fixed benchmark match">
        <Arena models={models} ids={matchIds} m={m} ready={ready} pick={() => {}} swap={() => {}} />
      </fieldset>}
      {message && <p className="message" role="status">{message}</p>}
      {report && <section className="panel device-results">
        <div className="device-heading"><Check size={20} /><h2>Device report ready</h2><span className="mini-tag">COMPLETE</span></div>
        <p className="device-note">{report.device} · {new Date(report.measuredAt).toLocaleString()} · {report.startup.build.mode}</p>
        <div className="device-metrics">
          <div><span>Playable after</span><strong>{ms(report.startup.arenaReadyMs)}</strong><small>This page load</small></div>
          <div><span>Frame interval · p95</span><strong>{ms(report.animation.frameIntervalsMs?.p95)}</strong><small>Animation callback spacing</small></div>
          <div><span>Gaps over 50 ms</span><strong>{report.animation.gapsOver50Ms}</strong><small>During the animated sample</small></div>
        </div>
        <div className="device-table-wrap"><table className="device-table">
          <thead><tr><th>Weight model</th><th>Median decision time</th><th>Parameters</th></tr></thead>
          <tbody>{report.inference.map((entry) => <tr key={entry.id}><td>{entry.name}</td><td>{entry.batchAverageDecisionUs.median.toFixed(2)} µs</td><td>{entry.params.toLocaleString()}</td></tr>)}</tbody>
        </table></div>
        <p className="device-note">Decision times are medians of batch averages, including the inference wrapper. Frame intervals describe browser callback timing, not measured display frames or touch latency. Intentional pauses between rallies are reported separately.</p>
        {report.inference.some((entry) => entry.zeroDurationBatches > 0) && <p className="device-warning">Some inference batches were below this browser’s timer resolution. Their zero readings are included in the report; do not interpret them as zero execution cost.</p>}
        <div className="device-actions">
          <button className="import-btn" onClick={() => void copyReport()}><Copy size={16} /> Copy report</button>
          <button className="secondary-btn" aria-expanded={showReport} onClick={() => setShowReport((value) => !value)}>{showReport ? 'Hide report' : 'Show report'}</button>
        </div>
        {copyMessage && <p className="device-copy-message" role="status">{copyMessage}</p>}
        {showReport && <label className="device-raw-label">Report text<textarea ref={reportBox} readOnly value={reportText} aria-label="Device test report" spellCheck={false} /></label>}
      </section>}
    </div>
  )
}

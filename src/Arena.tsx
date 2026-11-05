import {
  Activity,
  ArrowLeft,
  ArrowLeftRight,
  ArrowRight,
  AudioLines,
  ChevronDown,
  CircleDot,
  Code2,
  Cpu,
  Dna,
  FastForward,
  Gamepad2,
  Keyboard,
  List,
  Maximize,
  Minus,
  Minimize,
  Pause,
  Play,
  RotateCcw,
  Shuffle,
  SkipBack,
  SkipForward,
  SlidersHorizontal,
  Waves,
} from 'lucide-react'
import { useLayoutEffect, useRef, useState } from 'react'
import {
  choice,
  endedMatch,
  endedRally,
  RALLIES,
  RALLY_TICK_LIMIT,
  type Pair,
  type Side,
  type Frame,
} from './rules'
import { type Controller } from './models'
import type { Match } from './useMatch'
import { architectureLabel } from './Architecture'
import { HumanControls, useCourtPointerControls } from './HumanControls'
import { interpolate, progress } from './presentation'
import { ControllerBadge, controllerOptionLabel } from './ControllerBadge'
import { DIVISIONS, smallestDivision } from './competition'

export const pad = (n: number) => String(n).padStart(2, '0')
export function ModelIcon({ id, size = 20 }: { id: string; size?: number }) {
  const Icon =
    id === 'keyboard'
      ? Keyboard
      : id.startsWith('script-')
        ? Code2
        : id === 'challenger'
          ? Dna
          : id === 'tracker'
            ? CircleDot
            : id === 'anticipator'
              ? Waves
              : Cpu
  return <Icon size={size} strokeWidth={1.7} />
}
function eventText(f: Frame, names: Pair<string>) {
  if (f.event === 'serve') return 'Rally ' + pad(f.rally) + ' served'
  if (f.event === 'point') return names[f.actor!] + ' scores'
  if (f.event === 'draw')
    return f.drawReason === 'loop' ? 'Rally drawn · repeating loop' : `Rally drawn at ${RALLY_TICK_LIMIT.toLocaleString()} ticks`
  return names[f.actor!] + ' returns the ball'
}
function Court({ m, names, controllers, ready, humanSides, bottom }: {
  m: Match; names: Pair<string>; controllers: Pair<Controller | undefined>; ready: boolean; humanSides: Pair<boolean>; bottom: Side
}) {
  const f = m.frame
  const { active, ...input } = useCourtPointerControls(m, humanSides)
  const field = useRef<SVGSVGElement>(null)
  const court = useRef<HTMLElement>(null)
  const fullscreenControl = useRef<HTMLButtonElement>(null)
  const [fullscreen, setFullscreen] = useState(false)
  function exitFullscreen() {
    if (document.fullscreenElement === court.current)
      void document.exitFullscreen().catch(() => { /* The viewport mode can still be closed. */ })
    setFullscreen(false)
  }
  function enterFullscreen() {
    setFullscreen(true)
    // Keep a viewport-filling mode when native fullscreen is unavailable or denied.
    void court.current?.requestFullscreen?.().catch(() => {})
  }
  useLayoutEffect(() => {
    if (!fullscreen) return
    const scrollX = window.scrollX, scrollY = window.scrollY
    const previousFocus = document.activeElement as HTMLElement | null
    document.documentElement.classList.add('game-fullscreen')
    fullscreenControl.current?.focus({ preventScroll: true })
    function change() {
      if (document.fullscreenElement !== court.current) setFullscreen(false)
    }
    function keydown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        event.preventDefault()
        exitFullscreen()
      }
      if (event.key !== 'Tab') return
      const buttons = court.current!.querySelectorAll<HTMLButtonElement>('button:not(:disabled)')
      const first = buttons[0], last = buttons[buttons.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('fullscreenchange', change)
    document.addEventListener('keydown', keydown)
    return () => {
      document.removeEventListener('fullscreenchange', change)
      document.removeEventListener('keydown', keydown)
      document.documentElement.classList.remove('game-fullscreen')
      previousFocus?.focus({ preventScroll: true })
      window.scrollTo(scrollX, scrollY)
    }
  }, [fullscreen])
  useLayoutEffect(() => {
    const svg = field.current!
    const paddles = svg.querySelectorAll<SVGRectElement>('.paddle')
    const balls = svg.querySelectorAll<SVGCircleElement>('.moving-ball')
    const trail = svg.querySelectorAll<SVGCircleElement>('.ball-trail')
    let raf = 0
    function draw(now: number) {
      const p = m.presentation.current, amount = progress(p, now)
      const position = interpolate(p.from, p.to, amount)
      for (const side of [0, 1] as const)
        paddles[side].setAttribute('y', String(position.paddles[side] - 12))
      for (const ball of balls) {
        ball.setAttribute('cx', String(position.x))
        ball.setAttribute('cy', String(position.y))
      }
      for (let i = 0; i < trail.length; i++) {
        const at = p.trail.length - 8 + i
        const from = p.trail[at], to = p.trail[at + 1]
        const visible = from && to && from.rally === p.to.rally && to.rally === p.to.rally
        trail[i].setAttribute('opacity', visible ? String(0.06 + i * 0.066) : '0')
        if (visible) {
          const point = interpolate(from, to, amount)
          trail[i].setAttribute('cx', String(point.x))
          trail[i].setAttribute('cy', String(point.y))
        }
      }
      if (p.animate && amount < 1) raf = requestAnimationFrame(draw)
    }
    draw(performance.now())
    return () => cancelAnimationFrame(raf)
  }, [m.presentation, m.tick, m.seed, m.playing, m.speed])
  const replay = m.tick < m.history.length - 1
  const status = !ready
    ? 'Loading'
    : endedMatch(f)
      ? 'Full time'
      : replay
        ? 'Replay'
        : m.playing
          ? 'Live'
          : f.tick
            ? 'Paused'
            : 'Ready'
  return (
    <section {...input} ref={court}
      className={'court court-vertical' + (active ? ' human-controlled' : '') + (fullscreen ? ' court-fullscreen' : '')}
      role={fullscreen ? 'dialog' : undefined} aria-modal={fullscreen || undefined} aria-label="Pong court">
      <div className="court-top">
        <span className="court-id">
          <span className="bracket">[</span> COURT 01 <span className="bracket">]</span>
        </span>
        <span className="court-line" />
        <span className="court-version">NEURAL PONG</span>
        <span className={'status ' + (status === 'Live' ? 'live' : '')}>
          <i />
          {status}
        </span>
        <div className="court-actions">
          {fullscreen && <>
            <button className="court-control" disabled={!ready} onClick={m.toggle}
              aria-label={m.playing ? 'Pause match' : endedMatch(f) ? 'Replay match' : 'Play match'}
              title={m.playing ? 'Pause match' : endedMatch(f) ? 'Replay match' : 'Play match'}>
              {m.playing ? <Pause size={18} /> : <Play size={18} />}
            </button>
            <button className="court-control" onClick={() => m.reset()} aria-label="Restart match" title="Restart match">
              <RotateCcw size={18} />
            </button>
          </>}
          <button ref={fullscreenControl} className="court-control" onClick={fullscreen ? exitFullscreen : enterFullscreen}
            aria-label={fullscreen ? 'Exit full screen' : 'Enter full screen'} aria-pressed={fullscreen}
            title={fullscreen ? 'Exit full screen (Esc)' : 'Enter full screen'}>
            {fullscreen ? <Minimize size={18} /> : <Maximize size={18} />}
            <span>{fullscreen ? 'Exit' : 'Full screen'}</span>
          </button>
        </div>
      </div>
      <div className="scoreboard">
        <div className="score-player">
          <small>
            <i className="p1-dot" />
            PLAYER 01{bottom === 0 ? ' · BOTTOM' : ' · TOP'}
          </small>
          <strong>{names[0]}</strong>
          <ControllerBadge model={controllers[0]} compact />
        </div>
        <div className="big-score" aria-label={'Score ' + f.points.join(' to ')}>
          <b>{pad(f.points[0])}</b>
          <span>:</span>
          <b>{pad(f.points[1])}</b>
        </div>
        <div className="score-player right">
          <small>
            PLAYER 02{bottom === 1 ? ' · BOTTOM' : ' · TOP'}
            <i className="p2-dot" />
          </small>
          <strong>{names[1]}</strong>
          <ControllerBadge model={controllers[1]} compact />
        </div>
      </div>
      <div className="court-field">
        <svg
          className="field"
          ref={field}
          viewBox="0 0 160 360"
          preserveAspectRatio="none"
          role="img"
          aria-label={`Live court: ${bottom === 0 ? 'mint' : 'orange'} bottom paddle, ${bottom === 0 ? 'orange' : 'mint'} top paddle and ball`}
        >
          <defs>
            <pattern id="grid" width="15" height="16" patternUnits="userSpaceOnUse">
              <path d="M15 0H0V16" fill="none" stroke="#9ebdb2" strokeWidth=".28" opacity=".18" />
              <circle r=".55" fill="#bdd1c5" opacity=".35" />
            </pattern>
            <filter id="softball" x="-100%" y="-100%" width="300%" height="300%">
              <feGaussianBlur stdDeviation="2" />
            </filter>
          </defs>
          <g transform={bottom === 0 ? 'matrix(0 -1 1 0 0 360)' : 'matrix(0 1 1 0 0 0)'}>
            <rect width="360" height="160" fill="url(#grid)" />
            <rect
              x=".4"
              y=".4"
              width="359.2"
              height="159.2"
              rx="1"
              fill="none"
              stroke="#aec7b6"
              strokeWidth=".6"
              opacity=".45"
            />
            <circle
              cx="180"
              cy="80"
              r="26"
              stroke="#bad2c2"
              opacity=".18"
              fill="none"
              strokeWidth=".65"
            />
            <path
              d="M180 0V160"
              stroke="#bdd2c1"
              strokeWidth=".8"
              strokeDasharray="2.2 2.7"
              opacity=".62"
            />
            <path
              d="M.7 7V.7H7 M353 .7h6.3V7 M.7 153v6.3H7 M353 159.3h6.3V153"
              fill="none"
              stroke="#b4c7b7"
              strokeWidth="1"
            />
            <rect x="6" y="68" width="5" height="24" rx="1.2" className="paddle left" />
            <rect
              x="349"
              y="68"
              width="5"
              height="24"
              rx="1.2"
              className="paddle right"
            />
            {Array.from({ length: 7 }, (_, i) => (
              <circle
                key={i}
                className="ball-trail"
                cx="180"
                cy="80"
                r={0.7 + i * 0.18}
                fill="#e8f8c9"
                opacity="0"
              />
            ))}
            <circle className="moving-ball" cx="180" cy="80" r="3.7" fill="#dafa9c" opacity=".44" filter="url(#softball)" />
            <circle className="moving-ball" cx="180" cy="80" r="2.55" fill="#f1ffd9" />
          </g>
        </svg>
      </div>
      {fullscreen && (humanSides[0] || humanSides[1]) && <p className="court-instruction">
        <span><ArrowLeft size={14} /> Hold left</span>
        <span>Hold right <ArrowRight size={14} /></span>
      </p>}
      {f.energy && m.rules.energyMax && (
        <div className="court-energy" aria-label="Burst energy">
          {([0, 1] as const).map((side) => (
            <div key={side} className={'energy-player side-' + side}>
              <span>P{side + 1} · BURST <b>{Math.round(f.energy![side] / m.rules.energyMax! * 100)}%</b></span>
              <div role="meter" aria-label={'Player ' + (side + 1) + ' burst energy'}
                aria-valuemin={0} aria-valuemax={m.rules.energyMax} aria-valuenow={f.energy![side]}>
                <i style={{ width: f.energy![side] / m.rules.energyMax! * 100 + '%' }} />
              </div>
            </div>
          ))}
        </div>
      )}
      <div className="court-foot">
        <span>
          <i />
          RALLY {pad(f.rally)} <em>/ {pad(RALLIES)}</em>
        </span>
        <span>
          {endedMatch(f)
            ? f.points[0] === f.points[1]
              ? 'MATCH DRAWN'
              : (f.points[0] > f.points[1] ? 'P1' : 'P2') + ' WINS'
            : endedRally(f)
              ? eventText(f, names).toUpperCase()
              : pad(f.returns) + ' RETURNS'}
        </span>
        <span className="tick-readout">TICK {String(f.tick).padStart(4, '0')}</span>
      </div>
    </section>
  )
}
function Transport({ m, ready }: { m: Match; ready: boolean }) {
  const end = m.history.length - 1,
    f = m.frame
  return (
    <section className="panel transport" aria-label="Replay controls">
      <div className="transport-head">
        <h3>
          <Activity size={16} /> Playback
        </h3>
        <span className="mono">
          FRAME {String(m.tick).padStart(4, '0')} / {String(end).padStart(4, '0')}
        </span>
      </div>
      <input
        className="timeline"
        aria-label="Replay frame"
        type="range"
        min={0}
        max={Math.max(1, end)}
        value={m.tick}
        disabled={end === 0}
        onChange={(e) => m.seek(Number(e.target.value))}
        style={{
          backgroundImage:
            'linear-gradient(to right, var(--forest) ' +
            (end ? (100 * m.tick) / end : 0) +
            '%, var(--line) 0%)',
        }}
      />
      <div className="transport-row">
        <button className="play-btn" disabled={!ready} onClick={m.toggle}>
          {m.playing ? (
            <Pause size={15} fill="currentColor" />
          ) : (
            <Play size={15} fill="currentColor" />
          )}
          {m.playing ? 'Pause' : endedMatch(f) ? 'Replay match' : 'Play match'}
        </button>
        <button
          className="icon-btn"
          title="Previous tick"
          aria-label="Previous tick"
          disabled={!m.tick}
          onClick={() => m.seek(m.tick - 1)}
        >
          <SkipBack size={17} />
        </button>
        <button
          className="icon-btn"
          title="Next tick"
          aria-label="Next tick"
          disabled={!ready || endedMatch(f)}
          onClick={m.step}
        >
          <SkipForward size={17} />
        </button>
        <span className="divider" />
        <label className="speed">
          <FastForward size={14} />
          <select
            aria-label="Playback speed"
            value={m.speed}
            onChange={(e) => m.setSpeed(Number(e.target.value))}
          >
            {[0.5, 1, 2, 4].map((x) => (
              <option key={x} value={x}>
                {x}×
              </option>
            ))}
          </select>
        </label>
        <button
          className="restart"
          aria-label="Restart"
          onClick={() => m.reset()}
          title="Restart the same five serves"
        >
          <RotateCcw size={15} />
          <span>Restart</span>
        </button>
      </div>
    </section>
  )
}
interface Props {
  m: Match
  models: Controller[]
  ids: Pair<string>
  pick: (index: Side, id: string) => void
  swap: () => void
  ready: boolean
}
export function Arena({ m, models, ids, pick, swap, ready }: Props) {
  const [side, setSide] = useState<Side>(0)
  const humanSides: Pair<boolean> = [ids[0] === 'keyboard', ids[1] === 'keyboard']
  const bottom: Side = humanSides[1] && !humanSides[0] ? 1 : 0
  const actions = ['Left', 'Stay', 'Right', 'Burst left', 'Burst right'].slice(0, m.rules.outputSize)
  const controllers = ids.map((id) => models.find((model) => model.id === id)) as Pair<Controller | undefined>
  const names = ids.map(
    (id, i) => controllers[i]?.name ?? id[0].toUpperCase() + id.slice(1),
  ) as Pair<string>
  const groups = [
    { name: 'Human', models: models.filter((c) => c.type === 'human') },
    { name: 'Scripts', models: models.filter((c) => c.type === 'script') },
    ...DIVISIONS.map((d) => ({ name: d.name + ' weights',
      models: models.filter((c) => c.type === 'weights' && smallestDivision(c) === d.id) })),
    { name: 'Unclassified weights', models: models.filter((c) => c.type === 'weights' && !smallestDivision(c)) },
  ].filter((group) => group.models.length)
  const inspected = controllers[side]
  const discrete = inspected?.type === 'human' || inspected?.type === 'script'
  const scores = m.frame.scores[side]
  const chosen = scores ? choice(scores) : -1
  const max = Math.max(0.001, ...(scores ?? []).map(Math.abs))
  const recent: Frame[] = []
  for (let at = m.tick; at >= 0 && recent.length < 5; at--)
    if (m.history[at].event) recent.push(m.history[at])
  return (
    <div className="arena-grid">
      <div className="main-column">
        <Court m={m} names={names} controllers={controllers} ready={ready} humanSides={humanSides} bottom={bottom} />
        <HumanControls m={m} humanSides={humanSides} />
        <Transport m={m} ready={ready} />
        <section className="panel feed">
          <div className="panel-heading">
            <h2>
              <List size={18} /> Match log
            </h2>
            <span className="mini-tag">THIS MATCH</span>
          </div>
          {recent.map((f) => (
            <button
              key={f.tick}
              onClick={() => m.seek(f.tick)}
              className="event-row"
              title="Jump to this frame"
            >
              <span className="event-marker" data-event={f.event} />
              <span>{eventText(f, names)}</span>
              <small>TICK {String(f.tick).padStart(4, '0')}</small>
              <ArrowRight size={14} />
            </button>
          ))}
        </section>
      </div>
      <aside className="side-column">
        <section className="panel matchup">
          <div className="panel-heading">
            <h2>
              <SlidersHorizontal size={17} /> Match setup
            </h2>
            <span className="mini-tag">1 VS 1</span>
          </div>
          <div className="picker-stack">
            {([0, 1] as const).map((i) => (
              <div key={i}>
                {i === 1 && (
                  <div className="swap-row">
                    <span />
                    <button aria-label="Swap players" title="Swap players" onClick={swap}>
                      <ArrowLeftRight size={13} /> Swap
                    </button>
                    <span />
                  </div>
                )}
                <label className={'pick side-' + i}>
                  <span className="pick-label">
                    <i />
                    PLAYER 0{i + 1}
                    <span>/</span>
                    {i === bottom ? 'BOTTOM' : 'TOP'}
                  </span>
                  <span className="pick-body">
                    <span className="model-icon">
                      <ModelIcon id={ids[i]} />
                    </span>
                    <span className="pick-name">
                      <b>{names[i]}</b>
                      <small>{controllers[i]?.kind ?? 'Loading controller'}</small>
                    </span>
                    <ChevronDown size={15} />
                  </span>
                  <select
                    aria-label={i === bottom ? 'Bottom controller' : 'Top controller'}
                    value={ids[i]}
                    onChange={(e) => pick(i, e.target.value)}
                  >
                    {groups.map((group) => (
                      <optgroup key={group.name} label={group.name}>
                        {group.models.map((c) => (
                          <option key={c.id} value={c.id}>{controllerOptionLabel(c)}</option>
                        ))}
                      </optgroup>
                    ))}
                  </select>
                </label>
                <div className="pick-details">
                  <ControllerBadge model={controllers[i]} />
                  {controllers[i]?.type === 'weights' && (
                    <div className="pick-architecture">{architectureLabel(controllers[i])}</div>
                  )}
                </div>
              </div>
            ))}
          </div>
          <p className="controller-class-note">Weight classes follow parameter and compute budgets: Nano, Small, Standard, Open. Size is not a skill rating.</p>
          <div className="setup-note">
            <Gamepad2 size={15} />
            {ids.includes('keyboard')
              ? ids.every((x) => x === 'keyboard')
                ? 'Hold left/right in your half · Bottom: A/D · Top: ←/→'
                : 'Hold the left or right half to move · Lift to stop · A/D or ←/→'
              : 'Select Human to play against a script or weights.'}
          </div>
          <div className="seed-row">
            <span>SEED {String(m.seed).padStart(4, '0')}</span>
            <button onClick={m.newServes}>
              <Shuffle size={13} /> New serves
            </button>
          </div>
        </section>
        <section className="panel inspector">
          <div className="panel-heading">
            <h2>
              <AudioLines size={17} /> {inspected?.type === 'script' ? 'Script decision' : inspected?.type === 'human' ? 'Human input' : 'Action scores'}
            </h2>
            <div className="segmented">
              {([0, 1] as const).map((i) => (
                <button
                  key={i}
                  aria-pressed={side === i}
                  aria-label={'Inspect player ' + (i + 1)}
                  onClick={() => setSide(i)}
                >
                  P{i + 1}
                </button>
              ))}
            </div>
          </div>
          <div className="inspect-body">
            <div className="inspect-source">
              <i className={side === 0 ? 'p1-dot' : 'p2-dot'} />
              {names[side]}
              <span>{discrete ? 'CHOSEN ACTION' : 'RAW OUTPUT'}</span>
            </div>
            <div className="inspect-controller"><ControllerBadge model={inspected} /></div>
            {actions.map((a, i) => {
              const Icon = [ArrowLeft, Minus, ArrowRight, ArrowLeft, ArrowRight][i],
                v = scores?.[i] ?? 0
              return (
                <div className={'score-row ' + (chosen === i ? 'chosen' : '')} key={a}>
                  <span>
                    <Icon size={14} />
                    {a}
                  </span>
                  <div className={'score-track' + (discrete ? ' discrete' : '')}>
                    <i
                      style={{
                        width: (Math.abs(v) / max) * (discrete ? 100 : 50) + '%',
                        left: discrete ? '0%' : v < 0 ? 50 - (Math.abs(v) / max) * 50 + '%' : '50%',
                        background: side === 0 ? '#329c79' : '#d07849',
                      }}
                    />
                  </div>
                  <output>{scores ? discrete ? v.toFixed(0) : (v >= 0 ? '+' : '') + v.toFixed(2) : '—'}</output>
                </div>
              )
            })}
            <div className="decision">
              <span>SELECTED MOVE</span>
              <b>{chosen >= 0 ? actions[chosen] : 'Awaiting tick'}</b>
            </div>
            {inspected?.type === 'script' && <p className="controller-decision-note">One-hot decision: 1 marks the action chosen by the script; 0 marks the others.</p>}
            <div className="trace">
              <span>LAST 40 DECISIONS</span>
              <div
                className="trace-bars"
                role="img"
                aria-label={'Movement history for ' + names[side] + ', from oldest to newest'}
              >
                {m.history.slice(Math.max(0, m.tick - 39), m.tick + 1).map((f, i, arr) => {
                  const score = f.scores[side],
                    move = score ? choice(score) : null
                  return (
                    <i
                      key={f.tick}
                      title={'Tick ' + f.tick + ' · ' + (move === null ? 'Serve' : actions[move])}
                      style={{
                        gridColumn: 41 - arr.length + i,
                        gridRow: move === null ? 2 : move === 3 ? 1 : move === 4 ? 3 : move + 1,
                        background:
                          move === null ? 'transparent' : side === 0 ? '#488866' : '#be7648',
                      }}
                    />
                  )
                })}
              </div>
            </div>
          </div>
        </section>
      </aside>
    </div>
  )
}

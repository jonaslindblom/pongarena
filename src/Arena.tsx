import {
  ArrowLeft,
  ArrowLeftRight,
  ArrowRight,
  ChevronDown,
  CircleDot,
  Code2,
  Cpu,
  Keyboard,
  Maximize,
  Minimize,
  Pause,
  Play,
  RotateCcw,
  SlidersHorizontal,
} from 'lucide-react'
import { useLayoutEffect, useRef, useState } from 'react'
import {
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
import { BrainPanel } from './BrainPanel'
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
        : id === 'tracker'
          ? CircleDot
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
type Mode = 'play' | 'watch'
function playLabel(m: Match, mode: Mode) {
  if (m.playing) return 'Pause'
  if (endedMatch(m.frame)) return mode === 'play' ? 'Play again' : 'Watch again'
  if (m.tick) return 'Resume'
  return mode === 'play' ? 'Play' : 'Watch'
}
function Court({ m, names, controllers, ready, humanSides, bottom, mode }: {
  m: Match; names: Pair<string>; controllers: Pair<Controller | undefined>; ready: boolean; humanSides: Pair<boolean>; bottom: Side; mode: Mode
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
  }, [m.presentation, m.tick, m.seed, m.playing])
  const status = !ready
    ? 'Loading'
    : endedMatch(f)
      ? 'Full time'
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
              aria-label={playLabel(m, mode)} title={playLabel(m, mode)}>
              {m.playing ? <Pause size={18} /> : <Play size={18} />}
            </button>
            <button className="court-control" disabled={!ready} onClick={m.restart} aria-label="Restart match" title="Restart match (R)">
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
function MatchControls({ m, ready, mode }: { m: Match; ready: boolean; mode: Mode }) {
  function showCourt(button: HTMLButtonElement) {
    if (window.matchMedia('(max-width: 720px), (pointer: coarse)').matches)
      button.closest('.main-column')?.querySelector('.court')?.scrollIntoView({ block: 'start' })
  }
  return (
    <section className="panel match-controls" aria-label="Match controls">
      <button className="play-btn" disabled={!ready} onClick={(event) => {
        m.toggle()
        if (!m.playing) showCourt(event.currentTarget)
      }}>
        {m.playing ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}
        {playLabel(m, mode)}
      </button>
      <button className="secondary-btn" disabled={!ready} onClick={(event) => {
        m.restart()
        showCourt(event.currentTarget)
      }} title="Restart the same serves (R)">
        <RotateCcw size={16} /> Restart <kbd>R</kbd>
      </button>
      <span>{mode === 'play' ? 'Move with A/D or ←/→' : 'Five rallies per match'}</span>
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
  mode?: Mode
}
export function Arena({ m, models, ids, pick, swap, ready, mode = 'watch' }: Props) {
  const humanSides: Pair<boolean> = [ids[0] === 'keyboard', ids[1] === 'keyboard']
  const bottom: Side = humanSides[1] && !humanSides[0] ? 1 : 0
  const controllers = ids.map((id) => models.find((model) => model.id === id)) as Pair<Controller | undefined>
  const names = ids.map(
    (id, i) => controllers[i]?.name ?? id[0].toUpperCase() + id.slice(1),
  ) as Pair<string>
  const groups = [
    { name: 'Scripts', models: models.filter((c) => c.type === 'script') },
    ...DIVISIONS.map((d) => ({ name: d.name + ' weights',
      models: models.filter((c) => c.type === 'weights' && smallestDivision(c) === d.id) })),
    { name: 'Unclassified weights', models: models.filter((c) => c.type === 'weights' && !smallestDivision(c)) },
  ].filter((group) => group.models.length)
  return (
    <div className="arena-grid">
      <div className="main-column">
        <MatchControls m={m} ready={ready} mode={mode} />
        <Court m={m} names={names} controllers={controllers} ready={ready} humanSides={humanSides} bottom={bottom} mode={mode} />
        <HumanControls m={m} humanSides={humanSides} />
      </div>
      <aside className="side-column">
        <section className="panel matchup">
          <div className="panel-heading">
            <h2>
              <SlidersHorizontal size={17} /> {mode === 'play' ? 'Choose an opponent' : 'Choose the players'}
            </h2>
          </div>
          <div className="picker-stack">
            {([0, 1] as const).filter((i) => mode === 'watch' || i !== bottom).map((i) => (
              <div key={i}>
                {mode === 'watch' && i === 1 && (
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
                    {mode === 'play' ? 'OPPONENT' : `PLAYER 0${i + 1}`}
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
                    aria-label={mode === 'play' ? 'Opponent' : i === bottom ? 'Bottom controller' : 'Top controller'}
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
                </div>
              </div>
            ))}
          </div>
        </section>
        <BrainPanel m={m} controllers={controllers} bottom={bottom} />
      </aside>
    </div>
  )
}

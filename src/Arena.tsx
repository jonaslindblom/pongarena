import {
  Activity,
  ArrowDown,
  ArrowLeftRight,
  ArrowRight,
  ArrowUp,
  AudioLines,
  ChevronDown,
  CircleDot,
  Cpu,
  Dna,
  FastForward,
  Gamepad2,
  Keyboard,
  List,
  Minus,
  Pause,
  Play,
  RotateCcw,
  Shuffle,
  SkipBack,
  SkipForward,
  SlidersHorizontal,
  Waves,
} from 'lucide-react'
import { useState } from 'react'
import {
  ACTIONS,
  choice,
  endedMatch,
  endedRally,
  RALLIES,
  type Pair,
  type Side,
  type Frame,
} from './game'
import { type Controller } from './models'
import type { Match } from './useMatch'

export const pad = (n: number) => String(n).padStart(2, '0')
export function ModelIcon({ id, size = 20 }: { id: string; size?: number }) {
  const Icon =
    id === 'keyboard'
      ? Keyboard
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
  if (f.event === 'draw') return 'Rally drawn at 800 ticks'
  return names[f.actor!] + ' returns the ball'
}
function Court({ m, names, ready }: { m: Match; names: Pair<string>; ready: boolean }) {
  const f = m.frame
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
  const trail = m.history.slice(Math.max(0, m.tick - 7), m.tick).filter((p) => p.rally === f.rally)
  return (
    <section className="court" aria-label="Pong court">
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
      </div>
      <div className="scoreboard">
        <div className="score-player">
          <small>
            <i className="p1-dot" />
            PLAYER 01
          </small>
          <strong>{names[0]}</strong>
        </div>
        <div className="big-score" aria-label={'Score ' + f.points.join(' to ')}>
          <b>{pad(f.points[0])}</b>
          <span>:</span>
          <b>{pad(f.points[1])}</b>
        </div>
        <div className="score-player right">
          <small>
            PLAYER 02
            <i className="p2-dot" />
          </small>
          <strong>{names[1]}</strong>
        </div>
      </div>
      <svg
        className="field"
        viewBox="0 0 360 160"
        preserveAspectRatio="none"
        role="img"
        aria-label="Live court with a mint left paddle, orange right paddle and ball"
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
        <rect x="6" y={f.paddles[0] - 12} width="5" height="24" rx="1.2" className="paddle left" />
        <rect
          x="349"
          y={f.paddles[1] - 12}
          width="5"
          height="24"
          rx="1.2"
          className="paddle right"
        />
        {trail.map((p, i) => (
          <circle
            key={p.tick}
            cx={p.x}
            cy={p.y}
            r={0.7 + i * 0.18}
            fill="#e8f8c9"
            opacity={0.06 + i * 0.066}
          />
        ))}
        <circle cx={f.x} cy={f.y} r="3.7" fill="#dafa9c" opacity=".44" filter="url(#softball)" />
        <circle cx={f.x} cy={f.y} r="2.55" fill="#f1ffd9" />
      </svg>
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
          background:
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
  const names = ids.map(
    (id) => models.find((x) => x.id === id)?.name ?? id[0].toUpperCase() + id.slice(1),
  ) as Pair<string>
  const scores = m.frame.scores[side]
  const chosen = scores ? choice(scores) : -1
  const max = Math.max(0.001, ...(scores ?? []).map(Math.abs))
  const recent = m.history
    .slice(0, m.tick + 1)
    .filter((f) => f.event)
    .slice(-5)
    .reverse()
  return (
    <div className="arena-grid">
      <div className="main-column">
        <Court m={m} names={names} ready={ready} />
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
                    {i === 0 ? 'LEFT' : 'RIGHT'}
                  </span>
                  <span className="pick-body">
                    <span className="model-icon">
                      <ModelIcon id={ids[i]} />
                    </span>
                    <span className="pick-name">
                      <b>{names[i]}</b>
                      <small>{models.find((x) => x.id === ids[i])?.kind ?? 'Neural model'}</small>
                    </span>
                    <ChevronDown size={15} />
                  </span>
                  <select
                    aria-label={i === 0 ? 'Left controller' : 'Right controller'}
                    value={ids[i]}
                    onChange={(e) => pick(i, e.target.value)}
                  >
                    {models.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            ))}
          </div>
          <div className="setup-note">
            <Gamepad2 size={15} />
            {ids.includes('keyboard')
              ? ids.every((x) => x === 'keyboard')
                ? 'Left: W/S · Right: ↑/↓'
                : 'Move your paddle with W/S or ↑/↓'
              : 'Select Keyboard to play a model.'}
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
              <AudioLines size={17} /> Action scores
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
              <span>{ids[side] === 'keyboard' ? 'KEY INPUT' : 'RAW OUTPUT'}</span>
            </div>
            {ACTIONS.map((a, i) => {
              const Icon = [ArrowUp, Minus, ArrowDown][i],
                v = scores?.[i] ?? 0
              return (
                <div className={'score-row ' + (chosen === i ? 'chosen' : '')} key={a}>
                  <span>
                    <Icon size={14} />
                    {a}
                  </span>
                  <div className="score-track">
                    <i
                      style={{
                        width: (Math.abs(v) / max) * 50 + '%',
                        left: v < 0 ? 50 - (Math.abs(v) / max) * 50 + '%' : '50%',
                        background: side === 0 ? '#329c79' : '#d07849',
                      }}
                    />
                  </div>
                  <output>{scores ? (v >= 0 ? '+' : '') + v.toFixed(2) : '—'}</output>
                </div>
              )
            })}
            <div className="decision">
              <span>SELECTED MOVE</span>
              <b>{chosen >= 0 ? ACTIONS[chosen] : 'Awaiting tick'}</b>
            </div>
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
                      title={'Tick ' + f.tick + ' · ' + (move === null ? 'Serve' : ACTIONS[move])}
                      style={{
                        gridColumn: 41 - arr.length + i,
                        gridRow: move === null ? 2 : move + 1,
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

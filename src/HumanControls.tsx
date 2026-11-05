import { ArrowDown, ArrowUp, Hand, Zap } from 'lucide-react'
import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { endedMatch, type Pair, type Side } from './rules'
import { courtTargetAt, heldHumanAction, humanSideAt, type HeldDirection } from './humanInput'
import type { Match } from './useMatch'

type Props = { m: Match; humanSides: Pair<boolean> }
type CapturedPointer = { side: Side; element: Element }

function releaseCapture(element: Element, pointerId: number) {
  if (element.hasPointerCapture(pointerId)) element.releasePointerCapture(pointerId)
}

export function useCourtPointerControls(m: Match, humanSides: Pair<boolean>) {
  const latest = useRef(m)
  latest.current = m
  const pointers = useRef(new Map<number, CapturedPointer>())
  const enabled = m.playing && m.tick === m.history.length - 1 && !endedMatch(m.frame)
  const left = humanSides[0], right = humanSides[1]

  useEffect(() => {
    function clear() {
      const held = [...pointers.current]
      pointers.current.clear()
      for (const [id, pointer] of held) {
        latest.current.setHumanTarget(pointer.side, null)
        releaseCapture(pointer.element, id)
      }
    }
    const hidden = () => { if (document.hidden) clear() }
    window.addEventListener('blur', clear)
    document.addEventListener('visibilitychange', hidden)
    return () => {
      clear()
      window.removeEventListener('blur', clear)
      document.removeEventListener('visibilitychange', hidden)
    }
  }, [enabled, left, right, m.seed, m.history])

  function move(event: ReactPointerEvent<SVGSVGElement>, side: Side) {
    const bounds = event.currentTarget.getBoundingClientRect()
    latest.current.setHumanTarget(side, courtTargetAt(event.clientY - bounds.top, bounds.height))
  }
  function release(event: ReactPointerEvent<SVGSVGElement>) {
    const held = pointers.current.get(event.pointerId)
    if (!held) return
    pointers.current.delete(event.pointerId)
    latest.current.setHumanTarget(held.side, null)
    releaseCapture(held.element, event.pointerId)
  }
  return {
    className: 'field' + (enabled && (left || right) ? ' human-controlled' : ''),
    onPointerDown(event: ReactPointerEvent<SVGSVGElement>) {
      if (!enabled || event.button !== 0) return
      const bounds = event.currentTarget.getBoundingClientRect()
      const side = humanSideAt(event.clientX - bounds.left, bounds.width, humanSides)
      if (side === null || [...pointers.current.values()].some((p) => p.side === side)) return
      event.preventDefault()
      pointers.current.set(event.pointerId, { side, element: event.currentTarget })
      event.currentTarget.setPointerCapture(event.pointerId)
      latest.current.setHumanAction(side, null)
      move(event, side)
    },
    onPointerMove(event: ReactPointerEvent<SVGSVGElement>) {
      const held = pointers.current.get(event.pointerId)
      if (!held || !enabled) return
      event.preventDefault()
      move(event, held.side)
    },
    onPointerUp: release,
    onPointerCancel: release,
    onLostPointerCapture: release,
  }
}

export function HumanControls({ m, humanSides }: Props) {
  const held = useRef(new Map<string, HeldDirection>())
  const boosts = useRef(new Map<string, Side>())
  const pointers = useRef(new Map<number, Element>())
  const taps = useRef(new Map<string, ReturnType<typeof setTimeout>>())
  const latest = useRef(m)
  latest.current = m
  const [pressed, setPressed] = useState(new Set<string>())
  const enabled = m.playing && m.tick === m.history.length - 1 && !endedMatch(m.frame)
  const left = humanSides[0], right = humanSides[1]

  function publish() {
    for (const side of [0, 1] as const) {
      latest.current.setHumanAction(side, heldHumanAction(held.current.values(), side))
      latest.current.setHumanBoost(side, [...boosts.current.values()].includes(side))
    }
    setPressed(new Set([
      ...[...held.current.values()].map((entry) => entry.side + ':' + entry.action),
      ...[...boosts.current.values()].map((side) => 'boost:' + side),
    ]))
  }
  function start(key: string, side: Side, action: 0 | 2) {
    if (!enabled) return
    latest.current.setHumanTarget(side, null)
    held.current.set(key, { side, action })
    publish()
  }
  function stop(key: string) {
    const direction = held.current.delete(key), boost = boosts.current.delete(key)
    if (direction || boost) publish()
  }
  function releasePointer(event: ReactPointerEvent<HTMLButtonElement>) {
    stop('pointer:' + event.pointerId)
    const element = pointers.current.get(event.pointerId)
    pointers.current.delete(event.pointerId)
    if (element) releaseCapture(element, event.pointerId)
  }
  useEffect(() => {
    function clear() {
      held.current.clear()
      boosts.current.clear()
      for (const timer of taps.current.values()) clearTimeout(timer)
      taps.current.clear()
      const captures = [...pointers.current]
      pointers.current.clear()
      for (const [id, element] of captures) releaseCapture(element, id)
      latest.current.setHumanAction(0, null)
      latest.current.setHumanAction(1, null)
      latest.current.setHumanBoost(0, false)
      latest.current.setHumanBoost(1, false)
      setPressed(new Set())
    }
    const hidden = () => { if (document.hidden) clear() }
    window.addEventListener('blur', clear)
    document.addEventListener('visibilitychange', hidden)
    return () => {
      clear()
      window.removeEventListener('blur', clear)
      document.removeEventListener('visibilitychange', hidden)
    }
  }, [enabled, left, right, m.seed, m.history])

  if (!left && !right) return null
  const instruction = m.tick < m.history.length - 1
    ? 'Replay · controls resume when you reach live play.'
    : !m.playing
      ? 'Press Play, then drag the court or hold a button.'
      : left && right
        ? 'Drag your half of the court or hold a button.'
        : 'Drag the court or hold a button.'
  return (
    <section className="panel human-controls" aria-label="Human paddle controls">
      <div className="human-control-heading">
        <Hand size={16} />
        <p>{instruction}</p>
      </div>
      <div className={'human-control-pads' + (left && right ? ' two-players' : '')}>
        {([0, 1] as const).filter((side) => humanSides[side]).map((side) => (
          <div className={'human-control-pad side-' + side} key={side}>
            <div className="human-control-label">
              <span><i className={side === 0 ? 'p1-dot' : 'p2-dot'} /> P{side + 1} · {side === 0 ? 'Left' : 'Right'}</span>
              <small>{left && right ? side === 0 ? 'W / S' : '↑ / ↓' : 'W / S or ↑ / ↓'}</small>
            </div>
            <div className="human-direction-buttons">
              {([0, 2] as const).map((action) => {
                const label = action === 0 ? 'Up' : 'Down'
                const key = side + ':' + action
                const Icon = action === 0 ? ArrowUp : ArrowDown
                return (
                  <button
                    key={action}
                    className="human-direction"
                    disabled={!enabled}
                    aria-label={'Move player ' + (side + 1) + ' ' + label.toLowerCase()}
                    aria-pressed={pressed.has(key)}
                    onPointerDown={(event) => {
                      if (event.button !== 0 || !enabled) return
                      event.preventDefault()
                      event.currentTarget.setPointerCapture(event.pointerId)
                      pointers.current.set(event.pointerId, event.currentTarget)
                      start('pointer:' + event.pointerId, side, action)
                    }}
                    onPointerUp={releasePointer}
                    onPointerCancel={releasePointer}
                    onLostPointerCapture={releasePointer}
                    onKeyDown={(event) => {
                      if (event.key !== ' ' && event.key !== 'Enter') return
                      event.preventDefault()
                      start('key:' + key, side, action)
                    }}
                    onKeyUp={(event) => {
                      if (event.key === ' ' || event.key === 'Enter') {
                        event.preventDefault()
                        stop('key:' + key)
                      }
                    }}
                    onBlur={() => stop('key:' + key)}
                    onClick={(event) => {
                      // Assistive technology can activate a button without a
                      // pointer or held key; give that activation a short press.
                      if (event.detail !== 0 || !enabled) return
                      const tap = 'tap:' + key
                      clearTimeout(taps.current.get(tap))
                      start(tap, side, action)
                      taps.current.set(tap, setTimeout(() => {
                        taps.current.delete(tap)
                        stop(tap)
                      }, 120))
                    }}
                  ><Icon size={20} /><span>{label}</span></button>
                )
              })}
            </div>
            {m.rules.energyMax && (
              <button className="human-boost" disabled={!enabled}
                aria-label={'Burst player ' + (side + 1)} aria-pressed={pressed.has('boost:' + side)}
                onPointerDown={(event) => {
                  if (event.button !== 0 || !enabled) return
                  event.preventDefault()
                  event.currentTarget.setPointerCapture(event.pointerId)
                  pointers.current.set(event.pointerId, event.currentTarget)
                  boosts.current.set('pointer:' + event.pointerId, side)
                  publish()
                }}
                onPointerUp={releasePointer} onPointerCancel={releasePointer} onLostPointerCapture={releasePointer}
                onKeyDown={(event) => {
                  if (event.key !== ' ' && event.key !== 'Enter') return
                  event.preventDefault()
                  boosts.current.set('key:boost:' + side, side)
                  publish()
                }}
                onKeyUp={(event) => {
                  if (event.key === ' ' || event.key === 'Enter') {
                    event.preventDefault()
                    stop('key:boost:' + side)
                  }
                }}
                onBlur={() => stop('key:boost:' + side)}>
                <Zap size={16} /> Hold Burst <small>{(m.frame.energy?.[side] ?? 96) < (m.rules.boostCost ?? 8)
                  ? 'Release Burst to recharge'
                  : (left && right ? side === 0 ? 'Left Shift' : 'Right Shift' : 'Shift') + ' + direction'}</small>
              </button>
            )}
          </div>
        ))}
      </div>
    </section>
  )
}

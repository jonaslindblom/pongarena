import { ArrowLeft, ArrowRight } from 'lucide-react'
import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { endedMatch, type Pair, type Side } from './rules'
import { courtActionAt, heldHumanAction, humanSideAt, type HeldDirection } from './humanInput'
import type { Match } from './useMatch'

type Props = { m: Match; humanSides: Pair<boolean> }
type CapturedPointer = HeldDirection & { element: HTMLElement }

function releaseCapture(element: Element, pointerId: number) {
  if (element.hasPointerCapture(pointerId)) element.releasePointerCapture(pointerId)
}

export function useCourtPointerControls(m: Match, humanSides: Pair<boolean>) {
  const latest = useRef(m)
  latest.current = m
  const pointers = useRef(new Map<number, CapturedPointer>())
  const enabled = m.playing && !endedMatch(m.frame)
  const left = humanSides[0], right = humanSides[1]

  useEffect(() => {
    function clear() {
      const held = [...pointers.current]
      pointers.current.clear()
      for (const [id, pointer] of held) {
        latest.current.setHumanAction(pointer.side, null)
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

  function publish(side: Side) {
    latest.current.setHumanAction(side, heldHumanAction(pointers.current.values(), side))
  }
  function release(event: ReactPointerEvent<HTMLElement>) {
    const held = pointers.current.get(event.pointerId)
    if (!held) return
    pointers.current.delete(event.pointerId)
    publish(held.side)
    releaseCapture(held.element, event.pointerId)
  }
  return {
    active: enabled && (left || right),
    onPointerDown(event: ReactPointerEvent<HTMLElement>) {
      if (!enabled || event.button !== 0 || (event.target as Element).closest('button')) return
      const bounds = event.currentTarget.getBoundingClientRect()
      const side = humanSideAt(event.clientY - bounds.top, bounds.height, humanSides)
      if (side === null) return
      event.preventDefault()
      pointers.current.set(event.pointerId, {
        side, element: event.currentTarget,
        action: courtActionAt(event.clientX - bounds.left, bounds.width),
      })
      event.currentTarget.setPointerCapture(event.pointerId)
      publish(side)
    },
    onPointerMove(event: ReactPointerEvent<HTMLElement>) {
      const held = pointers.current.get(event.pointerId)
      if (!held || !enabled) return
      event.preventDefault()
      const bounds = event.currentTarget.getBoundingClientRect()
      held.action = courtActionAt(event.clientX - bounds.left, bounds.width)
      publish(held.side)
    },
    onPointerUp: release,
    onPointerCancel: release,
    onLostPointerCapture: release,
  }
}

export function HumanControls({ m, humanSides }: Props) {
  const held = useRef(new Map<string, HeldDirection>())
  const pointers = useRef(new Map<number, Element>())
  const taps = useRef(new Map<string, ReturnType<typeof setTimeout>>())
  const latest = useRef(m)
  latest.current = m
  const [pressed, setPressed] = useState(new Set<string>())
  const enabled = m.playing && !endedMatch(m.frame)
  const left = humanSides[0], right = humanSides[1]

  function publish() {
    for (const side of [0, 1] as const) {
      latest.current.setHumanAction(side, heldHumanAction(held.current.values(), side))
    }
    setPressed(new Set([...held.current.values()].map((entry) => entry.side + ':' + entry.action)))
  }
  function start(key: string, side: Side, action: 0 | 2) {
    if (!enabled) return
    held.current.set(key, { side, action })
    publish()
  }
  function stop(key: string) {
    if (held.current.delete(key)) publish()
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
      for (const timer of taps.current.values()) clearTimeout(timer)
      taps.current.clear()
      const captures = [...pointers.current]
      pointers.current.clear()
      for (const [id, element] of captures) releaseCapture(element, id)
      latest.current.setHumanAction(0, null)
      latest.current.setHumanAction(1, null)
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
  return (
    <section className="human-controls" aria-label="Human paddle controls">
      <p className="human-control-heading">Hold a button or either half of the court to move.</p>
      <div className={'human-control-pads' + (left && right ? ' two-players' : '')}>
        {([0, 1] as const).filter((side) => humanSides[side]).map((side) => (
          <div className={'human-control-pad side-' + side} key={side}>
            {left && right && <div className="human-control-label">
              <span><i className={side === 0 ? 'p1-dot' : 'p2-dot'} /> P{side + 1} · {left && right && side === 1 ? 'Top' : 'Bottom'}</span>
              <small>{left && right ? side === 0 ? 'A / D' : '← / →' : 'A / D or ← / →'}</small>
            </div>}
            <div className="human-direction-buttons">
              {([0, 2] as const).map((action) => {
                const label = action === 0 ? 'Left' : 'Right'
                const key = side + ':' + action
                const Icon = action === 0 ? ArrowLeft : ArrowRight
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
          </div>
        ))}
      </div>
    </section>
  )
}

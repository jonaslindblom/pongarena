import { ArrowLeft, ArrowRight } from 'lucide-react'
import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { endedMatch, type Pair, type Side } from './rules'
import { courtActionAt, heldHumanAction, humanSideAt, type HeldDirection, type HumanAction } from './humanInput'
import type { Match } from './useMatch'

type CapturedPointer = HeldDirection & { element: HTMLElement }

function releaseCapture(element: Element, pointerId: number) {
  if (element.hasPointerCapture(pointerId)) element.releasePointerCapture(pointerId)
}

export function useCourtPointerControls(m: Match, humanSides: Pair<boolean>) {
  const latest = useRef(m)
  latest.current = m
  const pointers = useRef(new Map<number, CapturedPointer>())
  const [directions, setDirections] = useState<Pair<HumanAction | null>>([null, null])
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
      setDirections((previous) => previous[0] === null && previous[1] === null ? previous : [null, null])
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
    const action = heldHumanAction(pointers.current.values(), side)
    latest.current.setHumanAction(side, action)
    setDirections((previous) => {
      if (previous[side] === action) return previous
      const next: Pair<HumanAction | null> = [...previous]
      next[side] = action
      return next
    })
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
    directions,
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
      const action = courtActionAt(event.clientX - bounds.left, bounds.width)
      if (held.action === action) return
      held.action = action
      publish(held.side)
    },
    onPointerUp: release,
    onPointerCancel: release,
    onLostPointerCapture: release,
  }
}

export function CourtTouchHints({ humanSides, directions, playing }: {
  humanSides: Pair<boolean>; directions: Pair<HumanAction | null>; playing: boolean
}) {
  if (!humanSides[0] && !humanSides[1]) return null
  return (
    <div className={'court-touch-hints' + (playing ? ' is-playing' : '') +
      (humanSides[0] && humanSides[1] ? ' two-players' : '')} aria-hidden="true">
      {([0, 1] as const).filter((side) => humanSides[side]).map((side) => (
        <div className={'touch-zone-pair side-' + side} key={side}>
          {([0, 2] as const).map((action) => {
            const left = action === 0
            const Arrow = left ? ArrowLeft : ArrowRight
            const held = directions[side] === action || directions[side] === 1
            return (
              <div className={'touch-zone ' + (left ? 'left' : 'right')}
                data-held={held || undefined} key={action}>
                <span className="touch-zone-cue">
                  <Arrow size={24} strokeWidth={1.3} />
                  <span>Hold {left ? 'left' : 'right'}</span>
                </span>
              </div>
            )
          })}
        </div>
      ))}
    </div>
  )
}

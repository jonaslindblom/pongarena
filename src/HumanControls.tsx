import { ArrowLeft, ArrowRight, MousePointer2 } from 'lucide-react'
import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type RefObject } from 'react'
import { endedMatch, type Pair, type Side } from './rules'
import { courtActionAt, heldHumanAction, humanKeyAction, humanSideAt, type HeldDirection, type HumanAction } from './humanInput'
import type { Match } from './useMatch'

type CapturedPointer = HeldDirection & { element: HTMLElement }

function releaseCapture(element: Element, pointerId: number) {
  if (element.hasPointerCapture(pointerId)) element.releasePointerCapture(pointerId)
}

export function useCourtPointerControls(m: Match, humanSides: Pair<boolean>, field: RefObject<SVGSVGElement | null>) {
  const latest = useRef(m)
  latest.current = m
  const pointers = useRef(new Map<number, CapturedPointer>())
  const mouseSide = useRef<Side | null>(null)
  const [mouseActive, setMouseActive] = useState(false)
  const [keyboardUsed, setKeyboardUsed] = useState(false)
  const [directions, setDirections] = useState<Pair<HumanAction | null>>([null, null])
  const enabled = m.playing && !endedMatch(m.frame)
  const left = humanSides[0], right = humanSides[1]

  function releaseMouse() {
    if (mouseSide.current === null) return
    latest.current.setHumanTarget(mouseSide.current, null)
    mouseSide.current = null
    setMouseActive(false)
  }

  useEffect(() => {
    function clear() {
      releaseMouse()
      const held = [...pointers.current]
      pointers.current.clear()
      for (const [id, pointer] of held) {
        latest.current.setHumanAction(pointer.side, null)
        releaseCapture(pointer.element, id)
      }
      setDirections((previous) => previous[0] === null && previous[1] === null ? previous : [null, null])
    }
    const hidden = () => { if (document.hidden) clear() }
    function keydown(event: KeyboardEvent) {
      if (!enabled ||
        (event.target as Element).closest('input, select, textarea, dialog, [contenteditable="true"]')) return
      const held = new Set([event.code])
      if ((left && humanKeyAction(held, 0, left && right) !== null) ||
        (right && humanKeyAction(held, 1, left && right) !== null)) setKeyboardUsed(true)
      if (mouseSide.current !== null &&
        (event.key === 'Escape' || humanKeyAction(held, mouseSide.current, left && right) !== null))
        releaseMouse()
    }
    function moveMouse(event: PointerEvent) {
      if (enabled && event.pointerType === 'mouse') aimMouse(event.clientX)
    }
    function outsideClick(event: PointerEvent) {
      if (!field.current?.contains(event.target as Node)) releaseMouse()
    }
    function cancelMouse(event: PointerEvent) {
      if (event.pointerType === 'mouse') releaseMouse()
    }
    window.addEventListener('blur', clear)
    window.addEventListener('keydown', keydown)
    window.addEventListener('pointermove', moveMouse)
    window.addEventListener('pointerdown', outsideClick, true)
    window.addEventListener('pointercancel', cancelMouse)
    document.addEventListener('visibilitychange', hidden)
    return () => {
      clear()
      window.removeEventListener('blur', clear)
      window.removeEventListener('keydown', keydown)
      window.removeEventListener('pointermove', moveMouse)
      window.removeEventListener('pointerdown', outsideClick, true)
      window.removeEventListener('pointercancel', cancelMouse)
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
  function aimMouse(clientX: number) {
    if (mouseSide.current === null || !field.current) return
    const bounds = field.current.getBoundingClientRect()
    latest.current.setHumanTarget(mouseSide.current, (clientX - bounds.left) / bounds.width * 160)
  }
  return {
    active: enabled && (left || right),
    directions,
    mouseActive,
    showMouseHint: !keyboardUsed,
    mouseInput: {
      onPointerDown(event: ReactPointerEvent<HTMLElement>) {
        if (!enabled || event.pointerType !== 'mouse' || event.button !== 0 || !field.current) return
        const bounds = field.current.getBoundingClientRect()
        const side = humanSideAt(event.clientY - bounds.top, bounds.height, humanSides)
        if (side === null) return
        event.preventDefault()
        event.currentTarget.focus({ preventScroll: true })
        releaseMouse()
        mouseSide.current = side
        setMouseActive(true)
        setKeyboardUsed(false)
        aimMouse(event.clientX)
      },
    },
    onPointerDown(event: ReactPointerEvent<HTMLElement>) {
      if (!enabled || event.pointerType === 'mouse' || event.button !== 0 || (event.target as Element).closest('button')) return
      const bounds = event.currentTarget.getBoundingClientRect()
      const side = humanSideAt(event.clientY - bounds.top, bounds.height, humanSides)
      if (side === null) return
      event.preventDefault()
      releaseMouse()
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

export function CourtMouseHint({ active }: { active: boolean }) {
  return (
    <div className={'court-mouse-hint' + (active ? ' is-steering' : '')} role="status">
      <MousePointer2 size={15} aria-hidden="true" />
      {active ? 'Move left and right' : 'Click to steer · or use ← →'}
    </div>
  )
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

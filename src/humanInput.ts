import { PADDLE_MAX_Y, PADDLE_MIN_Y, PADDLE_SPEED, type Pair, type Side } from './game'

export type HumanAction = 0 | 1 | 2
export type HeldDirection = { side: Side; action: 0 | 2 }

export function humanSideAt(position: number, size: number, humanSides: Pair<boolean>): Side | null {
  if (!humanSides[0] && !humanSides[1]) return null
  if (!humanSides[0]) return 1
  if (!humanSides[1]) return 0
  return position < size / 2 ? 1 : 0
}

export function courtActionAt(position: number, size: number): 0 | 2 {
  return position < size / 2 ? 0 : 2
}

export function humanTargetAction(position: number, target: number | null): HumanAction | null {
  if (target === null) return null
  const destination = Math.max(PADDLE_MIN_Y, Math.min(PADDLE_MAX_Y, target))
  const distance = destination - position
  // Stop at the nearest reachable position instead of oscillating around the
  // cursor. At either edge, the game's clamp lets the paddle arrive exactly.
  if (distance === 0 || (Math.abs(distance) <= PADDLE_SPEED / 2 &&
    destination > PADDLE_MIN_Y && destination < PADDLE_MAX_Y)) return 1
  return distance < 0 ? 0 : 2
}

export function humanKeyAction(held: Set<string>, side: Side, both: boolean): HumanAction | null {
  const left = ((!both || side === 0) && (held.has('KeyA') || held.has('KeyW'))) ||
    ((!both || side === 1) && (held.has('ArrowLeft') || held.has('ArrowUp')))
  const right = ((!both || side === 0) && (held.has('KeyD') || held.has('KeyS'))) ||
    ((!both || side === 1) && (held.has('ArrowRight') || held.has('ArrowDown')))
  if (!left && !right) return null
  return left === right ? 1 : left ? 0 : 2
}

export function heldHumanAction(held: Iterable<HeldDirection>, side: Side): HumanAction | null {
  let left = false, right = false
  for (const entry of held) {
    if (entry.side === side) {
      left ||= entry.action === 0
      right ||= entry.action === 2
    }
  }
  if (!left && !right) return null
  return left === right ? 1 : left ? 0 : 2
}

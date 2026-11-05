import type { Pair, Side } from './game'

export type HumanAction = 0 | 1 | 2
export type HeldDirection = { side: Side; action: 0 | 2 }

export function humanTargetAction(position: number, target: number, burst: boolean): HumanAction | 3 | 4 {
  const error = target - position
  if (Math.abs(error) <= 1) return 1
  // Finish a drag with an ordinary step instead of bursting past the target
  // repeatedly and spending energy without making useful progress.
  if (burst && Math.abs(error) >= 4) return error < 0 ? 3 : 4
  return error < 0 ? 0 : 2
}

export function humanSideAt(x: number, width: number, humanSides: Pair<boolean>): Side | null {
  if (!humanSides[0] && !humanSides[1]) return null
  if (!humanSides[0]) return 1
  if (!humanSides[1]) return 0
  return x < width / 2 ? 0 : 1
}

export function courtTargetAt(y: number, height: number) {
  return Math.max(15, Math.min(145, (y / Math.max(1, height)) * 160))
}

export function heldHumanAction(held: Iterable<HeldDirection>, side: Side): HumanAction | null {
  let up = false, down = false
  for (const entry of held) {
    if (entry.side === side) {
      up ||= entry.action === 0
      down ||= entry.action === 2
    }
  }
  if (!up && !down) return null
  return up === down ? 1 : up ? 0 : 2
}

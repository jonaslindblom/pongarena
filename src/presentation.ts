import { endedRally, type Frame, type Pair, TICKS_PER_SECOND } from './rules'

export const TICK_MS = 1000 / TICKS_PER_SECOND

export interface Presentation {
  from: Frame
  to: Frame
  startedAt: number
  duration: number
  animate: boolean
  trail: Frame[]
}

export function frameDuration(frame: Frame, speed: number) {
  return (endedRally(frame) ? 650 : TICK_MS) / speed
}

// A slow browser may lose wall-clock time, but never skips a simulation state.
// Limit catch-up to three decision intervals so input cannot be buried in debt.
export function boundedDeadline(due: number, now: number, speed: number) {
  return Math.max(due, now - 3 * TICK_MS / speed)
}

export function progress(p: Presentation, now: number) {
  return p.animate ? Math.max(0, Math.min(1, (now - p.startedAt) / p.duration)) : 1
}

export function interpolate(from: Frame, to: Frame, amount: number) {
  const t = Math.max(0, Math.min(1, amount))
  if (from.rally !== to.rally || to.tick !== from.tick + 1 || to.event === 'serve' || t === 1)
    return { x: to.x, y: to.y, paddles: to.paddles }
  // Follow the incoming trajectory through the wall instead of cutting across
  // its reflected corner. Contact X ends at the engine's exact paddle plane.
  const unfolded = from.y + from.vy * t
  const folded = ((unfolded - 4) % 304 + 304) % 304
  return {
    x: from.x + (to.x - from.x) * t,
    y: 4 + (folded <= 152 ? folded : 304 - folded),
    paddles: from.paddles.map((p, side) => p + (to.paddles[side] - p) * t) as Pair<number>,
  }
}

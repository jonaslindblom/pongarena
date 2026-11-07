export type Side = 0 | 1
export type Scores = [number, number, number]
export type Pair<T> = [T, T]
export const ACTIONS = ['Up', 'Stay', 'Down'] as const
export const RALLIES = 5
export const MAX_BALL_VX = 13
export const MAX_BALL_VY = 9
export const TICKS_PER_SECOND = 30
export const RALLY_TICK_LIMIT = 1600
export type Event = 'serve' | 'return' | 'point' | 'draw' | null

export interface Frame {
  tick: number
  rally: number
  rallyTick: number
  x: number
  y: number
  vx: number
  vy: number
  paddles: Pair<number>
  points: Pair<number>
  returns: number
  event: Event
  drawReason?: 'loop' | 'timeout'
  actor?: Side
  scores: Pair<Scores | null>
}

const clamp = (n: number, min: number, max: number) => Math.max(min, Math.min(max, n))
export const endedRally = (f: Frame) => f.event === 'point' || f.event === 'draw'
export const endedMatch = (f: Frame) => f.rally === RALLIES && endedRally(f)

// Only use for two fixed, deterministic, stateless policies. Human decisions and
// sampled training actions can escape a repeated physical state.
export class LoopDetector {
  private seen = new Set<string>()
  private rally = 0

  reset() {
    this.seen.clear()
    this.rally = 0
  }

  observe(f: Frame) {
    if (f.rally !== this.rally) {
      this.seen.clear()
      this.rally = f.rally
    }
    if (f.event !== 'return') return false
    const key = [f.x, f.y, f.vx, f.vy, ...f.paddles].join(',')
    if (this.seen.has(key)) return true
    this.seen.add(key)
    return false
  }
}

export function serve(seed: number, prior?: Frame): Frame {
  const rally = prior ? prior.rally + 1 : 1
  let n = seed >>> 0
  function random() {
    n = (Math.imul(1664525, n) + 1013904223) >>> 0
    return n / 4294967296
  }
  const sign = () => (random() < 0.5 ? -1 : 1)
  const firstVx = 5 * sign()
  // Each receiver gets a rising and a falling serve in the first four rallies.
  const firstSlope = sign()
  const secondSlope = sign()
  const directions = [firstSlope, secondSlope, -firstSlope, -secondSlope, sign()]
  // Five varied angle bands, all within 10–15° of the receiving axis.
  // Hundredth-unit velocities keep the independent Python implementation exact.
  const openings = [89, 98, 107, 116, 125].map((minimum) =>
    (minimum + Math.floor(random() * 9)) / 100)
  for (let i = openings.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1))
    ;[openings[i], openings[j]] = [openings[j], openings[i]]
  }
  const speedY = openings[rally - 1]
  return {
    tick: prior ? prior.tick + 1 : 0,
    rally,
    rallyTick: 0,
    x: 180,
    y: 80,
    vx: rally % 2 ? firstVx : -firstVx,
    vy: speedY * directions[rally - 1],
    paddles: [80, 80],
    points: prior ? [...prior.points] : [0, 0],
    returns: 0,
    event: 'serve',
    scores: [null, null],
  }
}

export function observation(f: Frame, side: Side) {
  const mirror = side === 0 ? 1 : -1
  return Float32Array.of(
    (mirror * (f.x - 180)) / 180,
    (f.y - 80) / 80,
    (mirror * f.vx) / 6,
    f.vy / 6,
    (f.paddles[side] - 80) / 80,
    (f.paddles[1 - side] - 80) / 80,
  )
}

export function choice(scores: Scores) {
  if (!scores.every(Number.isFinite))
    throw new Error('The controller produced a non-finite action score.')
  return scores.indexOf(Math.max(...scores))
}

export function advance(prior: Frame, scores: Pair<Scores>, loops?: LoopDetector): Frame {
  const paddles = prior.paddles.map((p, i) =>
    clamp(p + 2 * (choice(scores[i]) - 1), 15, 145),
  ) as Pair<number>
  const f: Frame = {
    ...prior,
    tick: prior.tick + 1,
    rallyTick: prior.rallyTick + 1,
    x: prior.x + prior.vx,
    y: prior.y + prior.vy,
    paddles,
    points: [...prior.points],
    event: null,
    drawReason: undefined,
    actor: undefined,
    scores,
  }
  if (f.y < 4 || f.y > 156) {
    f.y = f.y < 4 ? 8 - f.y : 312 - f.y
    f.vy *= -1
  }
  if ((f.vx < 0 && f.x <= 15) || (f.vx > 0 && f.x >= 345)) {
    const side: Side = f.vx < 0 ? 0 : 1
    const offset = f.y - paddles[side]
    if (Math.abs(offset) > 15) {
      f.actor = side === 0 ? 1 : 0
      f.points[f.actor]++
      f.event = 'point'
    } else {
      f.x = side === 0 ? 15 : 345
      const speed = Math.min(MAX_BALL_VX, Math.abs(f.vx) + 1)
      f.vx = (side === 0 ? 1 : -1) * speed
      // Canonicalize negative zero so browser, Python and saved traces agree.
      f.vy = Math.round((MAX_BALL_VY * speed * offset) / (15 * MAX_BALL_VX)) || 0
      f.returns++
      f.actor = side
      f.event = 'return'
    }
  }
  if (loops?.observe(f)) {
    f.event = 'draw'
    f.drawReason = 'loop'
    f.actor = undefined
  }
  if (f.rallyTick >= RALLY_TICK_LIMIT && f.event !== 'point' && f.event !== 'draw') {
    f.event = 'draw'
    f.drawReason = 'timeout'
    f.actor = undefined
  }
  return f
}

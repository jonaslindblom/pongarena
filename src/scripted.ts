import type { Scores } from './game'

export type ScriptKind = 'attack' | 'defend' | 'interceptor'

// Exact stateless ports of pace_study.make_policy('pace13-v9-scaled'). These
// geometric reference opponents are scripts, not networks or submitted weights.
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value))
const next = (center: number, action: number) => clamp(center + 2 * (action - 1), 15, 145)
const reflect = (height: number) => {
  const z = ((height - 4) % 304 + 304) % 304
  return 4 + (z <= 152 ? z : 304 - z)
}

// NumPy rint uses ties-to-even; this is deliberately not the shot's Math.round.
function roundEven(value: number) {
  const floor = Math.floor(value)
  return value - floor === 0.5 ? floor + (floor % 2 === 0 ? 0 : 1) : Math.round(value)
}

function reachable(ticks: number, from: number, to: number) {
  if ((from - to) % 2 === 0) return Math.abs(from - to) <= 2 * ticks
  if (from % 2 !== 0) return false
  // Only a clamp at an odd boundary can change an even center's parity.
  return Math.min(
    Math.ceil((from - 15) / 2) + (to - 15) / 2,
    Math.ceil((145 - from) / 2) + (145 - to) / 2,
  ) <= ticks
}

// Equivalent to offense_eval.contact_paths()[ticks, from-15, to-15, 3], without
// allocating its large dynamic-programming table. Exported for exhaustive parity
// checks against that independently generated Python table.
export function firstContactAction(ticks: number, from: number, to: number) {
  if (ticks === 1) {
    // Python initializes in action order: duplicate clamped destinations retain
    // the LAST action (notably down at the lower boundary, center 145).
    for (let action = 2; action >= 0; action--) if (next(from, action) === to) return action
    return -1
  }
  const toward = to < from ? 0 : to > from ? 2 : 1
  if (reachable(ticks - 1, next(from, toward), to)) return toward
  const second = to === from ? 0 : 1
  if (reachable(ticks - 1, next(from, second), to)) return second
  const third = to <= from ? 2 : 0
  return reachable(ticks - 1, next(from, third), to) ? third : -1
}

interface Coverage {
  flight: number
  landing: Int16Array
  low: Float64Array
  high: Float64Array
  mid: Float64Array
}

const coverage: (Coverage | undefined)[] = []

function table(stage: number): Coverage {
  const cached = coverage[stage]
  if (cached) return cached
  const speed = Math.min(13, stage + 1)
  const flight = Math.ceil(330 / speed)
  const landing = new Int16Array(157 * 31)
  const low = new Float64Array(157).fill(15)
  const high = new Float64Array(157).fill(145)
  const mid = new Float64Array(157)
  for (let height = 0; height <= 156; height++) {
    for (let offset = -15; offset <= 15; offset++) {
      const destination = reflect(height + flight * Math.floor(9 * speed * offset / 195 + 0.5))
      landing[height * 31 + offset + 15] = destination
      const center = height - offset
      if (height >= 4 && center >= 15 && center <= 145) {
        low[height] = Math.max(low[height], destination - 15 - 2 * flight)
        high[height] = Math.min(high[height], destination + 15 + 2 * flight)
      }
    }
    mid[height] = (low[height] + high[height]) / 2
  }
  return coverage[stage] = { flight, landing, low, high, mid }
}

function moveToward(center: number, target: number) {
  let selected = 0
  let best = Infinity
  for (let action = 0; action < 3; action++) {
    const cost = Math.abs(next(center, action) - target) + (action === 1 ? 0 : 1e-6)
    if (cost < best) {
      best = cost
      selected = action
    }
  }
  return selected
}

export function scriptScores(kind: ScriptKind, input: Float32Array): Scores {
  if (input.length !== 6 || !input.every(Number.isFinite))
    throw new Error('Classic scripts require six finite observations.')
  const x = roundEven(input[0] * 180 + 180)
  const y = roundEven(input[1] * 80 + 80)
  const vx = roundEven(input[2] * 6)
  const vy = roundEven(input[3] * 6)
  const own = roundEven(input[4] * 80 + 80)
  const other = roundEven(input[5] * 80 + 80)
  const ticks = clamp(Math.ceil((vx < 0 ? x - 15 : 345 - x) / Math.max(1, Math.abs(vx))), 1, 66)
  const height = reflect(y + ticks * vy)
  const stage = Math.min(13, Math.abs(vx))
  const current = table(stage)
  let selected = moveToward(own, vx < 0 ? clamp(height, 15, 145) : current.mid[height])
  if (vx < 0 && kind !== 'interceptor') {
    const following = table(Math.min(13, stage + 1))
    const expected = clamp(current.mid[height], other - 2 * Math.max(0, ticks - 1), other + 2 * Math.max(0, ticks - 1))
    let best = -Infinity
    // Ascending offsets preserve Python argmax's first-maximum tie break.
    for (let offset = -15; offset <= 15; offset++) {
      const center = height - offset
      if (center < 15 || center > 145) continue
      const first = firstContactAction(ticks, own, center)
      if (first < 0) continue
      const destination = current.landing[height * 31 + offset + 15]
      const recovery = clamp(following.mid[destination], Math.max(15, center - 2 * current.flight), Math.min(145, center + 2 * current.flight))
      const risk = stage < 5 ? 0 : Math.max(following.low[destination] - recovery, recovery - following.high[destination])
      const travel = Math.abs(center - own)
      const margin = Math.abs(destination - expected) - 15 - 2 * current.flight - 2
      const quality = kind === 'defend' ? -risk - 0.001 * travel : margin - 0.05 * Math.max(0, risk) - 0.001 * travel
      if (quality > best) {
        best = quality
        selected = first
      }
    }
  }
  const scores: Scores = [0, 0, 0]
  scores[selected] = 1
  return scores
}

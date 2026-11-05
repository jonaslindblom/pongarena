// Independent TypeScript reference for the isolated return-law experiment.
// The released game.ts and the Pong v3 training contract stay unchanged.
import { choice, type Frame, type Pair, type Scores, type Side } from '../../src/game.ts'

export const candidates = {
  baseline: { verticalMax: 5, deflection: 0 },
  angle8: { verticalMax: 8, deflection: 0 },
  angle9: { verticalMax: 9, deflection: 0 },
  motion2: { verticalMax: 5, deflection: 2 },
} as const
export type Candidate = keyof typeof candidates

export function advanceCandidate(prior: Frame, scores: Pair<Scores>, name: Candidate, seen?: Set<string>): Frame {
  const rule = candidates[name]
  const paddles = prior.paddles.map((p, side) =>
    Math.max(15, Math.min(145, p + 2 * (choice(scores[side]) - 1))),
  ) as Pair<number>
  const frame: Frame = {
    ...prior,
    tick: prior.tick + 1,
    rallyTick: prior.rallyTick + 1,
    x: prior.x + prior.vx,
    y: prior.y + prior.vy,
    paddles,
    points: [...prior.points],
    scores,
    event: null,
    drawReason: undefined,
    actor: undefined,
  }
  if (frame.y < 4 || frame.y > 156) {
    frame.y = frame.y < 4 ? 8 - frame.y : 312 - frame.y
    frame.vy = -frame.vy
  }
  if ((frame.vx < 0 && frame.x <= 15) || (frame.vx > 0 && frame.x >= 345)) {
    const side: Side = frame.vx < 0 ? 0 : 1
    const offset = frame.y - paddles[side]
    if (Math.abs(offset) > 15) {
      frame.actor = side === 0 ? 1 : 0
      frame.points[frame.actor]++
      frame.event = 'point'
    } else {
      frame.x = side === 0 ? 15 : 345
      frame.vx = (side === 0 ? 1 : -1) * (5 + Math.floor(8 * Math.abs(offset) / 15 + 0.5))
      const movement = Math.sign(paddles[side] - prior.paddles[side])
      frame.vy = Math.floor(rule.verticalMax * offset / 15 + 0.5) + rule.deflection * movement
      frame.returns++
      frame.actor = side
      frame.event = 'return'
      const key = [frame.x, frame.y, frame.vx, frame.vy, ...paddles].join(',')
      if (seen?.has(key)) {
        frame.event = 'draw'
        frame.drawReason = 'loop'
        frame.actor = undefined
      }
      seen?.add(key)
    }
  }
  if (frame.rallyTick >= 1600 && frame.event !== 'point' && frame.event !== 'draw') {
    frame.event = 'draw'
    frame.drawReason = 'timeout'
    frame.actor = undefined
  }
  return frame
}

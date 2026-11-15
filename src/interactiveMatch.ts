import { endedMatch, endedRally, RALLIES, RALLY_TICK_LIMIT, TICKS_PER_SECOND, type Frame, type Ruleset } from './rules'

export const WATCH_TICK_LIMIT = 90 * TICKS_PER_SECOND

export function interactiveMatchEnded(frame: Frame, human: boolean) {
  return human ? endedRally(frame) && Math.max(...frame.points) >= 3 : endedMatch(frame)
}

export function interactiveServe(rules: Ruleset, seed: number, prior?: Frame): Frame {
  if (!prior || prior.rally < RALLIES) return rules.serve(seed, prior)
  // Human draws can require extra serves. Continue with deterministic five-serve sets.
  const set = Math.floor(prior.rally / RALLIES)
  const withinSet = prior.rally % RALLIES
  const nextSeed = (seed + set) >>> 0
  const next = rules.serve(nextSeed, withinSet ? { ...prior, rally: withinSet } : undefined)
  return { ...next, rally: prior.rally + 1, tick: prior.tick + 1, points: [...prior.points] }
}

export function interactiveTickLimit(human: boolean) {
  return human ? RALLY_TICK_LIMIT : WATCH_TICK_LIMIT
}

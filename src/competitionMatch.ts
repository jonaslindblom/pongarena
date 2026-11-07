import { endedMatch, endedRally, type Pair, type Scores, type Ruleset } from './rules'
import type { MatchResult } from './competition'

type Decision = (input: Float32Array) => Promise<Scores>

// Browser and CI use the same simulation, observation ordering and cancellation.
export async function competitionMatch(
  players: Pair<Decision>, rules: Ruleset, seed: number, signal?: AbortSignal,
): Promise<MatchResult> {
  let frame = rules.serve(seed)
  const loops = rules.detector()
  let yieldedAt = performance.now()
  while (!endedMatch(frame)) {
    signal?.throwIfAborted()
    if (endedRally(frame)) frame = rules.serve(seed, frame)
    else {
      const scores: Scores[] = []
      for (const side of [0, 1] as const) {
        scores.push(await players[side](rules.observation(frame, side)))
        signal?.throwIfAborted()
      }
      frame = rules.advance(frame, scores as Pair<Scores>, loops)
    }
    if (performance.now() - yieldedAt >= 8) {
      await new Promise<void>((resolve) => setTimeout(resolve, 0))
      yieldedAt = performance.now()
    }
  }
  return { points: frame.points, rallies: 5 }
}

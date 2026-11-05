import { infer, type Controller } from './models'
import { endedMatch, endedRally, type Pair, type Scores, type Ruleset } from './rules'
import type { PlayCompetitionMatch } from './competition'

export function browserCompetitionMatch(models: Controller[], rules: Ruleset): PlayCompetitionMatch {
  return async ({ left, right, seed, signal }) => {
    const players = [left, right].map((entry) => models.find((model) =>
      model.type === 'weights' && model.sha256 === entry.sha256 && model.inputSize === rules.inputSize &&
      model.outputSize === rules.outputSize))
    if (players.some((player) => !player?.session)) throw new Error('A model is unavailable for these rules.')
    let frame = rules.serve(seed)
    const loops = rules.detector()
    let yieldedAt = performance.now()
    while (!endedMatch(frame)) {
      signal?.throwIfAborted()
      if (endedRally(frame)) frame = rules.serve(seed, frame)
      else {
        const scores: Scores[] = []
        for (const side of [0, 1] as const) {
          scores.push(await infer(players[side]!, rules.observation(frame, side)))
          signal?.throwIfAborted()
        }
        frame = rules.advance(frame, scores as Pair<Scores>, loops)
      }
      // Yield to input and paint even when WASM inference resolves immediately.
      if (performance.now() - yieldedAt >= 8) {
        await new Promise<void>((resolve) => setTimeout(resolve, 0))
        yieldedAt = performance.now()
      }
    }
    return { points: frame.points, rallies: 5 }
  }
}

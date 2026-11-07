import { infer, type Controller } from './models'
import type { Pair, Ruleset } from './rules'
import type { PlayCompetitionMatch } from './competition'
import { competitionMatch } from './competitionMatch'

export function browserCompetitionMatch(models: Controller[], rules: Ruleset): PlayCompetitionMatch {
  return async ({ left, right, seed, signal }) => {
    const players = [left, right].map((entry) => models.find((model) =>
      model.type === (entry.type ?? 'weights') && model.sha256 === entry.sha256 && model.inputSize === rules.inputSize &&
      model.outputSize === rules.outputSize))
    if (players.some((player) => !player || (player.type === 'script' ? !player.script : !player.session)))
      throw new Error('A controller is unavailable for these rules.')
    return competitionMatch(players.map((player) => (input: Float32Array) => infer(player!, input)) as
      Pair<(input: Float32Array) => ReturnType<typeof infer>>, rules, seed, signal)
  }
}

import * as game from './game'

export type { Side, Pair, Scores, Frame } from './game'
export { RALLIES, RALLY_TICK_LIMIT, TICKS_PER_SECOND, endedRally, endedMatch, choice } from './game'

export interface Ruleset {
  id: string
  name: string
  contractId: string
  inputSize: 6
  outputSize: 3
  actions: readonly string[]
  serve: typeof game.serve
  observation: typeof game.observation
  advance: typeof game.advance
  detector(): game.LoopDetector
}

export const PONG: Ruleset = {
  id: 'pong', name: 'Pong', contractId: 'dense-mlp',
  inputSize: 6, outputSize: 3, actions: game.ACTIONS,
  serve: game.serve,
  observation: game.observation,
  advance: game.advance,
  detector: () => new game.LoopDetector(),
}

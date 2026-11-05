import * as classic from './game'
import * as burst from '../training/burst/game'

export type Side = classic.Side
export type Pair<T> = classic.Pair<T>
export type Scores = classic.Scores | burst.Scores
export type Frame = Omit<classic.Frame, 'scores'> & {
  scores: Pair<Scores | null>
  energy?: Pair<number>
}
type Detector = classic.LoopDetector | burst.LoopDetector
export interface Ruleset {
  id: string
  name: string
  contractId: string
  inputSize: 6 | 8
  outputSize: 3 | 5
  actions: readonly string[]
  energyMax?: number
  boostCost?: number
  serve(seed: number, prior?: Frame): Frame
  observation(frame: Frame, side: Side): Float32Array
  advance(frame: Frame, scores: Pair<Scores>, loops?: Detector): Frame
  detector(): Detector
}

export const CLASSIC: Ruleset = {
  id: 'pong-v4', name: 'Classic v4', contractId: 'observation-6/action-scores-3/v1',
  inputSize: 6, outputSize: 3, actions: classic.ACTIONS,
  serve: (seed, prior) => classic.serve(seed, prior as classic.Frame | undefined),
  observation: (frame, side) => classic.observation(frame as classic.Frame, side),
  advance: (frame, scores, loops) => classic.advance(frame as classic.Frame,
    scores as Pair<classic.Scores>, loops as classic.LoopDetector | undefined),
  detector: () => new classic.LoopDetector(),
}

export const BURST: Ruleset = {
  id: 'pong-burst-v1', name: 'Burst preview', contractId: 'observation-8/action-scores-5/v1',
  inputSize: 8, outputSize: 5, actions: burst.ACTIONS,
  energyMax: burst.MAX_ENERGY, boostCost: burst.BOOST_COST,
  serve: (seed, prior) => burst.serve(seed, prior as burst.Frame | undefined),
  observation: (frame, side) => burst.observation(frame as burst.Frame, side),
  advance: (frame, scores, loops) => burst.advance(frame as burst.Frame,
    scores as Pair<burst.Scores>, loops as burst.LoopDetector | undefined),
  detector: () => new burst.LoopDetector(),
}

export const RULESETS = [BURST, CLASSIC]
export const { RALLIES, RALLY_TICK_LIMIT, TICKS_PER_SECOND } = classic
export const endedRally = (frame: Frame) => frame.event === 'point' || frame.event === 'draw'
export const endedMatch = (frame: Frame) => frame.rally === RALLIES && endedRally(frame)
export function choice(scores: Scores) {
  if (!scores.every(Number.isFinite)) throw new Error('The controller produced a non-finite action score.')
  return scores.indexOf(Math.max(...scores))
}

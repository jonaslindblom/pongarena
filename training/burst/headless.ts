import { advance, endedMatch, endedRally, LoopDetector, observation, serve, type Scores } from './game.ts'

// A simple tracking policy, not a trained network or a strong competitor.
function policy(input: Float32Array): Scores {
  const error = input[1] - input[4]
  return [-error, .015, error, -1, -1]
}
const seed = Number(process.argv[2] ?? 24) >>> 0
const loops = process.argv.includes('--no-loops') ? undefined : new LoopDetector()
let frame = serve(seed)
while (!endedMatch(frame)) {
  frame = endedRally(frame) ? serve(seed, frame) :
    advance(frame, [policy(observation(frame, 0)), policy(observation(frame, 1))], loops)
  if (endedRally(frame)) console.log(JSON.stringify(frame))
}
console.log('Final score:', frame.points, 'Ticks:', frame.tick)

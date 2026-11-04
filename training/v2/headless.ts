import { advance, endedMatch, endedRally, observation, serve, type Scores } from './game.ts'

// Replace this function with inference. Both players see the same pre-move frame.
function policy(input: Float32Array): Scores {
  const error = input[1] - input[4]
  return [-error, 0.015, error]
}
const seed = Number(process.argv[2] ?? 24) >>> 0
let frame = serve(seed)
while (!endedMatch(frame)) {
  if (endedRally(frame)) frame = serve(seed, frame)
  else frame = advance(frame, [policy(observation(frame, 0)), policy(observation(frame, 1))])
  if (endedRally(frame)) console.log(JSON.stringify(frame))
}
console.log('Final score:', frame.points, 'Ticks:', frame.tick)

import { writeFileSync } from 'node:fs'
import { advance, endedMatch, endedRally, observation, serve, type Frame, type Pair, type Scores } from './game.ts'

const scores = (a: number): Scores => [a === 0 ? 1 : 0, a === 1 ? 1 : 0, a === 2 ? 1 : 0]
const record = (f: Frame) => ({ state: f, observations: [Array.from(observation(f, 0)), Array.from(observation(f, 1))] })
const matches = [0, 24, 902, 4294967295].map((seed, k) => {
  let f = serve(seed)
  const initial = record(f)
  const steps = []
  while (!endedMatch(f)) {
    const actions: Pair<number> | null = endedRally(f) ? null : k === 1 ? [1, 1] : [Math.floor(f.tick / 13) % 3, Math.floor(f.tick / 17 + 1) % 3]
    f = actions ? advance(f, [scores(actions[0]), scores(actions[1])]) : serve(seed, f)
    steps.push({ actions, ...record(f) })
  }
  return { seed, initial, steps }
})
const base = { ...serve(24), x: 20, y: 80, vx: -5, vy: 0 }
const cases: { name: string; before: Frame; scores: Pair<Scores> }[] = [
  { name: 'center contact', before: base, scores: [scores(1), scores(1)] },
  { name: 'maximum edge', before: { ...base, y: 95 }, scores: [scores(1), scores(1)] },
  { name: 'move before collision saves a miss', before: { ...base, y: 96 }, scores: [scores(2), scores(1)] },
  { name: 'wall before paddle', before: { ...base, y: 4, vy: -5, paddles: [15, 80] }, scores: [scores(1), scores(1)] },
  { name: 'negative half rounds toward positive infinity', before: { ...base, y: 75.5 }, scores: [scores(1), scores(1)] },
  { name: 'timeout', before: { ...base, x: 180, rallyTick: 799 }, scores: [scores(1), scores(1)] },
  { name: 'last tick point wins over timeout', before: { ...base, y: 30, rallyTick: 799 }, scores: [scores(1), scores(1)] },
  { name: 'last tick return still draws', before: { ...base, rallyTick: 799 }, scores: [scores(1), scores(1)] },
  { name: 'paddle clamp and score tie', before: { ...base, x: 180, paddles: [15, 145] }, scores: [[2, 2, 1], scores(2)] },
  { name: 'exact wall boundary does not reflect', before: { ...base, x: 180, y: 9, vy: -5 }, scores: [scores(1), scores(1)] },
  { name: 'right edge mirrors horizontal only', before: { ...base, x: 340, vx: 5, y: 65 }, scores: [scores(1), scores(1)] },
]
writeFileSync(new URL('fixtures.json', import.meta.url), JSON.stringify({ version: 'pong-v2', matches, cases: cases.map(c => ({ ...c, expected: record(advance(c.before, c.scores)) })) }) + '\n')
console.log('Wrote', matches.reduce((n, m) => n + m.steps.length, 0), 'match steps and', cases.length, 'boundary cases.')

import { writeFileSync } from 'node:fs'
import { advance, endedMatch, endedRally, LoopDetector, observation, serve, type Frame, type Pair, type Scores } from './game.ts'

const scores = (a: number): Scores => [a === 0 ? 1 : 0, a === 1 ? 1 : 0, a === 2 ? 1 : 0]
const record = (f: Frame) => ({ state: f, observations: [Array.from(observation(f, 0)), Array.from(observation(f, 1))] })
const matches = [0, 24, 902, 4294967295, 24, 902].map((seed, k) => {
  const loopDetection = k >= 4
  const loops = loopDetection ? new LoopDetector() : undefined
  let f = serve(seed)
  const initial = record(f)
  const steps = []
  while (!endedMatch(f)) {
    const actions: Pair<number> | null = endedRally(f) ? null : loopDetection
      ? f.paddles.map(p => f.y - p < -1 ? 0 : f.y - p > 1 ? 2 : 1) as Pair<number>
      : k === 1 ? [1, 1] : [Math.floor(f.tick / 13) % 3, Math.floor(f.tick / 17 + 1) % 3]
    f = actions ? advance(f, [scores(actions[0]), scores(actions[1])], loops) : serve(seed, f)
    steps.push({ actions, ...record(f) })
  }
  return { seed, loopDetection, initial, steps }
})
const base = { ...serve(24), x: 20, y: 80, vx: -5, vy: 0 }
const contact = advance(base, [scores(1), scores(1)])
const cases: { name: string; before: Frame; scores: Pair<Scores>; loopDetection?: boolean; seenContacts?: Frame[] }[] = [
  { name: 'center contact', before: base, scores: [scores(1), scores(1)] },
  { name: 'maximum edge', before: { ...base, y: 95 }, scores: [scores(1), scores(1)] },
  { name: 'move before collision saves a miss', before: { ...base, y: 96 }, scores: [scores(2), scores(1)] },
  { name: 'wall before paddle', before: { ...base, y: 4, vy: -5, paddles: [15, 80] }, scores: [scores(1), scores(1)] },
  { name: 'negative half rounds toward positive infinity', before: { ...base, y: 75.5 }, scores: [scores(1), scores(1)] },
  { name: 'timeout', before: { ...base, x: 180, rallyTick: 1599 }, scores: [scores(1), scores(1)] },
  { name: 'last tick point wins over timeout', before: { ...base, y: 30, rallyTick: 1599 }, scores: [scores(1), scores(1)] },
  { name: 'last tick return still draws', before: { ...base, rallyTick: 1599 }, scores: [scores(1), scores(1)] },
  { name: 'paddle clamp and score tie', before: { ...base, x: 180, paddles: [15, 145] }, scores: [[2, 2, 1], scores(2)] },
  { name: 'exact wall boundary does not reflect', before: { ...base, x: 180, y: 9, vy: -5 }, scores: [scores(1), scores(1)] },
  { name: 'right edge mirrors horizontal only', before: { ...base, x: 340, vx: 5, y: 65 }, scores: [scores(1), scores(1)] },
  { name: 'old 800-tick limit no longer ends a rally', before: { ...base, x: 180, rallyTick: 799 }, scores: [scores(1), scores(1)] },
  { name: 'repeated contact ends model rally', before: { ...base, rallyTick: 132 }, scores: [scores(1), scores(1)], loopDetection: true, seenContacts: [contact] },
  { name: 'human or stochastic mode keeps repeated contact alive', before: { ...base, rallyTick: 132 }, scores: [scores(1), scores(1)], loopDetection: false, seenContacts: [contact] },
  { name: 'loop reason takes precedence over simultaneous timeout', before: { ...base, rallyTick: 1599 }, scores: [scores(1), scores(1)], loopDetection: true, seenContacts: [contact] },
  { name: 'point takes precedence with detector on deadline', before: { ...base, y: 30, rallyTick: 1599 }, scores: [scores(1), scores(1)], loopDetection: true, seenContacts: [contact] },
  { name: 'new rally does not reuse previous contact history', before: { ...base, rally: 2 }, scores: [scores(1), scores(1)], loopDetection: true, seenContacts: [contact] },
  { name: 'old draw reason does not leak into next transition', before: { ...base, x: 180, drawReason: 'timeout' }, scores: [scores(1), scores(1)] },
]
const traces = [true, false].map(loopDetection => {
  const loops = loopDetection ? new LoopDetector() : undefined
  let f: Frame = { ...serve(24), x: 180, y: 80, vx: -5, vy: 0 }
  const initial = record(f)
  const steps = []
  while (!endedRally(f)) {
    const actions: Pair<number> = [1, 1]
    f = advance(f, [scores(1), scores(1)], loops)
    steps.push({ actions, ...record(f) })
  }
  return { name: loopDetection ? 'deterministic center loop' : 'same actions without loop proof', loopDetection, initial, steps }
})
const detectorCases = [
  { name: 'counters, points and scores do not change future policy inputs', frames: [contact, { ...contact, tick: 999, rallyTick: 998, returns: 80, points: [2, 1] as Pair<number>, scores: [[5, 0, 1], [1, 7, 0]] as Pair<Scores> }] },
  { name: 'both paddles belong in the exact key', frames: [contact, { ...contact, paddles: [80, 82] as Pair<number> }, contact] },
  { name: 'velocity belongs in the exact key', frames: [contact, { ...contact, vy: 1 }, contact] },
  { name: 'history resets automatically on next rally', frames: [contact, { ...contact, rally: 2 }, { ...contact, rally: 2 }] },
  { name: 'only post-contact states count', frames: [{ ...contact, event: null }, contact, contact] as Frame[] },
]
const data = {
  version: 'pong-v3', matches, traces,
  cases: cases.map(c => {
    const loops = c.loopDetection ? new LoopDetector() : undefined
    for (const f of c.seenContacts ?? []) loops?.observe(f)
    return { ...c, expected: record(advance(c.before, c.scores, loops)) }
  }),
  detectorCases: detectorCases.map(c => {
    const loops = new LoopDetector()
    return { ...c, expected: c.frames.map(f => loops.observe(f)) }
  }),
}
writeFileSync(new URL('fixtures.json', import.meta.url), JSON.stringify(data) + '\n')
console.log('Wrote', matches.reduce((n, m) => n + m.steps.length, 0), 'match steps,', traces.reduce((n, t) => n + t.steps.length, 0), 'trace steps,', cases.length, 'boundary cases and', detectorCases.length, 'detector histories.')

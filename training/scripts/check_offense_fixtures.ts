import assert from 'node:assert/strict'
import { readFileSync, writeFileSync } from 'node:fs'
import { advance, LoopDetector, endedRally, type Frame, type Pair, type Scores } from '../../src/game.ts'
import { advanceCandidate, type Candidate } from './offense_reference.ts'

const file = process.argv[2] ?? 'training/results/offense-study/verification/fixtures.json'
const fixtures = JSON.parse(readFileSync(file, 'utf8'))
const clean = (frame: Frame) => JSON.parse(JSON.stringify(frame))
const scoresFor = (actions: Pair<number>) => actions.map((action) =>
  [0, 1, 2].map((candidate) => Number(candidate === action)),
) as Pair<Scores>
let transitions = 0
for (const fixture of fixtures.cases) {
  const seen = new Set<string>((fixture.seen_before ?? []).map((state: number[]) => state.join(',')))
  const frame = advanceCandidate(fixture.before, scoresFor(fixture.actions), fixture.rules, seen)
  assert.deepEqual(clean(frame), fixture.after, fixture.name)
  transitions++
}
for (const trace of fixtures.trajectories) {
  let frame: Frame = trace.initial
  let baseline = frame
  const seen = trace.loop_detection ? new Set<string>() : undefined
  const loops = trace.loop_detection ? new LoopDetector() : undefined
  for (const step of trace.steps) {
    assert.equal(endedRally(frame), false, 'Fixtures must stop when the rally ends')
    const scores = scoresFor(step.actions)
    frame = advanceCandidate(frame, scores, trace.rules as Candidate, seen)
    assert.deepEqual(clean(frame), step.after, `${trace.rules}, seed ${trace.seed}, tick ${frame.tick}`)
    if (trace.rules === 'baseline') {
      baseline = advance(baseline, scores, loops)
      assert.deepEqual(clean(frame), clean(baseline), 'Experimental baseline must exactly preserve released v3')
    }
    transitions++
  }
  assert.equal(endedRally(frame), true, 'Trajectory must include its terminal point/draw')
}
const result = { cases: fixtures.cases.length, completeTrajectories: fixtures.trajectories.length, transitions,
  pythonTypeScriptExact: true, baselineMatchesReleasedV3: true }
writeFileSync('training/results/offense-study/verification/typescript.json', JSON.stringify(result, null, 2) + '\n')
console.log(result)

import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { advance, choice, endedMatch, endedRally, LoopDetector, observation, serve, type Frame, type Pair, type Scores } from '../src/game'
import { firstContactAction, scriptScores, type ScriptKind } from '../src/scripted'

const fixture = JSON.parse(readFileSync(new URL('./fixtures/scripted-v4.json', import.meta.url), 'utf8')) as {
  config: string
  kinds: ScriptKind[]
  sources: Record<string, string>
  contact_path_sha256: string[]
  probes: number[][]
  rounding: number[][]
  matches: { seed: number; kinds: Pair<ScriptKind>; actions: string; trajectory_sha256: string; endings: number[][] }[]
}

test('script fixtures identify the exact evaluated policies and independent frozen physics', () => {
  assert.equal(fixture.config, 'pace13-v9-scaled')
  for (const [path, digest] of Object.entries(fixture.sources)) {
    assert.equal(createHash('sha256').update(readFileSync(new URL(`../${path}`, import.meta.url))).digest('hex'), digest,
      `${path} changed: regenerate and review script fixtures against the current evaluated opponents`)
  }
})

test('contact planning preserves every exact Python path, including clamp parity and action ties', () => {
  // Every legal starting/target center at all 66 horizons, not sampled paths.
  for (let ticks = 1; ticks <= 66; ticks++) {
    const actions = new Int8Array(131 * 131)
    for (let from = 15; from <= 145; from++) for (let to = 15; to <= 145; to++) {
      actions[(from - 15) * 131 + to - 15] = firstContactAction(ticks, from, to)
    }
    assert.equal(createHash('sha256').update(actions).digest('hex'), fixture.contact_path_sha256[ticks - 1], `horizon ${ticks}`)
  }
  assert.equal(firstContactAction(1, 145, 145), 2, 'One-tick boundary initialization deliberately retains down')
  assert.equal(firstContactAction(2, 145, 145), 1, 'Longer paths prefer stay at an already reached target')
  assert.equal(firstContactAction(66, 15, 80), -1, 'An odd boundary center can never recover even movement parity')
})

test('all three browser policies choose the evaluated Python action across speeds, walls and rounding ties', () => {
  for (const [index, row] of fixture.probes.entries()) {
    const input = Float32Array.of((row[0] - 180) / 180, (row[1] - 80) / 80, row[2] / 6,
      row[3] / 6, (row[4] - 80) / 80, (row[5] - 80) / 80)
    fixture.kinds.forEach((kind, i) => assert.equal(choice(scriptScores(kind, input)), row[6 + i], `${kind}, physical probe ${index}: ${row}`))
  }
  for (const [index, row] of fixture.rounding.entries()) {
    fixture.kinds.forEach((kind, i) => assert.equal(choice(scriptScores(kind, Float32Array.from(row.slice(0, 6)))), row[6 + i], `${kind}, rounding probe ${index}`))
  }
})

function state(frame: Frame) {
  return [frame.tick, frame.rally, frame.rallyTick, frame.x, frame.y, frame.vx, frame.vy,
    ...frame.paddles, ...frame.points, frame.returns,
    [null, 'serve', 'return', 'point', 'draw'].indexOf(frame.event),
    frame.actor ?? -1, [undefined, 'loop', 'timeout'].indexOf(frame.drawReason)]
}

test('browser script matches preserve every Python action and frame on both sides through all five rallies', () => {
  const reasons = new Set<number>()
  for (const match of fixture.matches) {
    let frame = serve(match.seed)
    const loops = new LoopDetector()
    const digest = createHash('sha256')
    const endings: number[][] = []
    let tick = 0
    const label = `${match.kinds.join(' vs ')} seed ${match.seed}`
    while (true) {
      digest.update(`${state(frame).join(',')}\n`)
      if (endedRally(frame)) {
        endings.push(state(frame))
        reasons.add(state(frame).at(-1)!)
        if (endedMatch(frame)) break
        frame = serve(match.seed, frame)
        continue
      }
      assert.ok(tick < match.actions.length, `${label}: match outlasted Python`)
      const scores = match.kinds.map((kind, side) => scriptScores(kind, observation(frame, side as 0 | 1))) as Pair<Scores>
      assert.equal(3 * choice(scores[0]) + choice(scores[1]), Number(match.actions[tick]), `${label}, action ${tick}`)
      frame = advance(frame, scores, loops)
      tick++
    }
    assert.equal(tick, match.actions.length, `${label}: all recorded decisions were played`)
    assert.deepEqual(endings, match.endings, `${label}: scores, durations and terminal causes`)
    assert.equal(digest.digest('hex'), match.trajectory_sha256, `${label}: every physical frame, including serve transitions`)
  }
  assert.ok(reasons.has(0), 'Fixtures include points, so scoring is checked')
  assert.ok(reasons.has(1), 'Fixtures include exact-loop draws, so script statelessness matters')
})

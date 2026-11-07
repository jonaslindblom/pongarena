import type { CompetitionRun } from './competition'

export const LEAGUE_INTERVAL_RESAMPLES = 5000

export interface LeagueScoreInterval {
  rate: number
  serveSets: number
  interval: [number, number] | null
  reason?: 'insufficient-sets' | 'no-variation'
}

function quantile(sorted: Float64Array, fraction: number) {
  const index = (sorted.length - 1) * fraction
  const lower = Math.floor(index), weight = index - lower
  return sorted[lower] * (1 - weight) + sorted[Math.ceil(index)] * weight
}

/** Approximate 95% percentile-bootstrap intervals for league-point rates.
 * One draw contains a whole shared seed: both sides and every opponent together.
 * Reproducible resampling adds no new observations or independent game results.
 */
export function leagueScoreIntervals(run: CompetitionRun): Map<string, LeagueScoreInterval> {
  const result = new Map<string, LeagueScoreInterval>()
  if (run.status !== 'complete') return result
  const seeds = [...run.spec.seeds].sort((a, b) => a - b)
  const hashes = run.spec.entrants.map(entrant => entrant.sha256).sort()
  const seedIndex = new Map(seeds.map((seed, index) => [seed, index]))
  const modelIndex = new Map(hashes.map((hash, index) => [hash, index]))
  const scores = seeds.map(() => new Float64Array(hashes.length))
  for (const contest of run.contests) {
    const [first, second] = contest.matches
    const a = first.points[0] + second.points[1]
    const b = first.points[1] + second.points[0]
    const point = a > b ? 1 : a === b ? .5 : 0
    const row = scores[seedIndex.get(contest.seed)!]
    row[modelIndex.get(contest.a)!] += point
    row[modelIndex.get(contest.b)!] += 1 - point
  }
  const denominator = seeds.length * (hashes.length - 1)
  const varying = hashes.map((_, index) => scores.some(row => row[index] !== scores[0][index]))
  for (const [index, hash] of hashes.entries()) {
    result.set(hash, {
      rate: scores.reduce((sum, row) => sum + row[index], 0) / denominator,
      serveSets: seeds.length, interval: null,
      ...seeds.length < 2 ? { reason: 'insufficient-sets' as const }
        : !varying[index] ? { reason: 'no-variation' as const } : {},
    })
  }
  if (seeds.length < 2 || !varying.some(Boolean)) return result

  // Use the benchmark identity so browsers and generated reports agree exactly.
  let random = Number.parseInt(run.benchmark.slice(0, 8), 16) || 0x9e3779b9
  const samples = hashes.map(() => new Float64Array(LEAGUE_INTERVAL_RESAMPLES))
  for (let sample = 0; sample < LEAGUE_INTERVAL_RESAMPLES; sample++) {
    const totals = new Float64Array(hashes.length)
    for (let draw = 0; draw < seeds.length; draw++) {
      random ^= random << 13
      random ^= random >>> 17
      random ^= random << 5
      const row = scores[Math.floor((random >>> 0) / 0x100000000 * seeds.length)]
      for (let model = 0; model < hashes.length; model++) totals[model] += row[model]
    }
    for (let model = 0; model < hashes.length; model++) samples[model][sample] = totals[model] / denominator
  }
  for (const [index, hash] of hashes.entries()) {
    if (!varying[index]) continue
    const sorted = samples[index].sort()
    result.get(hash)!.interval = [quantile(sorted, .025), quantile(sorted, .975)]
  }
  return result
}

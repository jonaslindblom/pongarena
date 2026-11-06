export const COMPETITION_PROTOCOL = 'paired-round-robin-v1'

export const DIVISIONS = [
  { id: 'nano', name: 'Nano', maxParams: 512, maxMacs: 2048 },
  { id: 'small', name: 'Small', maxParams: 2048, maxMacs: 8192 },
  { id: 'standard', name: 'Standard', maxParams: 20480, maxMacs: 65536 },
  { id: 'open', name: 'Open', maxParams: 65536, maxMacs: 65536 },
] as const
export type Division = typeof DIVISIONS[number]['id']
export type CompetitionClass = Division | 'all'

export interface CompetitionEntrant {
  id: string
  sha256: string
  name: string
  // Omitted in existing weights-only records.
  type?: 'weights' | 'script'
  params: number | null
  macs: number | null
}
export interface CompetitionSpec {
  rulesId: string
  contractId: string
  division: CompetitionClass
  seeds: number[]
  entrants: CompetitionEntrant[]
}
export interface MatchResult {
  points: [number, number]
  rallies: number
}
export interface PairedContest {
  a: string
  b: string
  seed: number
  matches: [MatchResult, MatchResult]
}
export interface CompetitionRun {
  protocol: typeof COMPETITION_PROTOCOL
  benchmark: string
  spec: CompetitionSpec
  status: 'running' | 'complete' | 'cancelled' | 'failed'
  contests: PairedContest[]
  completedMatches: number
  totalMatches: number
  error?: string
}
export interface Standing {
  entrant: CompetitionEntrant
  rank: number | null
  played: number
  wins: number
  draws: number
  losses: number
  leaguePoints: number
  rallyPointsFor: number
  rallyPointsAgainst: number
  rallyDraws: number
}
export type PlayCompetitionMatch = (request: {
  left: CompetitionEntrant
  right: CompetitionEntrant
  seed: number
  signal?: AbortSignal
}) => Promise<MatchResult>

type BudgetStats = Pick<CompetitionEntrant, 'params' | 'macs'> & { type?: string }

export function fitsDivision(stats: BudgetStats, id: Division) {
  const limit = DIVISIONS.find((d) => d.id === id)!
  return (stats.type === undefined || stats.type === 'weights') &&
    stats.params !== null && Number.isSafeInteger(stats.params) && stats.params >= 0 && stats.params <= limit.maxParams &&
    stats.macs !== null && Number.isSafeInteger(stats.macs) && stats.macs >= 0 && stats.macs <= limit.maxMacs
}
export function smallestDivision(stats: BudgetStats): Division | null {
  return DIVISIONS.find((d) => fitsDivision(stats, d.id))?.id ?? null
}

export function fitsCompetition(entrant: CompetitionEntrant, division: CompetitionClass) {
  if (entrant.type === 'script')
    return division === 'all' && entrant.params === null && entrant.macs === null
  return fitsDivision(entrant, division === 'all' ? 'open' : division)
}

export async function sha256(bytes: Uint8Array) {
  const input = Uint8Array.from(bytes)
  // Phones opening the local Network URL use HTTP, where SubtleCrypto is absent.
  const digest = globalThis.crypto?.subtle
    ? await globalThis.crypto.subtle.digest('SHA-256', input)
    : portableSha256(input)
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('')
}

function normalizeSpec(spec: CompetitionSpec): CompetitionSpec {
  if (!spec || typeof spec.rulesId !== 'string' || !spec.rulesId ||
    typeof spec.contractId !== 'string' || !spec.contractId)
    throw new Error('A rules version and input/output contract are required.')
  if (spec.division !== 'all' && !DIVISIONS.some((d) => d.id === spec.division)) throw new Error('Choose a valid competition class.')
  if (!Array.isArray(spec.entrants) || spec.entrants.length < 2) throw new Error('Select at least two different controllers.')
  const hashes = new Set<string>(), ids = new Set<string>()
  for (const entrant of spec.entrants) {
    if (!/^[a-f0-9]{64}$/.test(entrant.sha256)) throw new Error('Every controller needs its SHA-256 identity.')
    if (hashes.has(entrant.sha256)) throw new Error('Identical controllers can enter only once.')
    if (typeof entrant.id !== 'string' || !entrant.id || ids.has(entrant.id) || typeof entrant.name !== 'string')
      throw new Error('Controller identifiers must be unique and names must be text.')
    if (!fitsCompetition(entrant, spec.division)) throw new Error(entrant.name + ' is ineligible for this class or exceeds its budget.')
    hashes.add(entrant.sha256)
    ids.add(entrant.id)
  }
  if (!Array.isArray(spec.seeds) || !spec.seeds.length || spec.seeds.length > 64 ||
    spec.seeds.some((seed) => !Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) ||
    new Set(spec.seeds).size !== spec.seeds.length)
    throw new Error('Use 1–64 distinct unsigned 32-bit seeds.')
  return {
    rulesId: spec.rulesId, contractId: spec.contractId, division: spec.division,
    seeds: [...spec.seeds].sort((a, b) => a - b),
    entrants: spec.entrants.map((e) => ({ ...e })).sort((a, b) => a.sha256.localeCompare(b.sha256)),
  }
}

export async function benchmarkIdentity(spec: CompetitionSpec) {
  const normalized = normalizeSpec(spec)
  const canonical = {
    protocol: COMPETITION_PROTOCOL,
    rulesId: normalized.rulesId,
    contractId: normalized.contractId,
    division: normalized.division,
    budget: normalized.division === 'all' ? null : DIVISIONS.find((d) => d.id === normalized.division),
    seeds: normalized.seeds,
    roster: normalized.entrants.map(({ sha256: hash, type, params, macs }) =>
      type === 'script' ? { hash, type, params, macs } : { hash, params, macs }),
    ralliesPerMatch: 5,
    matchesPerContest: 2,
    scoring: 'aggregate-rally-points; contest-win=1,draw=0.5,loss=0; equal-league-points-share-rank',
  }
  return sha256(new TextEncoder().encode(JSON.stringify(canonical)))
}

function checkedMatch(result: MatchResult): MatchResult {
  if (result.rallies !== 5 || result.points.length !== 2 ||
    result.points.some((n) => !Number.isInteger(n) || n < 0) ||
    result.points[0] + result.points[1] > 5)
    throw new Error('A ranked match must finish all five rallies and return valid points.')
  return { points: [...result.points], rallies: 5 }
}

export function standings(run: CompetitionRun): Standing[] {
  const rows = new Map(run.spec.entrants.map((entrant) => [entrant.sha256, {
    entrant, rank: null, played: 0, wins: 0, draws: 0, losses: 0, leaguePoints: 0,
    rallyPointsFor: 0, rallyPointsAgainst: 0, rallyDraws: 0,
  } as Standing]))
  for (const contest of run.contests) {
    const a = rows.get(contest.a)!, b = rows.get(contest.b)!
    const [first, second] = contest.matches
    const ap = first.points[0] + second.points[1], bp = first.points[1] + second.points[0]
    a.played++; b.played++
    a.rallyPointsFor += ap; a.rallyPointsAgainst += bp
    b.rallyPointsFor += bp; b.rallyPointsAgainst += ap
    a.rallyDraws += 10 - ap - bp; b.rallyDraws += 10 - ap - bp
    if (ap === bp) {
      a.draws++; b.draws++
      a.leaguePoints += 0.5; b.leaguePoints += 0.5
    } else {
      const winner = ap > bp ? a : b, loser = ap > bp ? b : a
      winner.wins++; winner.leaguePoints++; loser.losses++
    }
  }
  const sorted = [...rows.values()].sort((a, b) =>
    b.leaguePoints - a.leaguePoints || a.entrant.sha256.localeCompare(b.entrant.sha256))
  if (run.status === 'complete') {
    for (let i = 0; i < sorted.length; i++)
      sorted[i].rank = i && sorted[i].leaguePoints === sorted[i - 1].leaguePoints ? sorted[i - 1].rank : i + 1
  }
  return sorted
}

// playMatch must yield regularly during simulation and check signal before
// committing inference results. This runner also yields between whole matches.
export async function runCompetition(options: CompetitionSpec & {
  playMatch: PlayCompetitionMatch
  signal?: AbortSignal
  onProgress?: (run: CompetitionRun) => void
}): Promise<CompetitionRun> {
  const spec = normalizeSpec(options)
  const run: CompetitionRun = {
    protocol: COMPETITION_PROTOCOL, benchmark: await benchmarkIdentity(spec), spec,
    status: 'running', contests: [], completedMatches: 0,
    totalMatches: spec.entrants.length * (spec.entrants.length - 1) * spec.seeds.length,
  }
  const publish = () => options.onProgress?.({ ...run, contests: [...run.contests] })
  publish()
  try {
    for (let i = 0; i < spec.entrants.length; i++) for (let j = i + 1; j < spec.entrants.length; j++) {
      const a = spec.entrants[i], b = spec.entrants[j]
      for (const seed of spec.seeds) {
        const matches: MatchResult[] = []
        for (const [left, right] of [[a, b], [b, a]]) {
          if (options.signal?.aborted) throw new DOMException('Cancelled', 'AbortError')
          const result = await options.playMatch({ left, right, seed, signal: options.signal })
          if (options.signal?.aborted) throw new DOMException('Cancelled', 'AbortError')
          matches.push(checkedMatch(result))
          run.completedMatches++
          if (matches.length === 2) run.contests.push({ a: a.sha256, b: b.sha256, seed,
            matches: matches as [MatchResult, MatchResult] })
          publish()
          await new Promise<void>((resolve) => setTimeout(resolve, 0))
        }
      }
    }
    if (options.signal?.aborted) throw new DOMException('Cancelled', 'AbortError')
    run.status = 'complete'
  } catch (error) {
    run.status = options.signal?.aborted || (error instanceof Error && error.name === 'AbortError')
      ? 'cancelled' : 'failed'
    if (run.status === 'failed') run.error = error instanceof Error ? error.message : String(error)
  }
  publish()
  return run
}

// Local files are reproducible records, not trusted public leaderboard results.
// Validate completeness and rebuild standings rather than accepting saved ranks.
export async function restoreCompetition(value: unknown): Promise<CompetitionRun> {
  const saved = value as CompetitionRun
  if (!saved || saved.protocol !== COMPETITION_PROTOCOL || saved.status !== 'complete')
    throw new Error('Only complete results from this ranking protocol can be restored.')
  const spec = normalizeSpec(saved.spec)
  const benchmark = await benchmarkIdentity(spec)
  if (saved.benchmark !== benchmark) throw new Error('The saved benchmark identity does not match its inputs.')
  const hashes = new Set(spec.entrants.map((e) => e.sha256)), seen = new Set<string>()
  const expected = spec.entrants.length * (spec.entrants.length - 1) * spec.seeds.length / 2
  if (!Array.isArray(saved.contests) || saved.contests.length !== expected)
    throw new Error('The saved round-robin is incomplete.')
  const contests = saved.contests.map((contest) => {
    const key = [contest.a, contest.b, contest.seed].join(':')
    if (!hashes.has(contest.a) || !hashes.has(contest.b) || contest.a >= contest.b ||
      !spec.seeds.includes(contest.seed) || seen.has(key) ||
      !Array.isArray(contest.matches) || contest.matches.length !== 2)
      throw new Error('The saved round-robin contains an invalid or repeated contest.')
    seen.add(key)
    return { a: contest.a, b: contest.b, seed: contest.seed,
      matches: contest.matches.map(checkedMatch) as [MatchResult, MatchResult] }
  })
  return { protocol: COMPETITION_PROTOCOL, benchmark, spec, status: 'complete', contests,
    completedMatches: expected * 2, totalMatches: expected * 2 }
}
import { sha256 as portableSha256 } from '@noble/hashes/sha2.js'


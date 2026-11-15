export function newMatchSeed(previous?: number) {
  const seed = crypto.getRandomValues(new Uint32Array(1))[0]
  return seed === previous ? (seed + 1) >>> 0 : seed
}

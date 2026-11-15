import type { Pair, Side } from './rules'

// Side is a stable simulation slot. Put a lone human at the bottom when
// projecting those slots onto the court; visible player numbers follow position.
export function bottomSide(ids: Pair<string>): Side {
  return ids[1] === 'keyboard' && ids[0] !== 'keyboard' ? 1 : 0
}

export function topSide(bottom: Side): Side {
  return bottom === 0 ? 1 : 0
}

export function playerNumber(side: Side, bottom: Side) {
  return side === bottom ? 2 : 1
}

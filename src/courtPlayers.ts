import type { Pair, Side } from './rules'

export function bottomSide(ids: Pair<string>): Side {
  return ids[1] === 'keyboard' && ids[0] !== 'keyboard' ? 1 : 0
}

export function playerNumber(side: Side, bottom: Side) {
  return side === bottom ? 2 : 1
}

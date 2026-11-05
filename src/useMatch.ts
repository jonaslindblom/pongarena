import { useEffect, useRef, useState } from 'react'
import {
  advance,
  endedMatch,
  endedRally,
  LoopDetector,
  observation,
  serve,
  TICKS_PER_SECOND,
  type Frame,
  type Pair,
  type Scores,
  type Side,
} from './game'
import { infer, type Controller } from './models'

function newSeed() {
  return crypto.getRandomValues(new Uint32Array(1))[0] % 1000000
}

export function useMatch(
  players: Pair<Controller | undefined>,
  onError: (message: string) => void,
  arenaActive: boolean,
) {
  const [seed, setSeed] = useState(newSeed)
  const history = useRef<Frame[]>([])
  const loops = useRef(new LoopDetector())
  if (history.current.length === 0) history.current.push(serve(seed))
  const cursor = useRef(0),
    revision = useRef(0),
    keys = useRef(new Set<string>())
  const [tick, setTick] = useState(0),
    [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState(1)
  const busy = useRef(false)
  function pause() {
    revision.current++
    setPlaying(false)
  }
  function reset(newSeed = seed) {
    pause()
    setSeed(newSeed)
    history.current = [serve(newSeed)]
    loops.current = new LoopDetector()
    cursor.current = 0
    setTick(0)
    keys.current.clear()
  }
  function seek(at: number) {
    pause()
    cursor.current = Math.max(0, Math.min(at, history.current.length - 1))
    setTick(cursor.current)
  }
  function toggle() {
    if (playing) {
      pause()
      return
    }
    if (!players[0] || !players[1]) return
    if (endedMatch(history.current[cursor.current])) {
      cursor.current = 0
      setTick(0)
    }
    setPlaying(true)
  }
  function human(side: Side): Scores {
    const held = keys.current
    const both = players[0]?.id === 'keyboard' && players[1]?.id === 'keyboard'
    const up =
      ((!both || side === 0) && held.has('KeyW')) || ((!both || side === 1) && held.has('ArrowUp'))
    const down =
      ((!both || side === 0) && held.has('KeyS')) ||
      ((!both || side === 1) && held.has('ArrowDown'))
    return up === down ? [0, 1, 0] : up ? [1, 0, 0] : [0, 0, 1]
  }
  async function step() {
    if (busy.current || !players[0] || !players[1]) return
    const version = revision.current,
      at = cursor.current
    const previous = history.current[at]
    if (endedMatch(previous)) {
      setPlaying(false)
      return
    }
    busy.current = true
    try {
      if (at === history.current.length - 1) {
        let next: Frame
        if (endedRally(previous)) next = serve(seed, previous)
        else {
          const scores: Scores[] = []
          for (const side of [0, 1] as const) {
            const player = players[side]!
            scores.push(
              player.id === 'keyboard'
                ? human(side)
                : await infer(player, observation(previous, side)),
            )
          }
          if (revision.current !== version) return
          const modelMatch = players.every((player) => player?.session && player.id !== 'keyboard')
          next = advance(previous, scores as Pair<Scores>, modelMatch ? loops.current : undefined)
        }
        if (revision.current !== version) return
        history.current.push(next)
      }
      cursor.current = at + 1
      setTick(at + 1)
      if (endedMatch(history.current[at + 1])) setPlaying(false)
    } catch (e) {
      if (revision.current === version) {
        pause()
        onError(e instanceof Error ? e.message : 'Model inference failed.')
      }
    } finally {
      busy.current = false
    }
  }
  // Schedule from the displayed frame; inference is never queued behind a slow frame.
  useEffect(() => {
    if (!playing) return
    let alive = true
    let timer: number
    async function next() {
      await step()
      if (alive && !endedMatch(history.current[cursor.current])) schedule()
    }
    function schedule() {
      const ms = endedRally(history.current[cursor.current]) ? 650 : 1000 / TICKS_PER_SECOND
      timer = window.setTimeout(next, ms / speed)
    }
    schedule()
    return () => {
      alive = false
      clearTimeout(timer)
    }
  }, [playing, speed, players[0], players[1], seed])
  const usesKeyboard = arenaActive && players.some((p) => p?.id === 'keyboard')
  useEffect(() => {
    if (!usesKeyboard) return
    function keydown(e: KeyboardEvent) {
      const target = e.target as HTMLElement
      if (target.closest('input, select, textarea, dialog')) return
      if (['ArrowUp', 'ArrowDown', 'KeyW', 'KeyS'].includes(e.code)) {
        e.preventDefault()
        keys.current.add(e.code)
      }
    }
    const keyup = (e: KeyboardEvent) => keys.current.delete(e.code)
    const blur = () => keys.current.clear()
    window.addEventListener('keydown', keydown)
    window.addEventListener('keyup', keyup)
    window.addEventListener('blur', blur)
    return () => {
      window.removeEventListener('keydown', keydown)
      window.removeEventListener('keyup', keyup)
      window.removeEventListener('blur', blur)
      keys.current.clear()
    }
  }, [usesKeyboard])
  return {
    history: history.current,
    frame: history.current[tick],
    tick,
    seed,
    playing,
    speed,
    setSpeed,
    toggle,
    pause,
    play: () => setPlaying(true),
    seek,
    reset,
    newServes: () => reset(newSeed()),
    step: () => {
      pause()
      void step()
    },
  }
}
export type Match = ReturnType<typeof useMatch>

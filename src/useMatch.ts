import { useEffect, useRef, useState } from 'react'
import {
  endedMatch,
  endedRally,
  type Frame,
  type Pair,
  type Scores,
  type Side,
  type Ruleset,
} from './rules'
import { infer, type Controller } from './models'
import { boundedDeadline, frameDuration, TICK_MS, type Presentation } from './presentation'
import { humanKeyAction } from './humanInput'
import { matchShortcut } from './matchShortcuts'
import { newMatchSeed } from './matchSeed'

export interface MatchTickTiming {
  completedAt: number
  scheduledAt: number
  tick: number
  previousEndedRally: boolean
}

export function useMatch(
  players: Pair<Controller | undefined>,
  onError: (message: string) => void,
  arenaActive: boolean,
  rules: Ruleset,
  onTick?: (sample: MatchTickTiming) => void,
  keyboardShortcuts = false,
) {
  const observer = useRef(onTick)
  observer.current = onTick
  const [seed, setSeed] = useState(newMatchSeed)
  const history = useRef<Frame[]>([])
  const loops = useRef(rules.detector())
  if (history.current.length === 0) history.current.push(rules.serve(seed))
  const cursor = useRef(0),
    revision = useRef(0),
    keys = useRef(new Set<string>()),
    running = useRef(false),
    pending = useRef<Promise<boolean> | null>(null)
  const currentPlayers = useRef(players)
  currentPlayers.current = players
  const humanActions = useRef<Pair<number | null>>([null, null])
  const [position, setPosition] = useState({ tick: 0 }),
    [playing, setPlaying] = useState(false),
    [runVersion, setRunVersion] = useState(0)
  const [started, setStarted] = useState(false)
  const [countdown, setCountdown] = useState<number | null>(null)
  const tick = position.tick
  const setTick = (tick: number) => setPosition({ tick })
  const presentation = useRef<Presentation>({
    from: history.current[0], to: history.current[0], startedAt: 0,
    duration: TICK_MS, animate: false, trail: [],
  })

  function clearHumanInput() {
    keys.current.clear()
    humanActions.current = [null, null]
  }
  function present(animate = false) {
    const at = cursor.current
    const to = history.current[at]
    const from = history.current[Math.max(0, at - 1)]
    presentation.current = {
      from, to, startedAt: performance.now(), duration: TICK_MS,
      animate: animate && from.rally === to.rally && to.event !== 'serve',
      trail: history.current.slice(Math.max(0, at - 8), at + 1),
    }
  }
  function pause() {
    revision.current++
    // React can batch reset()+play() into an unchanged playing=true value.
    // A separate run identity must still discard the old timing deadline.
    setRunVersion((version) => version + 1)
    running.current = false
    setPlaying(false)
    setCountdown(null)
    clearHumanInput()
    present()
  }
  function reset(nextSeed = newMatchSeed(seed)) {
    pause()
    setStarted(false)
    setSeed(nextSeed)
    history.current = [rules.serve(nextSeed)]
    loops.current = rules.detector()
    cursor.current = 0
    setTick(0)
    present()
  }
  function play() {
    if (endedMatch(history.current[cursor.current])) reset()
    running.current = true
    setStarted(true)
    setPlaying(true)
  }
  function restart() {
    if (!players[0] || !players[1]) return
    reset(seed)
    play()
  }
  function toggle() {
    if (running.current) pause()
    else if (players[0] && players[1]) play()
  }
  function setHumanAction(side: Side, action: 0 | 1 | 2 | null) {
    humanActions.current[side] = running.current ? action : null
  }
  function human(side: Side): Scores {
    const both = players[0]?.id === 'keyboard' && players[1]?.id === 'keyboard'
    const action = humanKeyAction(keys.current, side, both) ?? humanActions.current[side] ?? 1
    return Array.from({ length: rules.outputSize }, (_, index) => index === action ? 1 : 0) as Scores
  }
  function advanceOne(scheduledAt: number): Promise<boolean> {
    if (pending.current || !players[0] || !players[1]) return Promise.resolve(false)
    const version = revision.current, at = cursor.current
    const previous = history.current[at]
    if (endedMatch(previous)) return Promise.resolve(false)
    const job = (async () => {
      try {
        let next: Frame
        if (endedRally(previous)) next = rules.serve(seed, previous)
        else {
          // Human input is sampled with the same pre-tick state as both models,
          // before either asynchronous inference has a chance to finish.
          const scores = players.map((player, side) => player!.id === 'keyboard'
            ? human(side as Side) : null) as Pair<Scores | null>
          for (const side of [0, 1] as const) {
            if (!scores[side]) scores[side] = await infer(players[side]!, rules.observation(previous, side))
            if (revision.current !== version) return false
          }
          if (players.some((player, side) => player !== currentPlayers.current[side])) return false
          const automaticMatch = players.every((player) => player && player.type !== 'human')
          next = rules.advance(previous, scores as Pair<Scores>, automaticMatch ? loops.current : undefined)
        }
        if (revision.current !== version) return false
        history.current.push(next)
        cursor.current = at + 1
        present(true)
        setTick(at + 1)
        observer.current?.({
          completedAt: performance.now(), scheduledAt, tick: at + 1,
          previousEndedRally: endedRally(previous),
        })
        if (endedMatch(history.current[at + 1])) {
          running.current = false
          setPlaying(false)
          clearHumanInput()
        }
        return true
      } catch (e) {
        if (revision.current === version) {
          pause()
          onError(e instanceof Error ? e.message : 'Model inference failed.')
        }
        return false
      }
    })()
    pending.current = job
    void job.then(() => { if (pending.current === job) pending.current = null })
    return job
  }

  useEffect(() => {
    if (!playing || !arenaActive || !players[0] || !players[1]) return
    let alive = true, pumping = false, raf = 0
    const humanMatch = players.some((player) => player?.type === 'human')
    let serveAt: number | null = null
    let due = performance.now() + frameDuration(history.current[cursor.current])
    function prepareServe(now: number) {
      if (humanMatch && history.current[cursor.current].event === 'serve') {
        serveAt = now + 3000
        setCountdown(3)
      }
    }
    prepareServe(performance.now())
    async function pump(now: number) {
      if (pumping || pending.current) return
      if (serveAt !== null) {
        if (now < serveAt) {
          setCountdown(Math.ceil((serveAt - now) / 1000))
          return
        }
        serveAt = null
        setCountdown(null)
        due = now
      }
      pumping = true
      // Deadlines follow the simulation cadence rather than inference completion.
      // At most four states are advanced per animation callback after a stall.
      due = boundedDeadline(due, now)
      for (let count = 0; count < 4 && alive && running.current && now >= due; count++) {
        if (!await advanceOne(due)) break
        due += frameDuration(history.current[cursor.current])
        prepareServe(performance.now())
        if (serveAt !== null) break
      }
      pumping = false
    }
    function animate(now: number) {
      if (!alive || !running.current) return
      raf = requestAnimationFrame(animate)
      void pump(now)
    }
    raf = requestAnimationFrame(animate)
    return () => {
      alive = false
      cancelAnimationFrame(raf)
    }
  }, [playing, runVersion, players[0], players[1], seed, arenaActive, rules])

  useEffect(() => { reset() }, [rules])

  useEffect(() => {
    if (!arenaActive || !keyboardShortcuts) return
    function keydown(event: KeyboardEvent) {
      const shortcut = matchShortcut(event)
      if (!shortcut || shortcut === 'fullscreen' || !players[0] || !players[1]) return
      event.preventDefault()
      if (shortcut === 'play') toggle()
      else restart()
    }
    window.addEventListener('keydown', keydown)
    return () => window.removeEventListener('keydown', keydown)
  }, [arenaActive, keyboardShortcuts, players[0], players[1], seed, rules])

  const usesKeyboard = arenaActive && players.some((p) => p?.id === 'keyboard')
  useEffect(() => {
    if (!usesKeyboard) return
    function keydown(e: KeyboardEvent) {
      const target = e.target as HTMLElement
      if (target.closest('input, select, textarea, dialog, [contenteditable="true"]')) return
      if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'KeyA', 'KeyD', 'KeyW', 'KeyS'].includes(e.code)) {
        e.preventDefault()
        keys.current.add(e.code)
      }
    }
    const keyup = (e: KeyboardEvent) => keys.current.delete(e.code)
    window.addEventListener('keydown', keydown)
    window.addEventListener('keyup', keyup)
    return () => {
      window.removeEventListener('keydown', keydown)
      window.removeEventListener('keyup', keyup)
      clearHumanInput()
    }
  }, [usesKeyboard])
  useEffect(() => {
    const visibility = () => { if (document.hidden) pause() }
    const blur = () => {
      clearHumanInput()
      if (currentPlayers.current.some((p) => p?.id === 'keyboard')) pause()
    }
    document.addEventListener('visibilitychange', visibility)
    window.addEventListener('blur', blur)
    return () => {
      document.removeEventListener('visibilitychange', visibility)
      window.removeEventListener('blur', blur)
    }
  }, [])
  useEffect(() => () => {
    revision.current++
    running.current = false
    clearHumanInput()
  }, [])
  return {
    history: history.current,
    frame: history.current[tick],
    presentation,
    tick,
    seed,
    playing,
    started,
    countdown,
    setHumanAction,
    rules,
    toggle,
    pause,
    play,
    reset,
    restart,
  }
}
export type Match = ReturnType<typeof useMatch>

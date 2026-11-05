export interface StartupSnapshot {
  pageId: string
  startedAt: string
  build: { id: string; mode: string }
  navigationType: string
  instrumentationStartedMs: number
  arenaReadyMs: number | null
  libraryReadyMs: number | null
  hiddenDuringStartup: boolean
  failures: string[]
  timings: { name: string; durationMs: number }[]
}

// performance.now() is relative to navigation, so readiness includes the initial
// document, JavaScript and WASM load, not just time spent inside React.
const startup: StartupSnapshot = {
  pageId: String(performance.timeOrigin),
  startedAt: new Date(performance.timeOrigin).toISOString(),
  build: { id: import.meta.env.VITE_SOURCE_ID, mode: import.meta.env.DEV ? 'development' : 'production' },
  navigationType: (performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming | undefined)?.type ?? 'unknown',
  instrumentationStartedMs: performance.now(),
  arenaReadyMs: null,
  libraryReadyMs: null,
  hiddenDuringStartup: document.hidden,
  failures: [],
  timings: [],
}
const visibility = () => { if (document.hidden) startup.hiddenDuringStartup = true }
document.addEventListener('visibilitychange', visibility)

export function getStartupSnapshot(): StartupSnapshot {
  return { ...startup, build: { ...startup.build }, failures: [...startup.failures],
    timings: startup.timings.map((timing) => ({ ...timing })) }
}

export function recordStartupTiming(name: string, startedAt: number) {
  startup.timings.push({ name, durationMs: performance.now() - startedAt })
}

export function recordStartupFailure(message: string) {
  startup.failures.push(message)
}

export function markArenaReady() {
  startup.arenaReadyMs ??= performance.now()
}

export function finishStartup(success: boolean) {
  if (success) startup.libraryReadyMs ??= performance.now()
  document.removeEventListener('visibilitychange', visibility)
}

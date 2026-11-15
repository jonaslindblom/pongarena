export function matchShortcut(event: KeyboardEvent): 'play' | 'restart' | 'fullscreen' | null {
  const action = event.code === 'KeyP' ? 'play'
    : event.code === 'KeyR' ? 'restart'
      : event.code === 'KeyF' ? 'fullscreen' : null
  if (!action || event.repeat || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey || event.defaultPrevented) return null
  if (document.querySelector('dialog[open]')) return null
  const target = event.target
  if (target instanceof Element && target.closest('input, select, textarea, dialog, [contenteditable]:not([contenteditable="false"])')) return null
  return action
}

import type { ModelOrigin as Origin } from './modelMetadata'
import './ModelOrigin.css'

const labels = { core: 'Core', community: 'Community', local: 'Local' }
const descriptions = {
  core: 'PongArena reference model and teaching example.',
  community: 'An accepted community contribution, evaluated under the same league rules.',
  local: 'Imported into this browser.',
}

export function ModelOrigin({ origin }: { origin?: Origin }) {
  return origin ? <span className="model-origin" data-origin={origin} title={descriptions[origin]}>
    {labels[origin]}
  </span> : null
}

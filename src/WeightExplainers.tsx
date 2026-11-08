import { WeightsLab } from './WeightsLab'
import { CounterWeightsLab } from './CounterWeightsLab'
import type { Controller } from './models'

export type ExplainerId = 'follow' | 'counter97'

export function WeightExplainers({ models, selected, onSelect, onPlay }: {
  models: Controller[]
  selected: ExplainerId
  onSelect: (id: ExplainerId) => void
  onPlay: (id: string) => void
}) {
  return <>
    <div className="weight-explainer-picker" role="group" aria-label="Choose a weight explainer">
      {([['follow', 'Basics · Follow'], ['counter97', 'Building behavior · Counter97']] as const).map(([id, name]) =>
        <button key={id} className="secondary-btn" aria-pressed={selected === id} onClick={() => onSelect(id)}>{name}</button>)}
    </div>
    {selected === 'follow'
      ? <WeightsLab model={models.find(model => model.id === 'follow')} onPlay={() => onPlay('follow')} />
      : <CounterWeightsLab model={models.find(model => model.id === 'counter97')} onPlay={() => onPlay('counter97')} />}
  </>
}

import { WeightsLab } from './WeightsLab'
import { CounterWeightsLab } from './CounterWeightsLab'
import { OneShotExplainer } from './OneShotExplainer'
import type { Controller } from './models'

export type ExplainerId = 'follow' | 'counter' | 'oneshot'

export function WeightExplainers({ models, selected, onSelect, onPlay }: {
  models: Controller[]
  selected: ExplainerId
  onSelect: (id: ExplainerId) => void
  onPlay: (id: string) => void
}) {
  return <>
    <div className="weight-explainer-picker" role="group" aria-label="Choose a lesson">
      {([['follow', '1 · Follow the ball'], ['counter', '2 · Know when to return'], ['oneshot', '3 · Write weights in one shot']] as const).map(([id, name]) =>
        <button key={id} className="secondary-btn" aria-pressed={selected === id} onClick={() => onSelect(id)}>{name}</button>)}
    </div>
    {selected === 'follow' && <WeightsLab model={models.find(model => model.id === 'follow')} onPlay={() => onPlay('follow')} onNext={() => { onSelect('counter'); window.scrollTo({ top: 0 }) }} />}
    {selected === 'counter' && <CounterWeightsLab model={models.find(model => model.id === 'counter')} onPlay={() => onPlay('counter')} />}
    {selected === 'oneshot' && <OneShotExplainer model={models.find(model => model.id === 'oneshot')} onPlay={() => onPlay('oneshot')} />}
  </>
}

import { WeightsLab } from './WeightsLab'
import { AttackWeightsLab } from './AttackWeightsLab'
import type { Controller } from './models'
import type { AttackModelId } from './attackExplanation'

export type ExplainerId = 'follow' | AttackModelId

export function WeightExplainers({ models, selected, onSelect, onPlay }: {
  models: Controller[]
  selected: ExplainerId
  onSelect: (id: ExplainerId) => void
  onPlay: (id: string) => void
}) {
  return <>
    <div className="weight-explainer-picker" role="group" aria-label="Choose a weight explainer">
      {([['follow', 'Follow', 13], ['attack75', 'Attack75', 75], ['counter97', 'Counter97', 97]] as const).map(([id, name, params]) =>
        <button key={id} className="secondary-btn" aria-pressed={selected === id} onClick={() => onSelect(id)}>{name} · {params} parameters</button>)}
    </div>
    {selected === 'follow'
      ? <WeightsLab model={models.find(model => model.id === 'follow')} onPlay={() => onPlay('follow')} />
      : <AttackWeightsLab selected={selected} attack={models.find(model => model.id === 'attack75')} counter={models.find(model => model.id === 'counter97')} onPlay={() => onPlay(selected)} />}
  </>
}

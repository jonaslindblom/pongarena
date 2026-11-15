import { ArrowRight } from 'lucide-react'
import type { MouseEvent } from 'react'
import { WeightsLab } from './WeightsLab'
import { CounterWeightsLab } from './CounterWeightsLab'
import type { Controller } from './models'

export type ExplainerId = 'follow' | 'counter'

export function WeightExplainers({ models, selected, onSelect, onFinding, onPlay }: {
  models: Controller[]
  selected: ExplainerId
  onSelect: (event: MouseEvent<HTMLAnchorElement>, id: ExplainerId) => void
  onFinding: (event: MouseEvent<HTMLAnchorElement>) => void
  onPlay: (id: string) => void
}) {
  return <>
    <div className="reading-picker" role="group" aria-label="Choose a lesson">
      <p className="section-code">HOW IT THINKS · TWO INTERACTIVE LESSONS</p>
      <div className="reading-options">
        {([
          ['follow', 'Follow the ball', 'Trace one neuron from inputs to a move.', '#how-it-thinks'],
          ['counter', 'Know when to return', 'Switch off a signal and change the strategy.', '#how-it-thinks/counter'],
        ] as const).map(([id, name, description, href], index) => <a key={id} href={href}
          aria-label={`${index + 1} · ${name}`} aria-describedby={`lesson-${id}-description`}
          aria-current={selected === id ? 'page' : undefined} onClick={event => onSelect(event, id)}>
          <span className="reading-position">LESSON {index + 1} OF 2 {selected === id && <span>READING</span>}</span>
          <strong>{name}<ArrowRight size={17} aria-hidden="true" /></strong>
          <span id={`lesson-${id}-description`} className="reading-description">{description}</span>
        </a>)}
      </div>
    </div>
    {selected === 'follow' && <WeightsLab model={models.find(model => model.id === 'follow')} onPlay={() => onPlay('follow')} />}
    {selected === 'counter' && <CounterWeightsLab model={models.find(model => model.id === 'counter')} onPlay={() => onPlay('counter')} />}
    {selected === 'follow'
      ? <a className="lesson-next" href="#how-it-thinks/counter" onClick={event => onSelect(event, 'counter')}>
        <span><span className="section-code">NEXT LESSON · 2 OF 2</span><strong>Know when to return</strong><span>Following is only the start. Switch off Counter’s departure signal and see what changes.</span></span>
        <span>Continue <ArrowRight size={18} aria-hidden="true" /></span>
      </a>
      : <a className="lesson-next" href="#findings/oneshot" onClick={onFinding}>
        <span><span className="section-code">NEXT · FROM HOW IT THINKS TO FINDINGS</span><strong>From a mechanism to a prompt</strong><span>Describe prediction, recovery and aiming in words. Did those ideas become behavior in the weights?</span></span>
        <span>See the evidence <ArrowRight size={18} aria-hidden="true" /></span>
      </a>}
  </>
}

import { Code2, Cpu, Keyboard } from 'lucide-react'
import { DIVISIONS, smallestDivision } from './competition'
import type { Controller } from './models'
import './ControllerBadge.css'

export function controllerOptionLabel(model: Controller) {
  if (model.type === 'human') return model.name + ' · Human'
  if (model.type === 'script') return model.name + ' · Script'
  const division = DIVISIONS.find((d) => d.id === smallestDivision(model))
  return `${model.name} · ${division?.name ?? 'Unclassified'} weights · ${model.params.toLocaleString()} params`
}

export function ControllerBadge({ model, compact = false }: { model?: Controller; compact?: boolean }) {
  if (!model) return null
  const weights = model.type === 'weights'
  const division = weights ? DIVISIONS.find((d) => d.id === smallestDivision(model)) : undefined
  const Icon = model.type === 'human' ? Keyboard : model.type === 'script' ? Code2 : Cpu
  const label = model.type === 'human' ? 'Human' : model.type === 'script' ? 'Script' : 'Weights'
  const count = compact && model.params >= 1000
    ? (model.params / 1000).toFixed(1).replace(/\.0$/, '') + 'k'
    : model.params.toLocaleString()
  return (
    <span className={'controller-identity' + (compact ? ' compact' : '')} data-controller-type={model.type}>
      <span className="controller-badge"><Icon size={11} />{label}</span>
      {weights && (
        <span className="controller-size">
          <span className="controller-class" data-division={division?.id}
            title={division ? `${division.name} class: up to ${division.maxParams.toLocaleString()} parameters and ${division.maxMacs.toLocaleString()} MACs` : 'Outside the competition class budgets'}>
            {division?.name ?? 'Unclassified'}
          </span>
          <span className="controller-params" title={model.params.toLocaleString() + ' parameters'}>{count}{compact ? '' : ' params'}</span>
        </span>
      )}
    </span>
  )
}

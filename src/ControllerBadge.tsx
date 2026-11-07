import { Cpu, Keyboard } from 'lucide-react'
import { DIVISIONS, smallestDivision } from './competition'
import type { Controller } from './models'
import './ControllerBadge.css'

export function controllerOptionLabel(model: Controller) {
  if (model.type === 'human') return model.name + ' · Human'
  const division = DIVISIONS.find((d) => d.id === smallestDivision(model))
  return `${model.name} · ${division?.name ?? 'Unclassified'} ${model.handcrafted ? 'handwritten weights' : 'weights'} · ${model.params.toLocaleString()} params`
}

export function ControllerBadge({ model, compact = false }: { model?: Controller; compact?: boolean }) {
  if (!model) return null
  const weights = model.type === 'weights'
  const handcrafted = weights && model.handcrafted
  const division = weights ? DIVISIONS.find((d) => d.id === smallestDivision(model)) : undefined
  const Icon = model.type === 'human' ? Keyboard : Cpu
  const label = model.type === 'human' ? 'Human' : handcrafted ? 'Handwritten' : 'Weights'
  const count = compact && model.params >= 1000
    ? (model.params / 1000).toFixed(1).replace(/\.0$/, '') + 'k'
    : model.params.toLocaleString()
  return (
    <span className={'controller-identity' + (compact ? ' compact' : '')} data-controller-type={model.type}
      data-weight-origin={handcrafted ? 'handcrafted' : undefined}>
      <span className="controller-badge" title={handcrafted ? 'Manually designed weights, generated from fixed formulas. No training.' : undefined}>
        <Icon size={11} aria-hidden="true" />{label}
      </span>
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

import rulesOnlyUrl from '../experiments/one-prompt-weights/follow-up/models/rules-163-a.onnx?url'
import promptTemplate from '../experiments/one-prompt-weights/prompt.txt?raw'
import tacticalBrief from '../experiments/one-prompt-weights/follow-up/tactical-brief.txt?raw'
import probeReport from '../experiments/one-prompt-weights/follow-up/probes.json?raw'
import { assetPath } from './assets'

export { rulesOnlyUrl }
export { oneShotResults, oneShotRalliesAgainstLarge } from './findingsData'
export const oneShotUrl = assetPath('models/oneshot.onnx')

export const rulesOnlyPrompt = promptTemplate.replace('{{WIDTHS}}', '6 → 12 → 4 → 3 → 3')
export const oneShotPrompt = rulesOnlyPrompt + '\n' + tacticalBrief

const probes: { id: string; pairsPerInput: number; velocityFlips: number; opponentFlips: number }[] = JSON.parse(probeReport)
export const oneShotProbes = [
  ['rules-163-a', 'Rules only · run 1'],
  ['rules-163-b', 'Rules only · run 2'],
  ['tactical-163-a', 'Rules + brief · run 1 (OneShot)'],
  ['tactical-163-b', 'Rules + brief · run 2'],
].map(([id, label]) => {
  const probe = probes.find(probe => probe.id === id)
  if (!probe) throw new Error('Missing saved input probe: ' + id)
  return { ...probe, label }
})

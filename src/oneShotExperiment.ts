import rulesOnlyUrl from '../experiments/one-prompt-weights/follow-up/models/rules-163-a.onnx?url'
import promptTemplate from '../experiments/one-prompt-weights/prompt.txt?raw'
import tacticalBrief from '../experiments/one-prompt-weights/follow-up/tactical-brief.txt?raw'
import { assetPath } from './assets'

export { rulesOnlyUrl }
export const oneShotUrl = assetPath('models/oneshot.onnx')

export const rulesOnlyPrompt = promptTemplate.replace('{{WIDTHS}}', '6 → 12 → 4 → 3 → 3')
export const oneShotPrompt = rulesOnlyPrompt + '\n' + tacticalBrief
export const oneShotRalliesAgainstLarge = 28

// Paired league points out of 64, in Follow, Small, Predict, Attack, Counter, Quill, Large order.
// Source: experiments/one-prompt-weights/follow-up/results/.
export const oneShotResults = [
  { label: 'Rules only · run 1', points: [64, 36, 63, 44.5, 0.5, 0, 0] },
  { label: 'Rules only · run 2', points: [62.5, 10.5, 0.5, 0, 0, 0, 0] },
  { label: 'Rules + brief · run 1 (OneShot)', points: [64, 26.5, 63, 60, 3.5, 1.5, 0] },
  { label: 'Rules + brief · run 2', points: [63.5, 14, 64, 64, 0, 0, 0] },
] as const

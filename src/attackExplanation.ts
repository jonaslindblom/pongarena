import pb from 'onnx-proto'
import { createBrainGraph, type BrainLayer } from './brain'
import { sha256 } from './competition'
import { choice, type Frame, type Scores } from './game'
import { validateArenaModel } from './validate'

export type AttackModelId = 'attack75' | 'counter97'

export const ATTACK_SCENARIOS: {
  id: string; title: string; description: string
  frame: Pick<Frame, 'x' | 'y' | 'vx' | 'vy' | 'paddles'>
}[] = [
  { id: 'outgoing', title: 'Recover after a return', description: 'The ball is leaving. Follow its forecast or return to center?',
    frame: { x: 184, y: 120, vx: 13, vy: 0, paddles: [100, 80] } },
  { id: 'long', title: 'Anticipate a long flight', description: 'The longer forecast changes which way the paddle moves.',
    frame: { x: 240, y: 80, vx: -13, vy: 3, paddles: [130, 80] } },
  { id: 'bounce', title: 'See beyond a wall bounce', description: 'Both models reflect the forecast, but predict different landing positions.',
    frame: { x: 240, y: 130, vx: -13, vy: 3, paddles: [140, 80] } },
  { id: 'corner', title: 'Choose the open corner', description: 'Move the opponent across center to reverse the intended shot.',
    frame: { x: 16, y: 80, vx: -13, vy: 0, paddles: [80, 80] } },
]

const inputRoles = ['Ball depth', 'Ball position', 'Depth velocity', 'Sideways velocity', 'Paddle position', 'Opponent position']
const outputRoles = ['Left score', 'Stay score', 'Right score']
const wallRoles = ['Target error carrier', 'Right wall hinge', 'Left wall hinge']
const models = {
  attack75: {
    hash: '646d1a1365e3e307bd4431fc73930c5efa19de16b7cc22bbb9200608c2b720f8',
    widths: [6, 6, 3, 3],
    roles: [
      ['Positive forecast hinge', 'Negative forecast hinge', 'Signed height carrier', 'Paddle carrier', 'Opponent ramp', 'Opponent cap hinge'],
      wallRoles, outputRoles,
    ],
  },
  counter97: {
    hash: '5f7a480b98c982cf403f2f8849fac1e5e02503ff12e6b0b5d1e377856534e35c',
    widths: [6, 6, 4, 3, 3],
    roles: [
      ['Positive forecast hinge', 'Negative forecast hinge', 'Signed height carrier', 'Paddle carrier', 'Outgoing detector', 'Opponent ramp'],
      ['Gated forecast carrier', 'Paddle and aiming carrier', 'Incoming gate', 'Opponent cap and recovery'],
      wallRoles, outputRoles,
    ],
  },
} satisfies Record<AttackModelId, { hash: string; widths: number[]; roles: string[][] }>

export interface AttackDenseLayer {
  inputName: string
  linearName: string
  outputName: string
  inputWidth: number
  width: number
  activation: 'Relu' | null
  weights: number[][]
  biases: number[]
  roles: string[]
}

export interface AttackInspection {
  id: AttackModelId
  // Identity of the frozen source model, before teaching outputs or interventions.
  sha256: string
  intervention?: 'outgoing-detector-muted'
  bytes: Uint8Array
  layers: BrainLayer[]
  denseLayers: AttackDenseLayer[]
  params: number
  macs: number
}

// These circuit labels describe specific frozen weights, not arbitrary models
// that happen to share their layer sizes. Expose affine values as well as ReLU
// outputs so the explanation reads real inference instead of running a policy.
export async function inspectAttackModel(bytes: Uint8Array, id: AttackModelId): Promise<AttackInspection> {
  const expected = models[id]
  if (!expected) throw new Error('Choose Attack or Counter for this explanation.')
  const stats = validateArenaModel(bytes)
  if (stats.architecture.widths.join(',') !== expected.widths.join(',') ||
    stats.architecture.operations.some(operation => operation.op !== 'Gemm' && operation.op !== 'Relu'))
    throw new Error(`This explanation needs ${id}'s ${expected.widths.join(' → ')} ReLU architecture.`)
  const hash = await sha256(bytes)
  if (hash !== expected.hash) throw new Error(`This explanation needs the frozen ${id} weights; the model has changed.`)
  const brain = createBrainGraph(bytes)
  const model = pb.onnx.ModelProto.decode(brain.bytes)
  const graph = model.graph!
  const nodes = graph.node!.filter(node => node.opType === 'Gemm')
  function tensorValues(name: string) {
    const tensor = graph.initializer!.find(tensor => tensor.name === name)!
    const raw = tensor.rawData
    if (!raw?.length) return Array.from(tensor.floatData ?? [])
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength)
    return Array.from({ length: raw.byteLength / 4 }, (_, i) => view.getFloat32(i * 4, true))
  }
  const denseLayers = nodes.map((node, layer): AttackDenseLayer => {
    const inputWidth = expected.widths[layer], width = expected.widths[layer + 1]
    const linearName = node.output![0]!
    const outputName = brain.layers[layer + 1].name
    const flat = tensorValues(node.input![1])
    const transposed = node.attribute?.some(attribute => attribute.name === 'transB' && Number(attribute.i) === 1)
    const weights = Array.from({ length: width }, (_, output) =>
      Array.from({ length: inputWidth }, (_, input) => flat[transposed ? output * inputWidth + input : input * width + output]))
    const final = layer === nodes.length - 1
    if (!final) graph.output!.push({ name: linearName, type: { tensorType: {
      elemType: 1, shape: { dim: [{ dimValue: 1 }, { dimValue: width }] },
    } } })
    return { inputName: node.input![0], linearName, outputName, inputWidth, width,
      activation: final ? null : 'Relu', weights, biases: tensorValues(node.input![2]), roles: [...expected.roles[layer]] }
  })
  return { id, sha256: hash, bytes: pb.onnx.ModelProto.encode(model).finish(), layers: brain.layers,
    denseLayers, params: stats.params, macs: stats.macs }
}

// Mute H1.5 on an isolated teaching copy. Removing this detector lets all of its
// downstream connections read zero; it does not recreate Attack's policy.
export function createCounterIntervention(inspection: AttackInspection): AttackInspection {
  if (inspection.id !== 'counter97' || inspection.sha256 !== models.counter97.hash || inspection.intervention)
    throw new Error('The outgoing-detector experiment needs an unmodified Counter inspection.')
  const model = pb.onnx.ModelProto.decode(Uint8Array.from(inspection.bytes))
  const first = model.graph!.node!.find(node => node.opType === 'Gemm')!
  const transposed = first.attribute?.some(attribute => attribute.name === 'transB' && Number(attribute.i) === 1)
  const denseLayers = inspection.denseLayers.map(layer => ({ ...layer,
    weights: layer.weights.map(row => [...row]), biases: [...layer.biases], roles: [...layer.roles],
  }))
  const dense = denseLayers[0], neuron = 4
  const weightTensor = model.graph!.initializer!.find(tensor => tensor.name === first.input![1])!
  const biasTensor = model.graph!.initializer!.find(tensor => tensor.name === first.input![2])!
  function replaceTensor(tensor: pb.onnx.ITensorProto, flat: number[]) {
    const raw = new Uint8Array(flat.length * 4)
    const view = new DataView(raw.buffer)
    flat.forEach((value, i) => view.setFloat32(i * 4, value, true))
    tensor.rawData = raw
    tensor.floatData = []
  }
  dense.weights[neuron].fill(0)
  dense.biases[neuron] = 0
  replaceTensor(weightTensor, transposed ? dense.weights.flat() :
    Array.from({ length: dense.inputWidth }, (_, i) => dense.weights.map(row => row[i])).flat())
  replaceTensor(biasTensor, dense.biases)
  return { ...inspection, intervention: 'outgoing-detector-muted',
    bytes: pb.onnx.ModelProto.encode(model).finish(), denseLayers }
}

export interface AttackNeuron {
  layer: number
  index: number
  role: string
  value: number
  preActivation: number
  bias: number
  contributions: { inputIndex: number; inputRole: string; input: number; weight: number; product: number }[]
}

export interface AttackExplanation {
  chosen: number
  scores: Scores
  forecast: number
  activeForecast: number | null
  reflected: number | null
  cornerTarget: number | null
  aimSignal: number | null
  target: number
  aimOffset: number
  error: number
  incomingGate: number | null
  neurons: AttackNeuron[]
}

function values(value: ArrayLike<number> | undefined, width: number, name: string) {
  if (!value || value.length !== width || Array.from(value).some(value => !Number.isFinite(value)))
    throw new Error(`The explanation needs ${width} finite values for ${name}.`)
  return Array.from(value)
}

export function getAttackExplanation(id: AttackModelId, input: ArrayLike<number>,
  activations: Readonly<Record<string, ArrayLike<number>>>, scores: ArrayLike<number>,
  inspection: AttackInspection): AttackExplanation {
  if (inspection.id !== id) throw new Error('The explanation and inspected model must match.')
  const observation = values(input, 6, 'observation')
  const actionScores = values(scores, 3, 'action_scores') as Scores
  if (activations.action_scores && values(activations.action_scores, 3, 'action_scores')
    .some((score, i) => score !== actionScores[i])) throw new Error('The explanation received different action scores.')
  const dense = inspection.denseLayers
  const hidden = dense.slice(0, -1).map(layer => values(activations[layer.outputName], layer.width, layer.outputName))
  const linear = dense.map(layer => layer.activation
    ? values(activations[layer.linearName], layer.width, layer.linearName) : actionScores)
  const neurons = dense.flatMap((layer, layerIndex) => {
    const previous = layerIndex ? hidden[layerIndex - 1] : observation
    const previousRoles = layerIndex ? dense[layerIndex - 1].roles : inputRoles
    const output = layer.activation ? hidden[layerIndex] : actionScores
    return layer.roles.map((role, index): AttackNeuron => ({ layer: layerIndex, index, role,
      value: output[index], preActivation: linear[layerIndex][index], bias: layer.biases[index],
      contributions: layer.weights[index].map((weight, inputIndex) => ({ inputIndex,
        inputRole: previousRoles[inputIndex], input: previous[inputIndex], weight, product: weight * previous[inputIndex] })),
    }))
  })

  const counter = id === 'counter97', foldLayer = dense.length - 2
  const incomingGate = counter ? hidden[1][2] : null
  const recovering = incomingGate === 0
  // Undo only the stored carry/threshold offsets on observed affine tensors.
  const active = linear[foldLayer][1] - dense[foldLayer].biases[1] - (counter ? 0 : dense[0].biases[2])
  const forecast = counter
    ? linear[1][0] - dense[1].weights[0][4] * hidden[0][4] - dense[1].biases[0] - dense[0].biases[2]
    : active
  const folded = active - 2 * hidden[foldLayer][1] + 2 * hidden[foldLayer][2]
  const aimSignal = recovering ? null : counter ? hidden[0][5] - hidden[1][3] - 1 : hidden[0][4] - hidden[0][5] - 1
  const error = actionScores[2] * 80
  const target = 80 + observation[4] * 80 + error
  const reflected = recovering ? null : 80 + folded * 80
  return { chosen: choice(actionScores), scores: actionScores, forecast: 80 + forecast * 80,
    activeForecast: recovering ? null : 80 + active * 80, reflected,
    cornerTarget: aimSignal === null ? null : 80 - 76 * aimSignal, aimSignal,
    target, aimOffset: target - (reflected ?? 80), error, incomingGate, neurons }
}

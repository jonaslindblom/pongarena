import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import pb from 'onnx-proto'
import * as ort from 'onnxruntime-web/wasm'
import { createCounterIntervention, getAttackExplanation, inspectAttackModel, type AttackModelId } from '../src/attackExplanation'
import { observation, serve, type Frame } from '../src/game'

const scenarios: { id: string; frame: Pick<Frame, 'x' | 'y' | 'vx' | 'vy' | 'paddles'> }[] = [
  { id: 'outgoing', frame: { x: 184, y: 120, vx: 13, vy: 0, paddles: [100, 80] } },
  { id: 'long', frame: { x: 240, y: 80, vx: -13, vy: 3, paddles: [130, 80] } },
  { id: 'bounce', frame: { x: 240, y: 130, vx: -13, vy: 3, paddles: [140, 80] } },
  { id: 'corner', frame: { x: 16, y: 80, vx: -13, vy: 0, paddles: [80, 80] } },
]

const ids = ['attack', 'counter'] as const
const bytes: Record<AttackModelId, Uint8Array> = {
  attack: readFileSync(new URL('../public/models/attack.onnx', import.meta.url)),
  counter: readFileSync(new URL('../public/models/counter.onnx', import.meta.url)),
}
const close = (actual: number, expected: number, message: string, tolerance = .0002) =>
  assert(Math.abs(actual - expected) < tolerance, `${message}: expected ${expected}, received ${actual}`)

test('explanation inspection preserves the frozen policy and exposes its real affine and ReLU tensors', async () => {
  for (const id of ids) {
    const original = Uint8Array.from(bytes[id])
    const before = pb.onnx.ModelProto.decode(bytes[id])
    const inspection = await inspectAttackModel(bytes[id], id)
    const after = pb.onnx.ModelProto.decode(inspection.bytes)
    assert.deepEqual(Uint8Array.from(bytes[id]), original, 'Inspection must not mutate the playable model')
    assert.deepEqual(after.graph!.node, before.graph!.node)
    assert.deepEqual(after.graph!.initializer, before.graph!.initializer,
      'Teaching labels must never replace or rewrite stored weights')
    assert.equal(inspection.params, id === 'attack' ? 75 : 97)
    assert.equal(inspection.macs, id === 'attack' ? 63 : 81)
    assert.equal(inspection.denseLayers.reduce((sum, layer) =>
      sum + layer.weights.flat().length + layer.biases.length, 0), inspection.params,
    'The inspector must expose every stored zero and bias, not only active connections')
    const outputNames = after.graph!.output!.map(output => output.name)
    for (const layer of inspection.denseLayers) {
      assert(outputNames.includes(layer.linearName), 'Affine values must come from ONNX outputs')
      assert(outputNames.includes(layer.outputName), 'Post-ReLU values must also remain visible')
      assert.equal(layer.roles.length, layer.width)
    }
  }
})

test('circuit labels reject malformed files, the wrong architecture, and changed weights of the same shape', async () => {
  await assert.rejects(inspectAttackModel(new Uint8Array([1, 2, 3]), 'attack'), /plain dense network/)
  await assert.rejects(inspectAttackModel(bytes.counter, 'attack'), /architecture/)
  await assert.rejects(inspectAttackModel(bytes.attack, 'counter'), /architecture/)
  const altered = pb.onnx.ModelProto.decode(Uint8Array.from(bytes.attack))
  const raw = altered.graph!.initializer![0].rawData!
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength)
  view.setFloat32(0, view.getFloat32(0, true) + .25, true)
  await assert.rejects(inspectAttackModel(pb.onnx.ModelProto.encode(altered).finish(), 'attack'), /frozen.*weights/,
    'A shape-compatible model must not receive explanations for different weights')
})

test('the same legal scenarios expose opposite decisions without changing scores, and every neuron can be accounted for', async () => {
  ort.env.wasm.numThreads = 1
  const explanations = new Map<string, ReturnType<typeof getAttackExplanation>>()
  for (const id of ids) {
    const inspection = await inspectAttackModel(bytes[id], id)
    const original = await ort.InferenceSession.create(bytes[id], { executionProviders: ['wasm'] })
    const instrumented = await ort.InferenceSession.create(inspection.bytes, { executionProviders: ['wasm'] })
    try {
      for (const scenario of scenarios) {
        const input = observation({ ...serve(1), ...scenario.frame }, 0)
        const feed = { observation: new ort.Tensor('float32', input, [1, 6]) }
        const result = await instrumented.run(feed)
        const originalResult = await original.run(feed)
        const activations = Object.fromEntries(Object.entries(result).map(([name, tensor]) =>
          [name, Array.from(tensor.data as Float32Array)]))
        assert.deepEqual(activations.action_scores, Array.from(originalResult.action_scores.data as Float32Array),
          'Adding teaching outputs must preserve every original score, not merely the chosen action')
        const explanation = getAttackExplanation(id, input, activations, activations.action_scores, inspection)
        explanations.set(`${id}:${scenario.id}`, explanation)
        for (const neuron of explanation.neurons) {
          close(neuron.bias + neuron.contributions.reduce((sum, part) => sum + part.product, 0), neuron.preActivation,
            'Stored weights, actual inputs and the bias must account for the observed affine value', .00002)
          close(inspection.denseLayers[neuron.layer].activation ? Math.max(0, neuron.preActivation) : neuron.preActivation,
            neuron.value, 'The displayed neuron must be its real post-activation value', .000001)
        }
        close(explanation.error, activations.action_scores[2] * 80, 'Court error must come from the actual Right score')
        close(explanation.target, scenario.frame.paddles[0] + explanation.error,
          'The displayed target must explain the action relative to the real paddle')
        const first = activations.hidden_0
        close(explanation.forecast, 80 + 80 * (.9 * first[0] - .9 * first[1] + first[2] - inspection.denseLayers[0].biases[2]),
          'The raw forecast must retain the observed hinge contributions even when recovery suppresses it')
      }
    } finally {
      await original.release()
      await instrumented.release()
    }
  }
  for (const [scenario, attackAction, counterAction] of [['outgoing', 2, 0], ['long', 0, 2], ['bounce', 2, 0]] as const) {
    assert.equal(explanations.get(`attack:${scenario}`)!.chosen, attackAction)
    assert.equal(explanations.get(`counter:${scenario}`)!.chosen, counterAction)
  }
  const recovery = explanations.get('counter:outgoing')!
  close(recovery.forecast, 120, 'Show what the forecast circuit computed before the gate')
  assert.equal(recovery.incomingGate, 0)
  assert.equal(recovery.activeForecast, null, 'Do not present a disabled forecast as a real center landing')
  assert.equal(recovery.reflected, null)
  assert.equal(recovery.cornerTarget, null, 'A recovery action has no intended return corner')
  close(recovery.target, 80, 'Maximum-speed departure should recover to court center')
  close(explanations.get('attack:outgoing')!.target, 127.4359, 'Attack continues following its forecast on departure')
  close(explanations.get('attack:long')!.forecast, 118.25, 'Attack retains its twelve-tick lead cap')
  close(explanations.get('counter:long')!.forecast, 136.25, 'Counter extends the lead to eighteen ticks')
  close(explanations.get('attack:bounce')!.reflected!, 143.75, 'Attack folds its shorter forecast')
  close(explanations.get('counter:bounce')!.reflected!, 125.75, 'Counter anticipates the return from the wall earlier')
})

test('the intended corner reads the real opponent hinges, while missing or mismatched samples fail explicitly', async () => {
  ort.env.wasm.numThreads = 1
  for (const id of ids) {
    const inspection = await inspectAttackModel(bytes[id], id)
    const session = await ort.InferenceSession.create(inspection.bytes, { executionProviders: ['wasm'] })
    try {
      for (const [opponent, expectedAction, expectedCorner] of [[60, 0, 156], [100, 2, 4]] as const) {
        const input = observation({ ...serve(1), x: 16, y: 80, vx: -13, vy: 0, paddles: [80, opponent] }, 0)
        const result = await session.run({ observation: new ort.Tensor('float32', input, [1, 6]) })
        const activations = Object.fromEntries(Object.entries(result).map(([name, tensor]) =>
          [name, Array.from(tensor.data as Float32Array)]))
        const explanation = getAttackExplanation(id, input, activations, activations.action_scores, inspection)
        assert.equal(explanation.chosen, expectedAction)
        close(explanation.cornerTarget!, expectedCorner, 'Move the aimed landing away from the opponent')
        close(explanation.aimOffset, opponent < 80 ? -190 / 39 : 190 / 39,
          'At center contact the offset isolates the opponent-dependent aiming term')
        assert.throws(() => getAttackExplanation(id, input, { ...activations, linear_0: [] }, activations.action_scores, inspection), /linear_0/)
        assert.throws(() => getAttackExplanation(id, input, activations, [0, NaN, 0], inspection), /finite/)
        assert.throws(() => getAttackExplanation(id, input, activations, [0, 0, 0], inspection), /different action scores/)
        assert.throws(() => getAttackExplanation(id === 'attack' ? 'counter' : 'attack', input, activations,
          activations.action_scores, inspection), /must match/)
      }
    } finally {
      await session.release()
    }
  }
})

test('muting the outgoing detector changes only a teaching copy and its downstream departure behavior', async () => {
  ort.env.wasm.numThreads = 1
  const inspection = await inspectAttackModel(bytes.counter, 'counter')
  const beforeBytes = Uint8Array.from(inspection.bytes)
  const beforeLayers = structuredClone(inspection.denseLayers)
  const muted = createCounterIntervention(inspection)
  assert.equal(muted.intervention, 'outgoing-detector-muted')
  assert.deepEqual(Uint8Array.from(inspection.bytes), beforeBytes, 'The normal inspection must remain byte-for-byte unchanged')
  assert.deepEqual(inspection.denseLayers, beforeLayers, 'The normal inspector must retain its stored weights')
  assert.notEqual(muted.bytes, inspection.bytes)
  assert.deepEqual(muted.denseLayers[0].weights[4], [0, 0, 0, 0, 0, 0])
  assert.equal(muted.denseLayers[0].biases[4], 0)
  for (let layer = 0; layer < inspection.denseLayers.length; layer++) {
    for (let neuron = 0; neuron < inspection.denseLayers[layer].width; neuron++) {
      if (layer === 0 && neuron === 4) continue
      assert.deepEqual(muted.denseLayers[layer].weights[neuron], inspection.denseLayers[layer].weights[neuron])
      assert.equal(muted.denseLayers[layer].biases[neuron], inspection.denseLayers[layer].biases[neuron])
    }
  }
  const beforeGraph = pb.onnx.ModelProto.decode(inspection.bytes).graph!
  const mutedGraph = pb.onnx.ModelProto.decode(muted.bytes).graph!
  assert.deepEqual(mutedGraph.node, beforeGraph.node)
  assert.deepEqual(mutedGraph.output, beforeGraph.output, 'Both runs expose identical tensor names')
  for (let i = 2; i < beforeGraph.initializer!.length; i++)
    assert.deepEqual(mutedGraph.initializer![i], beforeGraph.initializer![i], 'Every downstream weight stays unchanged')
  assert.throws(() => createCounterIntervention(muted), /unmodified Counter/)
  assert.throws(() => createCounterIntervention({ ...inspection, id: 'attack' }), /unmodified Counter/)
  // The normal frozen-weight validator must keep rejecting this altered copy.
  const playableMuted = pb.onnx.ModelProto.decode(muted.bytes)
  playableMuted.graph!.output = playableMuted.graph!.output!.filter(output => output.name === 'action_scores')
  await assert.rejects(inspectAttackModel(pb.onnx.ModelProto.encode(playableMuted).finish(), 'counter'), /frozen.*weights/)

  const normalSession = await ort.InferenceSession.create(inspection.bytes, { executionProviders: ['wasm'] })
  const mutedSession = await ort.InferenceSession.create(muted.bytes, { executionProviders: ['wasm'] })
  try {
    for (const velocity of [-13, -5, 5, 13]) {
      const input = observation({ ...serve(1), x: 184, y: 120, vx: velocity, vy: 0, paddles: [100, 80] }, 0)
      const feed = { observation: new ort.Tensor('float32', input, [1, 6]) }
      const normalResult = await normalSession.run(feed)
      const mutedResult = await mutedSession.run(feed)
      const activations = Object.fromEntries(Object.entries(mutedResult).map(([name, tensor]) =>
        [name, Array.from(tensor.data as Float32Array)]))
      const explanation = getAttackExplanation('counter', input, activations, activations.action_scores, muted)
      assert.equal(activations.hidden_0[4], 0, 'The intervention must force the detector output to zero')
      for (const neuron of explanation.neurons) {
        close(neuron.bias + neuron.contributions.reduce((sum, part) => sum + part.product, 0), neuron.preActivation,
          'The modified inspector must still account for the actual ONNX computation', .00002)
      }
      if (velocity < 0) {
        for (const [name, tensor] of Object.entries(normalResult)) {
          const expected = Array.from(tensor.data as Float32Array)
          // Its affine value becomes zero rather than negative; ReLU already
          // yielded zero, so every post-activation and downstream value agrees.
          if (name === inspection.denseLayers[0].linearName) expected[4] = 0
          assert.deepEqual(activations[name], expected, 'An already inactive detector has no downstream effect on approach')
        }
      } else {
        assert.notDeepEqual(activations.action_scores, Array.from(normalResult.action_scores.data as Float32Array))
        assert.equal(explanation.chosen, 2, 'The muted network follows the forecast to the right')
        close(explanation.target, 127.4359, 'Without the detector, downstream circuits keep the forecast active')
        assert.equal(explanation.incomingGate, 1)
        assert.notEqual(explanation.activeForecast, null)
        const normalActivations = Object.fromEntries(Object.entries(normalResult).map(([name, tensor]) =>
          [name, Array.from(tensor.data as Float32Array)]))
        const normal = getAttackExplanation('counter', input, normalActivations, normalActivations.action_scores, inspection)
        assert.equal(normal.chosen, 0, 'The normal network instead recovers toward center')
        assert.equal(normal.activeForecast, null)
      }
    }
  } finally {
    await normalSession.release()
    await mutedSession.release()
  }
  assert.deepEqual(Uint8Array.from(inspection.bytes), beforeBytes, 'Inference must leave the normal model unchanged too')
})

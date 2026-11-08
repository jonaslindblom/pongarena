import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import pb from 'onnx-proto'
import * as ort from 'onnxruntime-web/wasm'
import { ATTACK_SCENARIOS, getAttackExplanation, inspectAttackModel, type AttackModelId } from '../src/attackExplanation'
import { observation, serve } from '../src/game'

const ids = ['attack75', 'counter97'] as const
const bytes = Object.fromEntries(ids.map(id => [id,
  readFileSync(new URL(`../public/models/${id}.onnx`, import.meta.url))])) as Record<AttackModelId, Uint8Array>
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
    assert.equal(inspection.params, id === 'attack75' ? 75 : 97)
    assert.equal(inspection.macs, id === 'attack75' ? 63 : 81)
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
  await assert.rejects(inspectAttackModel(new Uint8Array([1, 2, 3]), 'attack75'), /plain dense network/)
  await assert.rejects(inspectAttackModel(bytes.counter97, 'attack75'), /architecture/)
  await assert.rejects(inspectAttackModel(bytes.attack75, 'counter97'), /architecture/)
  const altered = pb.onnx.ModelProto.decode(Uint8Array.from(bytes.attack75))
  const raw = altered.graph!.initializer![0].rawData!
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength)
  view.setFloat32(0, view.getFloat32(0, true) + .25, true)
  await assert.rejects(inspectAttackModel(pb.onnx.ModelProto.encode(altered).finish(), 'attack75'), /frozen.*weights/,
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
      for (const scenario of ATTACK_SCENARIOS) {
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
    assert.equal(explanations.get(`attack75:${scenario}`)!.chosen, attackAction)
    assert.equal(explanations.get(`counter97:${scenario}`)!.chosen, counterAction)
  }
  const recovery = explanations.get('counter97:outgoing')!
  close(recovery.forecast, 120, 'Show what the forecast circuit computed before the gate')
  assert.equal(recovery.incomingGate, 0)
  assert.equal(recovery.activeForecast, null, 'Do not present a disabled forecast as a real center landing')
  assert.equal(recovery.reflected, null)
  assert.equal(recovery.cornerTarget, null, 'A recovery action has no intended return corner')
  close(recovery.target, 80, 'Maximum-speed departure should recover to court center')
  close(explanations.get('attack75:outgoing')!.target, 127.4359, 'Attack75 continues following its forecast on departure')
  close(explanations.get('attack75:long')!.forecast, 118.25, 'Attack75 retains its twelve-tick lead cap')
  close(explanations.get('counter97:long')!.forecast, 136.25, 'Counter97 extends the lead to eighteen ticks')
  close(explanations.get('attack75:bounce')!.reflected!, 143.75, 'Attack75 folds its shorter forecast')
  close(explanations.get('counter97:bounce')!.reflected!, 125.75, 'Counter97 anticipates the return from the wall earlier')
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
        assert.throws(() => getAttackExplanation(id === 'attack75' ? 'counter97' : 'attack75', input, activations,
          activations.action_scores, inspection), /must match/)
      }
    } finally {
      await session.release()
    }
  }
})

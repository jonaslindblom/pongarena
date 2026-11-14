import { readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { basename } from 'node:path'
import * as ort from 'onnxruntime-web/wasm'
import { choice, observation, serve } from '../../../src/game.ts'

const files = process.argv.slice(2)
if (!files.length) throw new Error('Pass one or more ONNX files to probe.')
ort.env.wasm.numThreads = 1

// Synthetic paired observations isolate each input while holding the rest fixed.
const frames = []
for (const x of [30, 90, 165, 240, 315]) {
  for (const y of [24, 52, 80, 108, 136]) {
    for (const vy of [-3, 0, 3]) {
      for (const offset of [-10, 0, 10]) {
        const own = Math.max(15, Math.min(145, y + offset))
        frames.push({ ...serve(0), x, y, vx: -8, vy, paddles: [own, 80] })
      }
    }
  }
}

const results = []
for (const file of files) {
  const bytes = await readFile(file)
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
  let velocityFlips = 0
  let opponentFlips = 0
  try {
    async function action(frame) {
      const output = await session.run({ observation: new ort.Tensor('float32', observation(frame, 0), [1, 6]) })
      return choice(Array.from(output.action_scores.data))
    }
    for (const frame of frames) {
      if (await action(frame) !== await action({ ...frame, vx: 8 })) velocityFlips++
      const own = frame.paddles[0]
      if (await action({ ...frame, paddles: [own, 25] }) !==
          await action({ ...frame, paddles: [own, 135] })) opponentFlips++
    }
  } finally {
    await session.release()
  }
  results.push({ id: basename(file, '.onnx'), sha256: createHash('sha256').update(bytes).digest('hex'),
    pairsPerInput: frames.length, velocityFlips, opponentFlips })
}
console.log(JSON.stringify(results, null, 2))

# Rules alone versus a tactical brief

This follow-up tests whether a tactical plan helps GPT-6 Astra write stronger Pong weights in one response. Two independent calls used the [original rules prompt](../prompt.txt), and two appended the [tactical brief](tactical-brief.txt). All four used the same 163-parameter ReLU layout (6 → 12 → 4 → 3 → 3), low reasoning effort, a 6,000 output token cap, and the same literal-weight JSON schema. Each call ran without tools, opponent weights, match feedback, or repairs.

The [64 paired serve seeds](seeds.json) were fixed before the calls and have no overlap with the first experiment, the reference league, or the saved Attack and Counter evaluations. Each seed is played twice with sides swapped. The paired score is (wins + half of draws) / 64. Each candidate faces all seven bundled opponents. The prior two 163-parameter Astra controllers and Quill were replayed on the fresh seeds for context.

The [input probes](probe.mjs) hold other inputs fixed in 225 synthetic observations. They measure how often the chosen move changes when x-velocity flips from −8 to +8 or when opponent paddle position moves from 25 to 135 on an incoming ball.

## Results

All four responses compiled into playable ONNX models. Each cell below is paired league points out of 64.

The catalog calls Tactical A **OneShot**. Its ONNX file is the original response compiled to float32.

| Prompt | Follow | Small | Predict | Attack | Counter | Quill | Large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rules A | 64 | 17 | 63 | 44.5 | 0.5 | 0 | 0 |
| Rules B | 62.5 | 35 | 0.5 | 0 | 0 | 0 | 0 |
| Tactical A | 64 | 2 | 63 | 60 | 3.5 | 1.5 | 0 |
| Tactical B | 63.5 | 15.5 | 64 | 64 | 0 | 0 | 0 |

Tactical A won one paired set and drew one against Quill. It also won 109 of 640 individual rallies against Large, compared with 16 and 7 for the two rules-only attempts. All four lost every paired set to Large. On the same serves, Quill won 224 rallies and earned 8 of 64 paired points against Large.

The tactical prompt produced two controllers that were stronger against Attack and weaker against Small than either rules-only controller. Results describe these four outputs. Two attempts per prompt provide limited evidence about how consistently it improves the next sample.

| Model | Action changes when direction flips | Action changes when opponent moves |
| --- | ---: | ---: |
| Rules A | 0/225 | 0/225 |
| Rules B | 0/225 | 0/225 |
| Tactical A | 101/225 | 39/225 |
| Tactical B | 150/225 | 53/225 |
| Quill | 101/225 | 53/225 |

The [raw probes](probes.json) show that both tactical controllers used the two inputs in ways that changed actual decisions on this grid. The probe does not establish whether those changes caused their match results.

The four calls used 3,064 input and 8,482 output tokens, about $0.45 at [standard Astra rates](https://developers.openai.com/api/docs/models/gpt-6-astra). The experiment logged 3,584 candidate games and 1,408 control games.

## Replay

Run from the repository root after installing dependencies.

```sh
base=experiments/one-prompt-weights
dir="$base/follow-up"
node --import tsx "$base/evaluate.mjs" compile 6,12,4,3,3 Relu \
  "$dir/attempts/tactical-163-a.json" /tmp/tactical-163-a.onnx
node --import tsx "$base/evaluate.mjs" evaluate "$dir/seeds.json" \
  /tmp/tactical-163-a.onnx large quill counter attack predict small follow
node --import tsx "$dir/probe.mjs" /tmp/tactical-163-a.onnx public/models/quill.onnx
```

The attempt records, ONNX files, and raw paired game results are stored in this directory.

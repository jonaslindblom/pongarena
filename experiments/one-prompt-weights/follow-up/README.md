# Rules alone versus a tactical brief

**OneShot's prompt** combines the [game rules](../prompt.txt), with `{{WIDTHS}}` replaced by `6 → 12 → 4 → 3 → 3`, and the [tactical brief](tactical-brief.txt) appended. It asks GPT-6 Astra for literal weights using that 163-parameter ReLU layout, without tools, opponent weights or match feedback.

The four saved models compare two responses to the rules alone with two responses to the combined prompt. All use low reasoning effort and compile directly from the returned numbers, without repairs.

Each model faces the seven bundled opponents under Pong v3 on [64 saved serve seeds](seeds.json). Every seed is played from both sides; a paired win earns one point and a draw earns half.

## Results

The catalog calls Tactical A **OneShot**. Its [canonical ONNX file](../../../public/models/oneshot.onnx) is the saved response compiled to float32. Each cell below is paired league points out of 64.

| Prompt | Follow | Small | Predict | Attack | Counter | Quill | Large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rules A | 64 | 36 | 63 | 44.5 | 0.5 | 0 | 0 |
| Rules B | 62.5 | 10.5 | 0.5 | 0 | 0 | 0 | 0 |
| Tactical A | 64 | 26.5 | 63 | 60 | 3.5 | 1.5 | 0 |
| Tactical B | 63.5 | 14 | 64 | 64 | 0 | 0 | 0 |

OneShot won 28 of 640 individual rallies against Large, compared with 20 and 25 for the two rules-only attempts and 68 for Tactical B. All four lost every paired set to Large.

Both tactical models scored higher against Attack than either rules-only model. Against Small, their scores fell between the two rules-only models. With two saved models per prompt, these results do not establish how consistently the brief improves a new response.

The [input probes](probe.mjs) test 225 positions, changing only the ball's travel direction or the opponent's position to see whether either affects the chosen move:

| Model | Action changes when direction flips | Action changes when opponent moves |
| --- | ---: | ---: |
| Rules A | 0/225 | 0/225 |
| Rules B | 0/225 | 0/225 |
| Tactical A | 101/225 | 39/225 |
| Tactical B | 150/225 | 53/225 |
| Quill | 101/225 | 53/225 |

The [raw probes](probes.json) show that both tactical controllers used the two inputs in ways that changed actual decisions on this grid. The probe does not establish whether those changes caused their match results.

## Replay

The saved responses rebuild identical weights. This checkout's evaluator,
bundled opponents and saved seeds reproduce the results. A new AI response to
the same prompt may differ.

Use the [repository's Node setup](../../../README.md#run-locally), then run from
its root. These commands rebuild all four responses, compare every recorded
game and replay the input probes. No API call is needed.

```sh
npm ci
replay_dir=$(mktemp -d)
base=experiments/one-prompt-weights
dir="$base/follow-up"
for attempt in rules-163-a rules-163-b tactical-163-a tactical-163-b; do
  node --import tsx "$base/evaluate.mjs" compile 6,12,4,3,3 Relu \
    "$dir/attempts/$attempt.json" "$replay_dir/$attempt.onnx"
  node --import tsx "$base/evaluate.mjs" evaluate "$dir/seeds.json" \
    "$replay_dir/$attempt.onnx" follow small predict attack counter quill large \
    > "$replay_dir/$attempt.json"
  cmp "$replay_dir/$attempt.json" "$dir/results/$attempt.json"
done
cmp "$replay_dir/tactical-163-a.onnx" public/models/oneshot.onnx
node --import tsx "$dir/probe.mjs" "$replay_dir/rules-163-a.onnx" \
  "$replay_dir/rules-163-b.onnx" "$replay_dir/tactical-163-a.onnx" \
  "$replay_dir/tactical-163-b.onnx" public/models/quill.onnx public/models/counter.onnx \
  > "$replay_dir/probes.json"
cmp "$replay_dir/probes.json" "$dir/probes.json"
```

Saved responses and results are in `attempts/` and `results/`. The lesson loads
`models/rules-163-a.onnx` and the canonical `public/models/oneshot.onnx`; rebuild
other candidates from their saved responses.

To request new OneShot weights, set `OPENAI_API_KEY` and run from the repository
root. This makes a paid API request; the response may differ from the saved model.

```sh
request_dir=$(mktemp -d)
base=experiments/one-prompt-weights
cp "$base/generate.mjs" "$base/prompt.txt" "$request_dir/"
mkdir "$request_dir/follow-up"
cp "$base/follow-up/tactical-brief.txt" "$request_dir/follow-up/"
node "$request_dir/generate.mjs" tactical-163-a
```

The response is saved to `$request_dir/follow-up/attempts/tactical-163-a.json`.

# One prompt to Pong weights

The [prompt](prompt.txt) gives GPT-6 Astra the Pong rules and asks for literal neural-network weights. It specifies the inputs, layer layout and JSON output format. The [tactical-brief comparison](follow-up/README.md) uses an extended prompt and includes **OneShot**, the model in the catalog.

## Method

The four saved models use two ReLU layouts: Counter's 97-parameter layout (6 → 6 → 4 → 3 → 3) and Quill's 163-parameter layout (6 → 12 → 4 → 3 → 3), with two models per layout. All use [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra) at low reasoning effort.

The prompt supplies no model examples, opponent information or match feedback, and the model has no tools. Saved responses compile directly into float32 ONNX weights, without repairs.

All four models face the same seven bundled opponents under Pong v3 on [64 saved serve seeds](seeds.json). Each seed gives two five-rally games with sides swapped. A paired win earns one point and a draw earns half; the table shows points as a percentage of 64.

## Results

| Attempt | Parameters | Follow | Small | Predict | Attack | Counter | Quill | Large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Low A | 97 | 97.7% | 15.6% | 74.2% | 0.8% | 0.8% | 0% | 0% |
| Low B | 97 | 100% | 30.5% | 83.6% | 0% | 0% | 0% | 0% |
| Low A | 163 | 100% | 38.3% | 100% | 85.9% | 7.8% | 0% | 0% |
| Low B | 163 | 82.0% | 10.9% | 72.7% | 0% | 1.6% | 0% | 0% |

All four beat Follow and Predict overall, but lost to Small. The two 163-parameter outputs scored 85.9% and 0% against Attack, showing substantial variation at the same size and prompt. None earned a paired point against Quill or Large.

All four ignore the ball's along-court velocity and the opponent's paddle position: the corresponding first-layer weights are zero.

These results describe the four saved models; two models per size do not establish how reliably a prompt will produce a strong controller.

## Reproduce

The saved responses, model files, evaluator and seeds are all in this checkout.
Compiling a response reproduces its ONNX bytes; replaying it against the bundled
opponents reproduces the recorded results. A new AI response can differ.

Use the [repository's Node setup](../../README.md#run-locally), then run from its
root. No API call is needed. A fresh temporary directory keeps the compiler from
overwriting an existing file.

```sh
npm ci
replay_dir=$(mktemp -d)
dir=experiments/one-prompt-weights
node --import tsx "$dir/evaluate.mjs" compile 6,6,4,3,3 Relu \
  "$dir/attempts/astra-97-low.json" "$replay_dir/astra-97-low.onnx"
node --import tsx "$dir/evaluate.mjs" evaluate \
  "$dir/seeds.json" "$replay_dir/astra-97-low.onnx" \
  follow small predict attack counter quill large > "$replay_dir/astra-97-low.json"
cmp "$replay_dir/astra-97-low.json" "$dir/results/astra-97-low.json"
```

Saved responses are in `attempts/`; raw game records are in `results/`.
Repeat with the other response filenames to check all four models. Use
`6,12,4,3,3` for the 163-parameter responses. `cmp` checks the entire recorded
result, including model hashes, individual games and score intervals.

To collect a new output, copy the prompt and generator to a fresh directory, set `OPENAI_API_KEY`, and run `node generate.mjs astra-97-low` there. The generator lists the available layouts.

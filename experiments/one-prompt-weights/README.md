# One prompt to Pong weights

The [prompt](prompt.txt) gives GPT-6 Astra the Pong rules and asks for literal neural-network weights. It specifies the inputs, layer layout and JSON output format. The [tactical-brief comparison](follow-up/README.md) uses an extended prompt and includes **OneShot**, the model in the catalog.

## Method

The four saved models use two ReLU layouts: Counter's 97-parameter layout (6 → 6 → 4 → 3 → 3) and Quill's 163-parameter layout (6 → 12 → 4 → 3 → 3), with two models per layout. All use [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra) at low reasoning effort.

The prompt supplies no model examples, opponent information or match feedback, and the model has no tools. Saved responses compile directly into float32 ONNX weights, without repairs.

All four models faced the same opponents under Pong v3 on [64 fresh serve seeds](seeds.json), fixed before any output was available. Each seed gives two five-rally games with sides swapped. A paired win earns one point and a draw earns half; the table shows points as a percentage of 64.

## Results

| Attempt | Parameters | Follow | Small | Predict | Attack | Counter | Quill | Large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Low A | 97 | 97.7% | 71.9% | 74.2% | 0.8% | 0.8% | 0% | 0% |
| Low B | 97 | 100% | 88.3% | 83.6% | 0% | 0% | 0% | 0% |
| Low A | 163 | 100% | 95.3% | 100% | 85.9% | 7.8% | 0% | 0% |
| Low B | 163 | 82.0% | 95.3% | 72.7% | 0% | 1.6% | 0% | 0% |

All four beat Follow, Small and Predict overall. The two 163-parameter outputs scored 85.9% and 0% against Attack, showing substantial variation at the same size and prompt. None earned a paired point against Quill or Large.

All four ignore the ball's along-court velocity and the opponent's paddle position: the corresponding first-layer weights are zero.

These results describe the four saved models; two models per size do not establish how reliably a prompt will produce a strong controller.

## Reproduce

Compiling a saved response reproduces its ONNX bytes. Replaying those weights
with the pinned code and saved seeds reproduces the recorded results. Asking
the AI to answer the prompt again can produce different weights.

Use the [experiment revision](https://github.com/jonaslindblom/pongarena/commit/47800e7e921ffd4392bcadfd4ace30422e8da9f4).
It contains the saved responses, scripts and evaluator used for these results.
A separate checkout keeps future game changes out of the replay.
From the repository root:

```sh
git fetch origin 47800e7e921ffd4392bcadfd4ace30422e8da9f4
git worktree add --detach ../pongarena-prompt-replay 47800e7e921ffd4392bcadfd4ace30422e8da9f4
cd ../pongarena-prompt-replay
npm ci
```

Then compile a saved response and replay its recorded opponents. No API call is
needed. Choose an unused output filename; compilation will not
overwrite an existing file.

```sh
dir=experiments/one-prompt-weights
node --import tsx "$dir/evaluate.mjs" compile 6,6,4,3,3 Relu \
  "$dir/attempts/astra-97-low.json" /tmp/astra-97-low.onnx
node --import tsx "$dir/evaluate.mjs" evaluate \
  "$dir/seeds.json" /tmp/astra-97-low.onnx large quill counter follow small predict attack
```

Saved responses are in `attempts/`; raw game records are in `results/`.
The commands above rebuild candidate ONNX files from those responses.

To collect a new output, copy the prompt and generator to a fresh directory, set `OPENAI_API_KEY`, and run `node generate.mjs astra-97-low` there. The generator lists the available layouts.

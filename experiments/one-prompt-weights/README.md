# One prompt to Pong weights

Can GPT-6 Astra design a playable neural controller from the Pong rules and a fixed layer layout in one response?

## Protocol

The requests used [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra) with two ReLU layouts: Counter's 97-parameter layout (6 → 6 → 4 → 3 → 3) and Quill's 163-parameter layout (6 → 12 → 4 → 3 → 3). High effort used a 12,000 output token cap. Low and medium effort used 6,000.

Five synchronous high-effort calls (three at 97 parameters and two at 163) dropped their connection before delivering a response. Two separately labeled low-effort calls returned weights. Three additional calls used [Responses API background mode](https://developers.openai.com/api/docs/guides/background): one high-effort call at each layout and one medium-effort call at 97 parameters. The transport failures are recorded in `attempts/` and do not count as played candidates.

After the first low-effort runs, we repeated each size once with the same prompt and expanded the opponent panel to Small, Predict, and Attack. These follow-up comparisons use the same fixed seeds.

The [prompt template](prompt.txt) supplies game rules, observation and action conventions, the architecture, and the format for literal numbers. Each request starts fresh and receives no model examples, opponent information, tools, or evaluation feedback. The compiler checks the returned dimensions and serializes the literal coefficients as float32 ONNX weights.

The evaluator runs the arena's Pong v3 simulation and single-threaded ONNX Runtime Web WASM inference from commit `f549eb88d47be40e0ede19607e2fe6f077255146`. The [64 shared serve seeds](seeds.json) were fixed before any candidate output was available and have no overlap with the development league or the saved Attack and Counter evaluations. For each opponent and seed, it plays two five-rally games with sides swapped. A paired win earns one league point, and a paired draw earns half a point. The tables report league points divided by 64. Intervals in the raw results resample whole paired seed sets.

## Results

| Attempt | Parameters | Follow | Small | Predict | Attack | Counter | Quill | Large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Low A | 97 | 97.7% | 71.9% | 74.2% | 0.8% | 0.8% | 0% | 0% |
| Low B | 97 | 100% | 88.3% | 83.6% | 0% | 0% | 0% | 0% |
| Low A | 163 | 100% | 95.3% | 100% | 85.9% | 7.8% | 0% | 0% |
| Low B | 163 | 82.0% | 95.3% | 72.7% | 0% | 1.6% | 0% | 0% |

All four low-effort outputs compiled and beat Follow, Small, and Predict overall. The best 163-parameter attempt went 50 wins, 10 draws, and 4 losses in paired sets against Attack. Its approximate paired bootstrap interval was 78.1% to 92.2%. The other 163-parameter attempt lost all 64 sets to Attack, showing substantial variation at the same size and prompt.

Both handwritten controls scored above zero against Large on the same seeds: Counter earned 0.5 of 64 points, and Quill earned 12. The strongest low-effort candidate won 44 of its 640 individual rallies against Large and earned zero paired points. Every low-effort candidate has all-zero first-layer columns for ball x-velocity and opponent paddle position. For a fixed observation otherwise, none can change its decision in response to those inputs.

The attempt records include API status, returned weights, elapsed time, and token usage where available. The result files include every paired game, and the ONNX files can be imported into the arena. This is a small pilot with two completed low-effort samples per architecture.

All three background medium and high attempts exhausted their output token budgets on reasoning and returned no weights. Together with the five connection failures, that leaves four playable controllers from twelve requests. The seven requests with usage records totaled 4,906 input and 37,932 output tokens, about $1.95 at [standard Astra rates](https://developers.openai.com/api/docs/models/gpt-6-astra). Billing for the five dropped connections is unknown.

The experiment logged 3,584 games for the generated controllers and 768 baseline games.

## Reproduce

Run these commands from the repository root after installing the project dependencies. The saved attempt records are sufficient to replay the results without an API call.

```sh
dir=experiments/one-prompt-weights
node --import tsx "$dir/evaluate.mjs" compile 6,6,4,3,3 Relu \
  "$dir/attempts/astra-97-low.json" /tmp/astra-97-low.onnx
node --import tsx "$dir/evaluate.mjs" evaluate \
  "$dir/seeds.json" /tmp/astra-97-low.onnx large quill counter follow
```

To collect a new output, copy the prompt and generator to a fresh directory, set `OPENAI_API_KEY`, and run `node generate.mjs astra-97-low` there. Use the IDs in the generator to select other layouts and efforts. The generator reserves each attempt file before making its request and polls its response without further model input.

# PongArena league

All 7 bundled models, 16 paired serve sets per matchup, 672 games.

| Rank | Model | Kind | Size | Points | 95% score interval | W / D / L | Parameters |
| ---: | --- | --- | --- | ---: | --- | --- | ---: |
| 1 | Large | Trained weights | Large | 93 / 96 | 94.3%–99.0% | 91 / 4 / 1 | 50819 |
| 2 | Quill | Handwritten weights | Small | 81 / 96 | 82.3%–87.0% | 80 / 2 / 14 | 163 |
| 3 | Counter | Handwritten weights | Small | 61.5 / 96 | 60.9%–66.7% | 61 / 1 / 34 | 97 |
| 4 | Attack | Handwritten weights | Small | 47 / 96 | 47.4%–50.0% | 46 / 2 / 48 | 75 |
| 5 | Predict | Handwritten weights | Small | 30 / 96 | 29.2%–32.8% | 28 / 4 / 64 | 75 |
| 6 | Small | Trained weights | Small | 20 / 96 | 17.7%–24.0% | 14 / 12 / 70 | 235 |
| 7 | Follow | Handwritten weights | Small | 3.5 / 96 | 1.0%–6.8% | 0 / 7 / 89 | 13 |

Each contest combines two five-rally games with swapped sides on the same seed. A contest win earns one point; a tie earns half. Equal points share rank.

Approximate 95% score intervals are percentages of available league points. A reproducible percentile bootstrap draws 5,000 resamples of whole shared serve sets, retaining both sides and every matchup together. They describe seed-set uncertainty, not training variation or certainty about ranks. With fewer than two serve sets or no observed score variation, an interval is not estimated.

Rules: `pong-v3` · Contract: `dense-mlp` · Protocol: `paired-round-robin` · Seeds: `public-league`.

Benchmark: `71da50bc55dfeb1ae1ce4baa9b07846b23de53e5996a3927d391d1fdc0474328`.

Generated from the repository’s model bytes by `npm run league`, using ONNX Runtime Web CPU/WASM with one thread. CI reruns the league and checks both committed files. Review the passing workflow on the relevant commit before treating a proposed PR table as verified.

The public seeds are a repeatable development benchmark, not a private training holdout or a universal strength rating. Compare scores only within the same benchmark.

[Full results, model hashes and serve seeds](league.json)

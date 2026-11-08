# PongArena league

All 9 bundled models, 16 paired serve sets per matchup, 1152 games.

| Rank | Model | Kind | Size | Points | 95% score interval | W / D / L | Parameters |
| ---: | --- | --- | --- | ---: | --- | --- | ---: |
| 1 | Large | Trained weights | Large | 112.5 / 128 | 82.8%–92.2% | 105 / 15 / 8 | 50819 |
| 2 | Quill Geometry | Handwritten weights | Large | 110.5 / 128 | 81.6%–91.0% | 105 / 11 / 12 | 41430 |
| 3 | Quill Speed | Handwritten weights | Small | 97 / 128 | 72.7%–78.9% | 92 / 10 / 26 | 229 |
| 4 | Quill | Handwritten weights | Small | 91 / 128 | 66.8%–75.8% | 88 / 6 / 34 | 163 |
| 5 | Counter97 | Handwritten weights | Small | 63 / 128 | 46.1%–52.7% | 62 / 2 / 64 | 97 |
| 6 | Attack75 | Handwritten weights | Small | 47.5 / 128 | 35.9%–38.3% | 46 / 3 / 79 | 75 |
| 7 | Predict75 | Handwritten weights | Small | 30 / 128 | 21.9%–24.6% | 28 / 4 / 96 | 75 |
| 8 | Small | Trained weights | Small | 21 / 128 | 13.7%–19.5% | 14 / 14 / 100 | 235 |
| 9 | Follow | Handwritten weights | Small | 3.5 / 128 | 0.8%–5.1% | 0 / 7 / 121 | 13 |

Each contest combines two five-rally games with swapped sides on the same seed. A contest win earns one point; a tie earns half. Equal points share rank.

Approximate 95% score intervals are percentages of available league points. A reproducible percentile bootstrap draws 5,000 resamples of whole shared serve sets, retaining both sides and every matchup together. They describe seed-set uncertainty, not training variation or certainty about ranks. With fewer than two serve sets or no observed score variation, an interval is not estimated.

Rules: `pong-v3` · Contract: `dense-mlp` · Protocol: `paired-round-robin` · Seeds: `public-league`.

Benchmark: `2d1ac77dbe0405ba01f613c05095e515ac3deeef6fdaca5ae13be9d18f0a6039`.

Generated from the repository’s model bytes by `npm run league`, using ONNX Runtime Web CPU/WASM with one thread. CI reruns the league and checks both committed files. Review the passing workflow on the relevant commit before treating a proposed PR table as verified.

The public seeds are a repeatable development benchmark, not a private training holdout or a universal strength rating. Compare scores only within the same benchmark.

[Full results, model hashes and serve seeds](league.json)

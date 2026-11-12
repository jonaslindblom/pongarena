# PongArena league

All 8 bundled models, 16 paired serve sets per matchup, 896 games.

| Rank | Model | Kind | Size | Points | 95% score interval | W / D / L | Parameters |
| ---: | --- | --- | --- | ---: | --- | --- | ---: |
| 1 | Large | Trained weights | Large | 109 / 112 | 95.1%–99.1% | 107 / 4 / 1 | 50819 |
| 2 | Quill | Handwritten weights | Small | 97 / 112 | 84.8%–88.8% | 96 / 2 / 14 | 163 |
| 3 | Counter | Handwritten weights | Small | 77 / 112 | 65.2%–71.4% | 76 / 2 / 34 | 97 |
| 4 | OneShot | Prompted weights | Small | 48.5 / 112 | 42.9%–44.2% | 48 / 1 / 63 | 163 |
| 5 | Attack | Handwritten weights | Small | 47 / 112 | 40.6%–42.9% | 46 / 2 / 64 | 75 |
| 6 | Small | Trained weights | Small | 36 / 112 | 29.5%–34.8% | 30 / 12 / 70 | 235 |
| 7 | Predict | Handwritten weights | Small | 30 / 112 | 25.0%–28.1% | 28 / 4 / 80 | 75 |
| 8 | Follow | Handwritten weights | Small | 3.5 / 112 | 0.9%–5.8% | 0 / 7 / 105 | 13 |

Each contest combines two five-rally games with swapped sides on the same seed. A contest win earns one point; a tie earns half. Equal points share rank.

Approximate 95% score intervals are percentages of available league points. A reproducible percentile bootstrap draws 5,000 resamples of whole shared serve sets, retaining both sides and every matchup together. They describe seed-set uncertainty, not training variation or certainty about ranks. With fewer than two serve sets or no observed score variation, an interval is not estimated.

Rules: `pong-v3` · Contract: `dense-mlp` · Protocol: `paired-round-robin` · Seeds: `public-league`.

Benchmark: `f6052e6eb3308e8f8d7ea734cd1019111da7d392f03f2728c1e7c3d8b0ff1b9b`.

Generated from the repository’s model bytes by `npm run league`, using ONNX Runtime Web CPU/WASM with one thread. CI reruns the league and checks both committed files. Review the passing workflow on the relevant commit before treating a proposed PR table as verified.

The public seeds are a repeatable development benchmark, not a private training holdout or a universal strength rating. Compare scores only within the same benchmark.

[Full results, model hashes and serve seeds](league.json)

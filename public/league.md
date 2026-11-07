# PongArena league

All 5 bundled models, 16 paired serve sets per matchup, 320 games.

| Rank | Model | Kind | Size | Points | 95% score interval | W / D / L | Parameters |
| ---: | --- | --- | --- | ---: | --- | --- | ---: |
| 1 | Large | Trained weights | Large | 61 / 64 | 91.4%–98.4% | 59 / 4 / 1 | 50819 |
| 2 | Quill | Handwritten weights | Small | 49 / 64 | 73.4%–80.5% | 48 / 2 / 14 | 163 |
| 3 | Predict75 | Handwritten weights | Small | 30 / 64 | 43.8%–49.2% | 28 / 4 / 32 | 75 |
| 4 | Small | Trained weights | Small | 16.5 / 64 | 21.1%–30.5% | 12 / 9 / 43 | 235 |
| 5 | Follow | Handwritten weights | Small | 3.5 / 64 | 1.6%–10.2% | 0 / 7 / 57 | 13 |

Each contest combines two five-rally games with swapped sides on the same seed. A contest win earns one point; a tie earns half. Equal points share rank.

Approximate 95% score intervals are percentages of available league points. A reproducible percentile bootstrap draws 5,000 resamples of whole shared serve sets, retaining both sides and every matchup together. They describe seed-set uncertainty, not training variation or certainty about ranks. With fewer than two serve sets or no observed score variation, an interval is not estimated.

Rules: `pong-v3` · Contract: `dense-mlp` · Protocol: `paired-round-robin` · Seeds: `public-league`.

Benchmark: `0fdd7e9351491398c2d9c80c400201b9fc679937ee14059647211cb5ea80d0e8`.

Generated from the repository’s model bytes by `npm run league`, using ONNX Runtime Web CPU/WASM with one thread. CI reruns the league and checks both committed files. Review the passing workflow on the relevant commit before treating a proposed PR table as verified.

The public seeds are a repeatable development benchmark, not a private training holdout or a universal strength rating. Compare scores only within the same benchmark.

[Full results, model hashes and serve seeds](league.json)

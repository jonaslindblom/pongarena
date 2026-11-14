# PongArena league

All 8 bundled models, 16 paired serve sets per matchup, 896 games.

| Rank | Model | Kind | Size | Points | 95% score interval | W / D / L | Parameters |
| ---: | --- | --- | --- | ---: | --- | --- | ---: |
| 1 | Large | Trained weights | Large | 110.5 / 112 | 97.3%–100.0% | 109 / 3 / 0 | 50819 |
| 2 | Quill | Handwritten weights | Small | 96.5 / 112 | 84.4%–87.9% | 94 / 5 / 13 | 163 |
| 3 | Counter | Handwritten weights | Small | 78.5 / 112 | 67.9%–71.4% | 77 / 3 / 32 | 97 |
| 4 | OneShot | Prompted weights | Small | 56 / 112 | 46.9%–53.6% | 53 / 6 / 53 | 163 |
| 5 | Small | Trained weights | Small | 53 / 112 | 42.0%–52.7% | 47 / 12 / 53 | 235 |
| 6 | Attack | Handwritten weights | Small | 32.5 / 112 | 28.6%–29.9% | 32 / 1 / 79 | 75 |
| 7 | Predict | Handwritten weights | Small | 19.5 / 112 | 14.3%–21.0% | 17 / 5 / 90 | 75 |
| 8 | Follow | Handwritten weights | Small | 1.5 / 112 | 0.0%–2.7% | 0 / 3 / 109 | 13 |

Each contest combines two five-rally games with swapped sides on the same seed. A contest win earns one point; a tie earns half. Equal points share rank.

Approximate 95% score intervals are percentages of available league points. A reproducible percentile bootstrap draws 5,000 resamples of whole shared serve sets, retaining both sides and every matchup together. They describe seed-set uncertainty, not training variation or certainty about ranks. With fewer than two serve sets or no observed score variation, an interval is not estimated.

Rules: `pong-v3` · Contract: `dense-mlp` · Protocol: `paired-round-robin` · Seeds: `public-league`.

Benchmark: `a2c16c5686f5d99aebee501d0c22d050164a2325f39af72575350ccac5585ac1`.

Generated from the repository’s model bytes by `npm run league`, using ONNX Runtime Web CPU/WASM with one thread. CI reruns the league and checks both committed files. Review the passing workflow on the relevant commit before treating a proposed PR table as verified.

The public seeds are a repeatable development benchmark, not a private training holdout or a universal strength rating. Compare scores only within the same benchmark.

[Full results, model hashes and serve seeds](league.json)

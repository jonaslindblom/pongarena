# PongArena league

All 3 bundled models, 16 paired serve sets per matchup, 96 games.

| Rank | Model | Kind | Size | Points | W / D / L | Parameters |
| ---: | --- | --- | --- | ---: | --- | ---: |
| 1 | Quill | Handwritten weights | Small | 27 / 32 | 25 / 4 / 3 | 163 |
| 2 | Large | Trained weights | Large | 15.5 / 32 | 13 / 5 / 14 | 34307 |
| 3 | Small | Trained weights | Small | 5.5 / 32 | 4 / 3 / 25 | 235 |

Each contest combines two five-rally games with swapped sides on the same seed. A contest win earns one point; a tie earns half. Equal points share rank.

Rules: `pong` · Contract: `dense-mlp` · Protocol: `paired-round-robin` · Seeds: `public-league`.

Benchmark: `4900a68431d26a1c88c96da5e91a7a31127cf6c971c684a793c4706f63eb893d`.

Generated from the repository’s model bytes by `npm run league`, using ONNX Runtime Web CPU/WASM with one thread. CI reruns the league and checks both committed files. Review the passing workflow on the relevant commit before treating a proposed PR table as verified.

The public seeds are a repeatable development benchmark, not a private training holdout or a universal strength rating. Compare scores only within the same benchmark.

[Full results, model hashes and serve seeds](league.json)

# PongArena league

All 4 bundled models, 16 paired serve sets per matchup, 192 games.

| Rank | Model | Kind | Size | Points | W / D / L | Parameters |
| ---: | --- | --- | --- | ---: | --- | ---: |
| 1 | Quill | Handwritten weights | Small | 43 / 48 | 41 / 4 / 3 | 163 |
| 2 | Large | Trained weights | Large | 31.5 / 48 | 29 / 5 / 14 | 34307 |
| 3 | Small | Trained weights | Small | 20.5 / 48 | 19 / 3 / 26 | 235 |
| 4 | Follow | Handwritten weights | Small | 1 / 48 | 1 / 0 / 47 | 13 |

Each contest combines two five-rally games with swapped sides on the same seed. A contest win earns one point; a tie earns half. Equal points share rank.

Rules: `pong` · Contract: `dense-mlp` · Protocol: `paired-round-robin` · Seeds: `public-league`.

Benchmark: `579eeaf95c19da5f9629cad67ad57bd9570713f844de11b4647cf0dc61b1e2bf`.

Generated from the repository’s model bytes by `npm run league`, using ONNX Runtime Web CPU/WASM with one thread. CI reruns the league and checks both committed files. Review the passing workflow on the relevant commit before treating a proposed PR table as verified.

The public seeds are a repeatable development benchmark, not a private training holdout or a universal strength rating. Compare scores only within the same benchmark.

[Full results, model hashes and serve seeds](league.json)

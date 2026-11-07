# PongArena league

All 5 bundled models, 16 paired serve sets per matchup, 320 games.

| Rank | Model | Kind | Size | Points | W / D / L | Parameters |
| ---: | --- | --- | --- | ---: | --- | ---: |
| 1 | Large | Trained weights | Large | 59.5 / 64 | 57 / 5 / 2 | 50819 |
| 2 | Quill | Handwritten weights | Small | 52 / 64 | 50 / 4 / 10 | 163 |
| 3 | Predict75 | Handwritten weights | Small | 27 / 64 | 25 / 4 / 35 | 75 |
| 4 | Small | Trained weights | Small | 20 / 64 | 17 / 6 / 41 | 235 |
| 5 | Follow | Handwritten weights | Small | 1.5 / 64 | 0 / 3 / 61 | 13 |

Each contest combines two five-rally games with swapped sides on the same seed. A contest win earns one point; a tie earns half. Equal points share rank.

Rules: `pong-v2` · Contract: `dense-mlp` · Protocol: `paired-round-robin` · Seeds: `public-league`.

Benchmark: `26842e45ea71e52a77aac0dcf17d9c840d4345b0fed49905730649d9eaccf3de`.

Generated from the repository’s model bytes by `npm run league`, using ONNX Runtime Web CPU/WASM with one thread. CI reruns the league and checks both committed files. Review the passing workflow on the relevant commit before treating a proposed PR table as verified.

The public seeds are a repeatable development benchmark, not a private training holdout or a universal strength rating. Compare scores only within the same benchmark.

[Full results, model hashes and serve seeds](league.json)

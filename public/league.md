# PongArena league

All 4 bundled models, 16 paired serve sets per matchup, 192 games.

| Rank | Model | Kind | Size | Points | W / D / L | Parameters |
| ---: | --- | --- | --- | ---: | --- | ---: |
| 1 | Large | Trained weights | Large | 45 / 48 | 44 / 2 / 2 | 50819 |
| 2 | Quill | Handwritten weights | Small | 34 / 48 | 33 / 2 / 13 | 163 |
| 3 | Small | Trained weights | Small | 16 / 48 | 15 / 2 / 31 | 235 |
| 4 | Follow | Handwritten weights | Small | 1 / 48 | 1 / 0 / 47 | 13 |

Each contest combines two five-rally games with swapped sides on the same seed. A contest win earns one point; a tie earns half. Equal points share rank.

Rules: `pong` · Contract: `dense-mlp` · Protocol: `paired-round-robin` · Seeds: `public-league`.

Benchmark: `d0db6f77ce6167c5b54af0902bbea80d17d435fe4f3833847a091d6627d77aec`.

Generated from the repository’s model bytes by `npm run league`, using ONNX Runtime Web CPU/WASM with one thread. CI reruns the league and checks both committed files. Review the passing workflow on the relevant commit before treating a proposed PR table as verified.

The public seeds are a repeatable development benchmark, not a private training holdout or a universal strength rating. Compare scores only within the same benchmark.

[Full results, model hashes and serve seeds](league.json)

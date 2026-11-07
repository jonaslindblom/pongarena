# PongArena league

All 4 bundled models, 16 paired serve sets per matchup, 192 games.

| Rank | Model | Kind | Size | Points | W / D / L | Parameters |
| ---: | --- | --- | --- | ---: | --- | ---: |
| 1 | Large | Trained weights | Large | 43.5 / 48 | 41 / 5 / 2 | 50819 |
| 2 | Quill | Handwritten weights | Small | 36 / 48 | 34 / 4 / 10 | 163 |
| 3 | Small | Trained weights | Small | 15.5 / 48 | 14 / 3 / 31 | 235 |
| 4 | Follow | Handwritten weights | Small | 1 / 48 | 0 / 2 / 46 | 13 |

Each contest combines two five-rally games with swapped sides on the same seed. A contest win earns one point; a tie earns half. Equal points share rank.

Rules: `pong-v2` · Contract: `dense-mlp` · Protocol: `paired-round-robin` · Seeds: `public-league`.

Benchmark: `6f1ec7918e9b8a8791ee0e364be1cb55e4c0f1fa41ba9b0c9cf3b28e17831ede`.

Generated from the repository’s model bytes by `npm run league`, using ONNX Runtime Web CPU/WASM with one thread. CI reruns the league and checks both committed files. Review the passing workflow on the relevant commit before treating a proposed PR table as verified.

The public seeds are a repeatable development benchmark, not a private training holdout or a universal strength rating. Compare scores only within the same benchmark.

[Full results, model hashes and serve seeds](league.json)

# PongArena league

All 9 bundled controllers, 16 paired serve sets per matchup, 1152 games.

| Rank | Controller | Kind | Size | Points | W / D / L | Parameters |
| ---: | --- | --- | --- | ---: | --- | ---: |
| 1 | Attacker | Script | — | 123.5 / 128 | 121 / 5 / 2 | — |
| 2 | Defender | Script | — | 111.5 / 128 | 110 / 3 / 15 | — |
| 3 | Prism Small v4 | Handwritten weights | Large | 79 / 128 | 75 / 8 / 45 | 1815 |
| 4 | Quill Plus v4 | Handwritten weights | Small | 70.5 / 128 | 66 / 9 / 53 | 163 |
| 5 | Vector Nano v4 | Handwritten weights | Large | 66.5 / 128 | 60 / 13 / 55 | 487 |
| 6 | Robust Strong v4 | Learned / baseline weights | Large | 54 / 128 | 46 / 16 / 66 | 17795 |
| 7 | Plain Nano v4 | Learned / baseline weights | Large | 42.5 / 128 | 34 / 17 / 77 | 447 |
| 8 | Interceptor | Script | — | 25 / 128 | 23 / 4 / 101 | — |
| 9 | Tracker | Learned / baseline weights | Small | 3.5 / 128 | 1 / 5 / 122 | 163 |

Each contest combines two five-rally games with swapped sides on the same seed. A contest win earns one point; a tie earns half. Equal points share rank.

Rules: `pong-v4` · Contract: `observation-6/action-scores-3/dense-mlp-v3` · Protocol: `paired-round-robin-v1` · Seeds: `public-league-v1`.

Benchmark: `749a7c84860d0a18491f083a7c47621d0273a469ea40b640d6bc6ddaf7cd4a97`.

Generated from the repository’s model bytes and scripts by `npm run league`, using ONNX Runtime Web CPU/WASM with one thread. CI reruns the league and checks both committed files. Review the passing workflow on the relevant commit before treating a proposed PR table as verified.

The public seeds are a repeatable development benchmark, not a private training holdout or a universal strength rating. Compare scores only within the same benchmark.

[Full results, controller hashes and serve seeds](league.json)

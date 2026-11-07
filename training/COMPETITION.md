# Controller competition

The active arena accepts **plain dense neural networks** mapping the six raw observations directly to three action scores. Weight values may be hand-edited, transferred, or learned; an entrant may not encode a separate physics or planning program in its ONNX architecture. Eligibility is determined by the validated dense graph and resource budgets, not metadata or a training-method label.

## Dense-network contract

The active contract identifier is `observation-6/action-scores-3/dense-mlp-v3`. Classic v4 physics remains `pong-v4`; rankings saved under the earlier two-hidden-layer or general-graph contracts are not restored. This contract supersedes the export-graph section of the immutable v4 training package, which remains an archive of the earlier submission format.

Submit one opset 18 ONNX file, at most 256 KiB, with float32 `observation` input `[1, 6]` and `action_scores` output `[1, 3]`. The observations are `(ballX − 180) / 180`, `(ballY − 80) / 80`, `ballVX / 6`, `ballVY / 6`, `(ownPaddleY − 80) / 80`, and `(opponentPaddleY − 80) / 80`. For the right-side player, negate the centered ball X and X velocity so its own paddle is always on the left in observation space. Both players decide from the same pre-move state. Output scores must be finite; the first maximum selects `up`, `stay` or `down`.

The display rotates these simulation coordinates: the legacy actions appear as left, stay and right on screen. Models and training code retain their existing coordinates.

- Use a single feed-forward chain of one to five affine layers (`Gemm`): zero to four hidden layers, followed by a linear three-score output. Hidden widths may vary within the resource budgets.
- Apply the same ordinary `Relu` or `Tanh` activation after every hidden affine layer. There is no activation after the output layer.
- Every affine layer consumes only the immediately preceding activation and its own unique embedded float32 weight matrix and bias. Weights and biases must be finite and used exactly once. Dense coefficients are fixed parameters: `alpha = beta = 1`, `transA = 0`, and `transB` may only select matrix storage orientation.
- Supply the six observations unchanged. No branches, skip connections, scalar instruction chains, runtime input products, computed parameters, shared tensors, auxiliary input features, separate lookup/physics operations, or input/output processing are allowed in the submitted graph. Only the canonical dense/activation serialization is accepted; standalone `MatMul` or `Add` nodes are rejected.

Ordinary hidden neurons may represent useful features through their dense weights. A plain dense network can encode knowledge through its numerical weights and biases. This contract limits the architecture; it does not require formal training. The retained model roster and maintained training workflow are documented in [README.md](README.md); [catalog.json](catalog.json) records exact model identities.

The earlier Handcrafted Nano, Handcrafted Small, Counter Small, and Strategist exports are ineligible archives. Their recorded outcomes remain historical evidence under their old graph contract, not current dense-network standings.

The **League** page opens with a checked-in reference round robin. It also offers a separate local benchmark for comparing imported weights and inspecting size/strength tradeoffs. Local browser results and exported JSON files are under the user's control; they do not update the reference standings.

**All controllers** ranks eligible weight models and the built-in Attacker, Defender and Interceptor scripts together. Scripts receive the same league points and ranks, including in script-only fields. They use the same observations, movement limits, serves and side assignments as models. Scripts are labelled explicitly, with no parameter/MAC counts or model size class. All controllers does not change the dense-network submission contract for weight files.

## Model classes

These classes are **weights only**. Both limits must be satisfied. Small models may also compete in Large. A class run includes only the selected eligible entrants and runs its own complete round-robin; filtering an All controllers table does not create a class championship.

| Class | Embedded float32 scalars | Multiply–accumulates per decision |
| --- | ---: | ---: |
| Small | 256 | 1,024 |
| Large | 65,536 | 65,536 |

These are engineering budgets, not measured capability boundaries. A changed class budget changes the benchmark identity. Earlier Nano, Standard and Open classes are retired; older Small results with a larger budget also cannot be restored as current class runs. A model's historical name, such as Plain Nano or Prism Small, does not determine its class.

The existing file and intermediate-tensor limits also apply. The canonical dense structure permits at most nine operators: five affine layers and four hidden activations. Count every embedded weight and bias, including zeros; shared or unused tensors are ineligible. Parameters and dense multiply–accumulates are derived from the validated graph, not filename or a claimed trainable-parameter count.

For the active six-input/three-action interface, these plain network architectures illustrate the budgets. Sizes alone do not establish trained performance:

| Architecture | Parameters | Dense MACs | Smallest class |
| --- | ---: | ---: | --- |
| `6 → 12 → 4 → 3 → 3` (Quill Plus) | 163 | 141 | Small |
| `6 → 16 → 3` | 163 | 144 | Small |
| `6 → 16 → 16 → 3` | 435 | 400 | Large |
| `6 → 16 → 12 → 8 → 3` | 447 | 408 | Large |
| `6 → 32 → 32 → 3` | 1,379 | 1,312 | Large |
| `6 → 64 → 64 → 3` | 4,803 | 4,672 | Large |
| `6 → 128 → 128 → 3` | 17,795 | 17,536 | Large |

Parameter count measures capacity; it does not promise a tactical or entertaining opponent. Inference latency is reported separately when measured because it depends on browser, hardware, runtime and graph structure. It does not determine the class.

## Identity and compatibility

Each weight model's identity is the SHA-256 of its complete ONNX bytes. A renamed file remains the same entrant. Byte-identical files cannot enter twice, even under different display names. Different serializations of the same function can still have different hashes.

Each script's identity is the SHA-256 of UTF-8 JSON containing its policy key and the complete `src/scripted.ts` source, including shared planning helpers, with CRLF line endings normalized to LF. Different policies have different identities; source-content changes other than this line-ending normalization change those identities. Duplicate script identities cannot enter twice. Script entries explicitly record `type: "script"` and null parameter/MAC counts. Older weight records without a type remain compatible and retain their fingerprints.

Before entering a run, every model must pass the game's six-input/three-action **dense-only** validator. Archived eight-input/five-action Burst weights are not eligible for the active arena, and their historical rankings cannot be restored as Classic results. Physics-feature hybrids and the retired handcrafted programs are also rejected, even when they expose the correct ports. Compatibility adapters would require a separate contract. A controller's human label or training method cannot replace graph validation.

The benchmark fingerprint includes:

- Ranking protocol version (`paired-round-robin-v1`).
- An immutable rules identifier and the observation/action contract identifier.
- League identifier and its exact parameter/MAC budgets (null for All controllers).
- The selected roster's sorted identities and validated resource counts, with explicit script types and null script counts.
- The sorted distinct unsigned 32-bit seed set.
- Five rallies per game, both side assignments, and the scoring/tie policy.

Display names and UI ordering are excluded. The caller must change the rules identifier whenever simulation, termination or policy evaluation semantics change. An identifier containing the frozen engine's content hash is preferable for a published competition.

## Schedule and scoring

Each unordered controller pair faces every seed twice: first A on the left, then B on the left. Both games replay the same five seeded rallies. All pairs use the same seed set. The arena can run four, eight or sixteen seed sets per pairing; the runner supports up to 64 distinct seeds.

The two games form one **paired contest**, with ten rallies in total. Add each controller's rally points across its two side assignments. More rally points wins the contest; equal points draw it. League points are one for a contest win, half for a draw and zero for a loss. Sort the complete table by total league points, with equal scores sharing rank. Do not silently break a tie using returns, rally length, parameter count, inference time or a name. Rally points for/against and drawn rallies are descriptive diagnostics.

This makes side assignment symmetric and avoids ranking a player by a favorable five-rally serve order. It does not eliminate sampling noise or non-transitive matchups. An A-beats-B-beats-C-beats-A field may tie; the pairwise records are retained in the JSON export. Scores across different rosters or seed sets are different benchmarks and are not interchangeable ratings.

The game rules determine when a rally ends, including the 1,600-tick limit and exact-loop detection for deterministic automated controllers. A drawn rally awards no rally point. A technical inference failure stops the league as failed; it does not silently award the opponent a win.

## Reference standings and model PRs

The [reference table](../public/league.md) is generated from all bundled models and scripts in `src/roster.ts`, model bytes in `public/models/`, and the 16 fixed seeds in `src/league.ts` (`public-league-v1`). It is an All controllers league; Small and Large remain options for separate local runs. The generator validates each model and uses single-threaded ONNX Runtime Web CPU/WASM with the same match and scoring implementation as browser leagues.

From the repository root:

```sh
npm run league
npm run league:check
```

The first command writes `public/league.json` and `public/league.md`. The JSON contains the complete specification and both games for every paired contest; the Markdown table makes the standings readable on GitHub. The second command replays the tournament and requires both files to match byte for byte. Model PRs commit both refreshed files alongside the model and catalog changes; see [CONTRIBUTING.md](../CONTRIBUTING.md).

The included **Check and rank** GitHub Actions workflow runs tests, regenerates the results, checks the committed snapshot, builds the app, and publishes a workflow summary and downloadable artifact. Its repository token is read-only; it does not push updates. These workflows have been prepared locally, not yet run on GitHub. Once hosted, review the passing workflow on the relevant commit before treating a proposed PR table as verified.

The public seeds support reproducibility, not independent training claims. Keep development and final evaluation separate, freeze a candidate before opening fresh evaluation seeds, and compare it against a common opponent field. Seeds are not explicit inputs, but a network can recognize opening states. Scores across different rosters or seed sets are different benchmarks.

## Cancellation, storage and reproduction

The runner evaluates one game at a time and yields between games. The injected headless game runner must also yield periodically during the game and check its `AbortSignal` before accepting inference results. The UI can cancel without skipping simulation states. Only completed side pairs enter provisional totals. Cancelled, failed or still-running leagues have no assigned ranks.

The browser retains the last complete league in local storage when available. Restoring checks its benchmark fingerprint, full pair/seed schedule and valid game scores, then reconstructs standings. This catches accidental incomplete or incompatible records; it does not authenticate scores against a malicious local editor. If storage is unavailable, the completed result can still be exported. Exported JSON contains the full specification and both game results for every completed contest.

**Repeat benchmark** uses the saved controller identities, seed set, league and rules contract. All those model files and script versions must still be available. **New serves** draws a fresh comparison seed set. New model development should use fresh reserved evaluation seeds rather than repeatedly optimizing against a displayed leaderboard batch. Seeds are not explicit model inputs, but a network can recognize opening states, so a fixed public seed list is not a sufficient anti-overfitting measure.

Ranking quality also depends on the reference field. Include defensive and attacking opponents and multiple learned policies. Keep tactical probes and human playability measurements separate from league points: a long rally, high teacher agreement or a well-placed return is useful evidence but is not itself a match win.

## Verify

```sh
node --import tsx --test tests/competition.test.ts
npm run league:check
```

The tests enforce side-pair symmetry, weights-only size/compute boundaries, mixed and script-only standings, stable benchmark identity, shared ranks for non-transitive ties, duplicate-controller rejection, cancellation/failure handling and complete-record restoration.

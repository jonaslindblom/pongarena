# Rebuild the programmed models

Follow, Predict, Attack, Counter and Quill each have a standalone Python constructor.
They deterministically calculate fixed dense ONNX weights from explicit formulas
using only NumPy and ONNX. Running a constructor rebuilds the selected design;
it does not repeat the development process.

Quill's formulas were developed and revised with Codex using simulated match
feedback. This was iterative design optimization, without gradient training or a
conventional numerical parameter-fitting algorithm. **Programmed** describes the
formula-based construction, not an absence of feedback or optimization. See
[Quill's design](../docs/QUILL.md) for the development process and computations.

From the repository root, create a Python 3.12 environment with only NumPy and ONNX:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install numpy==2.5.4 onnx==1.23.2
```

Rebuild all five models into a temporary directory and check that their bytes match
the shipped files:

```sh
rebuild_dir=$(mktemp -d)
for model in follow predict attack counter quill; do
  .venv/bin/python "programmed/$model.py" --out "$rebuild_dir/$model.onnx"
  cmp "$rebuild_dir/$model.onnx" "public/models/$model.onnx" || exit 1
done
```

Each constructor prints the output path and SHA-256. No output from `cmp` means
the files match. These checks leave the shipped models and league results unchanged.
To rebuild just one model, run its constructor with the same `--out` option.

The [method-label revision](../experiments/model-metadata.json) and
[development-description revision](../experiments/programmed-method-metadata.json)
record metadata-only updates to these five ONNX files. Their graphs, tensors and
other model fields remain identical; only the file hashes changed. The audits
retain the earlier identities and matching hashes of each model payload with
metadata removed.

| Model | Constructor | Design |
| --- | --- | --- |
| Follow | [follow.py](follow.py) | [Every weight explained](../docs/FOLLOW.md) |
| Predict | [predict.py](predict.py) | [Landing prediction](../docs/PREDICT.md) |
| Attack | [attack.py](attack.py) | [Prediction and corner aiming](../docs/ATTACK.md) |
| Counter | [counter.py](counter.py) | [Prediction, aiming and recovery](../docs/COUNTER.md) |
| Quill | [quill.py](quill.py) | [Geometric weight formulas](../docs/QUILL.md) |

If you change a constructor, export its new weights to `public/models/<id>.onnx`
and follow the [model submission checks](../CONTRIBUTING.md#submit-a-model).

# Rebuild the handwritten models

Follow, Predict, Attack, Counter and Quill each have a standalone Python constructor.
They write fixed dense ONNX weights without training code, checkpoints or PyTorch.

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
  .venv/bin/python "handwritten/$model.py" --out "$rebuild_dir/$model.onnx"
  cmp "$rebuild_dir/$model.onnx" "public/models/$model.onnx" || exit 1
done
```

Each constructor prints the output path and SHA-256. No output from `cmp` means
the files match. These checks leave the shipped models and league results unchanged.
To rebuild just one model, run its constructor with the same `--out` option.

| Model | Constructor | Design |
| --- | --- | --- |
| Follow | [follow.py](follow.py) | [Every weight explained](../docs/FOLLOW.md) |
| Predict | [predict.py](predict.py) | [Landing prediction](../docs/PREDICT.md) |
| Attack | [attack.py](attack.py) | [Prediction and corner aiming](../docs/ATTACK.md) |
| Counter | [counter.py](counter.py) | [Prediction, aiming and recovery](../docs/COUNTER.md) |
| Quill | [quill.py](quill.py) | [Geometric weight formulas](../docs/QUILL.md) |

If you change a constructor, export its new weights to `public/models/<id>.onnx`
and follow the [model submission checks](../CONTRIBUTING.md#submit-a-model).

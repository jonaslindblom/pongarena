"""Replay a fixed finite Quill hypothesis matrix, without training or fitting.

Constructors write analytical dense weights once. The evaluator compares each
frozen candidate with the seven original frozen reference models on the same
exploratory serves. There is no optimizer, automated parameter search, model
selection, or modification of the submitted models in public/models.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import precision_variants
import recovery_variants
import shot_variants
import small_variants

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE_MODELS = ['quill', 'large', 'small', 'counter97', 'attack75', 'predict75', 'follow']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path,
                        default=Path('scratch/quill-experiments/replay'))
    parser.add_argument('--report', type=Path, default=Path('docs/quill-exploration.json'))
    parser.add_argument('--seed', type=int, default=480193)
    parser.add_argument('--count', type=int, default=32)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError('Choose a fresh --report path to preserve existing experiment evidence')
    if not 1 <= args.count <= 512 or args.seed < 0:
        raise ValueError('Use 1–512 serve seed sets and a nonnegative RNG seed')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = {'baseline-quill': ROOT / 'public/models/quill.onnx'}
    hypotheses = {'baseline-quill': {'source': 'handwritten/quill.py', 'settings': {}}}

    def record(name, source, settings, exporter):
        if name in candidates:
            raise ValueError(f'Duplicate hypothesis name: {name}')
        path = args.out_dir / f'{name}.onnx'
        exporter(path)
        candidates[name] = path
        hypotheses[name] = {'source': source, 'settings': settings}
        print(f'Constructed {name}: {digest(path)}', flush=True)

    for name in small_variants.variants():
        record(name, 'experiments/quill/small_variants.py', {'variant': name},
               lambda path, variant=name: small_variants.export(path, variant=variant))
    for name, settings in precision_variants.variants().items():
        record(name, 'experiments/quill/precision_variants.py', settings,
               lambda path, options=settings: precision_variants.export(path, **options))
    for name, settings in {
        'recovery': {'step': .25, 'exact_ticks': True},
        'recovery-max-speed': {'step': .25, 'exact_ticks': True, 'max_speed_only': True},
    }.items():
        record(name, 'experiments/quill/recovery_variants.py', settings,
               lambda path, options=settings: recovery_variants.export(path, **options))
    for name, settings in shot_variants.variants().items():
        record(name, 'experiments/quill/shot_variants.py', settings,
               lambda path, options=settings: shot_variants.export(path, **options))

    command = [sys.executable, str(HERE / 'evaluate.py')]
    for name, path in candidates.items():
        command += ['--candidate', f'{name}={path.resolve()}']
    for name in REFERENCE_MODELS:
        command += ['--opponent', f'{name}={ROOT / "public/models" / (name + ".onnx")}']
    command += ['--seed', str(args.seed), '--count', str(args.count),
                '--label', 'exploratory', '--out', str(args.report.resolve())]
    print(f'Evaluating {len(candidates)} fixed designs against {len(REFERENCE_MODELS)} '
          f'original reference models on {args.count} shared paired serve sets.', flush=True)
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)
    report = json.loads(args.report.read_text())
    report['design_matrix'] = hypotheses
    sources = ['experiments/quill/run.py', 'experiments/quill/evaluate.py',
               'experiments/quill/small_variants.py', 'experiments/quill/precision_variants.py',
               'experiments/quill/recovery_variants.py', 'experiments/quill/shot_variants.py',
               'handwritten/quill.py', 'training/game.py']
    report['source_sha256'] = {source: digest(ROOT / source) for source in sources}
    report['interpretation'] = (
        'Exploratory comparison of a fixed finite design matrix, not independent holdout evidence. '
        'The original Quill is both a baseline candidate and one of seven fixed reference '
        'opponents; its self-match is included as a draw. All candidates face exactly the same '
        'reference field and paired serve seeds. Published league rankings require separate '
        'fresh evaluation through the shared browser/WASM simulator after final model freeze.'
    )
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Added fixed-design and source provenance to {args.report}', flush=True)


if __name__ == '__main__':
    main()

"""Record complete native ONNX matches for live-engine and browser WASM parity."""
import hashlib
import json
import sys
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

from policy import ROOT
sys.path.insert(0, str(ROOT / 'training/v3'))
from reference import serve, advance, observation, ended_match, ended_rally, LoopDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT / 'training/results/training-improvement')
    parser.add_argument('--name', default='strategist-v3')
    parser.add_argument('--seed-start', type=int, default=1_270_000_000)
    args = parser.parse_args()
    directory = args.results
    frozen = json.loads((directory / 'candidate/freeze.json').read_text())
    paths = {args.name: ROOT / frozen['onnx'],
             'selfplay-v2': ROOT / 'public/models/selfplay-v2.onnx'}
    if hashlib.sha256(paths[args.name].read_bytes()).hexdigest() != frozen['onnx_sha256']:
        raise ValueError('The selected candidate changed.')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    sessions = {name: ort.InferenceSession(str(path), sess_options=options,
                 providers=['CPUExecutionProvider']) for name, path in paths.items()}
    matches = []
    for names in [(args.name, 'selfplay-v2'), ('selfplay-v2', args.name), (args.name, args.name)]:
        for seed in [args.seed_start + 7, args.seed_start + 43]:
            frame, loops, steps = serve(seed), LoopDetector(), []
            while not ended_match(frame):
                scores = None if ended_rally(frame) else [sessions[name].run(None,
                    {'observation': np.array(observation(frame, side), dtype=np.float32)[None]})[0][0].tolist()
                    for side, name in enumerate(names)]
                frame = serve(seed, frame) if scores is None else advance(frame, scores, loops)
                steps.append(dict(scores=scores, state=frame))
            matches.append(dict(seed=seed, players=names, steps=steps))
    output = directory / 'verification'
    output.mkdir(exist_ok=True)
    (output / 'onnx-replays.json').write_text(json.dumps(matches) + '\n')
    print(f'Recorded {len(matches)} matches, {sum(len(m["steps"]) for m in matches)} frames.')


if __name__ == '__main__':
    main()

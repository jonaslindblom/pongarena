"""Record complete native ONNX matches for live-engine and browser WASM parity."""
import hashlib
import json
import sys

import numpy as np
import onnxruntime as ort

from policy import ROOT
sys.path.insert(0, str(ROOT / 'training/v3'))
from reference import serve, advance, observation, ended_match, ended_rally, LoopDetector


def main():
    directory = ROOT / 'training/results/training-improvement'
    frozen = json.loads((directory / 'candidate/freeze.json').read_text())
    paths = {'strategist-v3': ROOT / frozen['onnx'],
             'selfplay-v2': ROOT / 'public/models/selfplay-v2.onnx'}
    if hashlib.sha256(paths['strategist-v3'].read_bytes()).hexdigest() != frozen['onnx_sha256']:
        raise ValueError('The selected candidate changed.')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    sessions = {name: ort.InferenceSession(str(path), sess_options=options,
                 providers=['CPUExecutionProvider']) for name, path in paths.items()}
    matches = []
    for names in [('strategist-v3', 'selfplay-v2'), ('selfplay-v2', 'strategist-v3'),
                  ('strategist-v3', 'strategist-v3')]:
        for seed in [1_270_000_007, 1_270_000_043]:
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

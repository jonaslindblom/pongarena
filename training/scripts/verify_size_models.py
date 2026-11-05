"""Make complete native-ONNX replay fixtures for browser parity checks."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from train_size_study import ROOT, digest


def main():
    release = ROOT/'training/results/burst-study/models/release'
    frozen = json.loads((release/'manifest.json').read_text())
    for name, sha in frozen['files'].items():
        assert digest(ROOT/name) == sha, name
    path = ROOT/'training/burst/reference.py'
    spec = importlib.util.spec_from_file_location('burst_scalar_parity', path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    sessions = {name:ort.InferenceSession(str(release/name/'model.onnx'), sess_options=options,
        providers=['CPUExecutionProvider']) for name in frozen['selected']}
    fixtures, frames, decisions = [], 0, 0
    pairs = [('micro', 'tiny'), ('tiny', 'small'), ('small', 'medium'),
             ('medium', 'large'), ('large', 'large'), ('tiny', 'medium')]
    for index, players in enumerate(pairs):
        seed = [3_010_000_007, 3_010_000_043][index % 2]
        frame = reference.serve(seed)
        loops = reference.LoopDetector()
        steps = []
        while not reference.ended_match(frame):
            if reference.ended_rally(frame):
                scores = None
                frame = reference.serve(seed, frame)
            else:
                scores = [sessions[name].run(None, {'observation':np.array(reference.observation(frame, side), dtype=np.float32)[None]})[0][0].tolist()
                          for side, name in enumerate(players)]
                frame = reference.advance(frame, scores, loops)
                decisions += 2
            steps.append(dict(scores=scores, state=frame))
            frames += 1
        fixtures.append(dict(seed=seed, players=['burst-'+name for name in players], steps=steps))
    encoded = json.dumps(fixtures, separators=(',', ':'))+'\n'
    (release/'onnx-replays.json').write_text(encoded)
    (ROOT/'public/training/burst-onnx-replays.json').write_text(encoded)
    record = dict(reference_sha256=digest(path), matches=len(fixtures), frames=frames,
                  decisions=decisions, backend='Native ONNX Runtime, scalar burst reference')
    (release/'native-replay-verification.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()

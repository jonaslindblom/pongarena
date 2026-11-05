"""Verify exported actors and produce independent Python/ONNX replays for the TS check."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from env import Arena, scripted
from policy import Actor, actions, ROOT
from reference import serve, advance, observation, ended_match, ended_rally, LoopDetector


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=Path('training/results/v3'))
    p.add_argument('--out', type=Path, default=Path('training/results/v3'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    sessions = {name: ort.InferenceSession(str(ROOT / f'public/models/{name}.onnx'), sess_options=options)
                for name in ['selfplay-v3', 'apprentice-v3', 'challenger']}
    env = Arena(64, seed=721)
    observations = []
    for _ in range(64):
        observations.extend([env.obs(), env.obs(1)])
        outcome, _ = env.step(np.stack([scripted(env.obs(), True), scripted(env.obs(1))], 1))
        env.reset(np.flatnonzero(outcome >= 0))
    obs = np.concatenate(observations)
    checks = {}
    for stage, name in [('imitation', 'apprentice-v3'), ('ppo', 'selfplay-v3')]:
        actor = Actor()
        actor.load_state_dict(torch.load(args.run / f'{stage}.pt', weights_only=True))
        with torch.no_grad():
            expected = actor(torch.from_numpy(obs)).numpy()
        actual = np.array([sessions[name].run(None, {'observation': row[None]})[0][0] for row in obs])
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
        checks[name] = dict(observations=len(obs), max_abs_error=float(np.abs(actual - expected).max()),
                            action_agreement=1., sha256=hashlib.sha256((ROOT / f'public/models/{name}.onnx').read_bytes()).hexdigest())
    matches = []
    for names in [('selfplay-v3', 'challenger'), ('challenger', 'selfplay-v3'), ('selfplay-v3', 'apprentice-v3'), ('selfplay-v3', 'selfplay-v3')]:
        for seed in [2_200_000_007, 2_200_000_043]:
            frame = serve(seed)
            loops = LoopDetector()
            steps = []
            while not ended_match(frame):
                scores = None if ended_rally(frame) else [sessions[name].run(None, {'observation': np.array(observation(frame, side), dtype=np.float32)[None]})[0][0].tolist() for side, name in enumerate(names)]
                frame = serve(seed, frame) if scores is None else advance(frame, scores, loops)
                steps.append(dict(scores=scores, state=frame))
            matches.append(dict(seed=seed, players=names, steps=steps))
    (args.out / 'export-check.json').write_text(json.dumps(checks, indent=2) + '\n')
    (args.out / 'onnx-replays.json').write_text(json.dumps(matches) + '\n')
    print(json.dumps(checks, indent=2))
    print('Wrote', len(matches), 'complete ONNX match replays for TypeScript verification.')


if __name__ == '__main__':
    main()

"""Check physics fixtures, teacher decisions, model contracts and export parity."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from game import Arena, LoopDetector, advance, observation, serve
from models import network
from teachers import teacher

HERE = Path(__file__).resolve().parent


def packed(frame):
    return ([frame[k] for k in ('tick', 'rally', 'rallyTick', 'x', 'y', 'vx', 'vy')]
            + frame['paddles'] + frame['points']
            + [frame['returns'], [None, 'serve', 'return', 'point', 'draw'].index(frame['event']),
               frame.get('actor', -1), {None: 0, 'loop': 1, 'timeout': 2}[frame.get('drawReason')]])


def physics():
    fixtures = json.loads((HERE / 'fixtures/game.json').read_text())
    for case in fixtures['serves']:
        frame = None
        for expected in case['frames']:
            frame = serve(case['seed'], frame)
            assert frame == expected
    for case in fixtures['traces']:
        frame, loops = case['initial'], LoopDetector()
        arena = Arena(1, loop_detection=True)
        arena.load([frame])
        for actions, expected in zip(case['actions'], case['states']):
            frame = advance(frame, np.eye(3, dtype=int)[actions].tolist(), loops)
            assert packed(frame) == expected, case['name']
            outcome, _ = arena.step(np.array([actions]))
            np.testing.assert_array_equal(arena.ball[0], expected[3:7])
            np.testing.assert_array_equal(arena.paddles[0], expected[7:9])
            assert arena.ticks[0] == expected[2] and arena.returns[0] == expected[11]
            assert arena.draw_reason[0] == expected[14]
            assert outcome[0] == (expected[13] if expected[12] == 3 else 2 if expected[12] == 4 else -1)
        for side in (0, 1):
            np.testing.assert_array_equal(observation(frame, side), case['observations'][side])
            np.testing.assert_array_equal(arena.obs(side)[0], case['observations'][side])
    # Independent scalar/vector one-step comparisons include both walls and
    # contact edges at every allowed along-court speed and across-court velocity.
    rng = np.random.default_rng(8019)
    frames = []
    for _ in range(2048):
        frame = serve(0)
        frame.update(x=int(rng.integers(15, 346)), y=int(rng.integers(4, 157)),
                     vx=int(rng.integers(5, 14) * rng.choice([-1, 1])),
                     vy=int(rng.integers(-9, 10)), paddles=rng.integers(15, 146, 2).tolist())
        frames.append(frame)
    actions = rng.integers(0, 3, (len(frames), 2))
    arena = Arena(len(frames))
    arena.load(frames)
    outcome, _ = arena.step(actions)
    for i, prior in enumerate(frames):
        frame = advance(prior, np.eye(3, dtype=int)[actions[i]].tolist())
        np.testing.assert_array_equal(arena.ball[i], [frame[k] for k in ('x', 'y', 'vx', 'vy')])
        np.testing.assert_array_equal(arena.paddles[i], frame['paddles'])
        assert outcome[i] == (frame['actor'] if frame['event'] == 'point' else -1)
        for side in (0, 1):
            np.testing.assert_array_equal(arena.obs(side)[i], observation(frame, side))


def teachers():
    data = json.loads((HERE / 'fixtures/teachers.json').read_text())
    probes = np.array(data['probes'])
    obs = ((probes[:, :6] - [180, 80, 0, 0, 80, 80]) / [180, 80, 6, 6, 80, 80]).astype(np.float32)
    for i, kind in enumerate(data['kinds']):
        np.testing.assert_array_equal(teacher(kind)(obs), probes[:, 6 + i])


def contract(path, limit):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    assert [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim] == [1, 6]
    assert [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim] == [1, 3]
    assert all(node.op_type in ('Gemm', 'Tanh', 'Relu') for node in model.graph.node)
    parameters = sum(int(np.prod(t.dims)) for t in model.graph.initializer)
    macs = sum(int(np.prod(t.dims)) for t in model.graph.initializer if len(t.dims) == 2)
    assert parameters <= limit and macs <= (1024 if limit == 256 else 65536)
    return parameters, macs


def export_parity(run):
    saved = torch.load(run / 'model.pt', map_location='cpu', weights_only=True)
    policy = network(saved['widths'], saved['activation'])
    policy.load_state_dict(saved['state_dict'])
    policy.eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(run / 'model.onnx'), options, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(8101)
    obs = rng.uniform([-1, -1, -13/6, -1.5, -1, -1], [1, 1, 13/6, 1.5, 1, 1], (512, 6)).astype(np.float32)
    with torch.inference_mode():
        for row in obs:
            expected = policy(torch.from_numpy(row[None])).numpy()
            actual = session.run(None, {'observation': row[None]})[0]
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-5)
            assert actual.argmax() == expected.argmax()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, help='Also verify the two trained models in this directory')
    parser.add_argument('--run', type=Path, help='Also compare a training checkpoint with its exported ONNX')
    args = parser.parse_args()
    torch.set_num_threads(1)
    from evaluate import DEVELOPMENT_SEEDS, FINAL_SEEDS
    assert all(1_000_000_000 <= seed < 2_000_000_000 for seed in DEVELOPMENT_SEEDS)
    assert len(set(DEVELOPMENT_SEEDS)) == 32
    assert all(2_510_000_000 <= seed < 2_600_000_000 for seed in FINAL_SEEDS)
    assert len(set(FINAL_SEEDS)) == 128
    assert set(DEVELOPMENT_SEEDS).isdisjoint(FINAL_SEEDS)
    physics()
    teachers()
    if args.models:
        published = json.loads((HERE / 'results.json').read_text())['models']
        for name in ('small', 'large'):
            path = args.models / f'{name}.onnx'
            params, macs = contract(path, 65536 if name == 'large' else 256)
            record = published[name]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], f'{name}: published model hash changed'
            assert (params, macs) == (record['parameters'], record['macs']), f'{name}: published resource counts changed'
            recipe = HERE / record['recipe']
            assert hashlib.sha256(recipe.read_bytes()).hexdigest() == record['recipe_sha256'], f'{name}: published recipe hash changed'
            print(f'{name}: {params} parameters, {macs} MACs; published hashes match')
    if args.run:
        export_parity(args.run)
    print('Physics, teacher decisions and model checks passed.')


if __name__ == '__main__':
    main()

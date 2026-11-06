"""Check current weights and training physics against independent runtimes."""
import importlib.util
import json
import numpy as np
import torch
from pace_study import CONFIGS, PaceArena
from policy import ROOT, OnnxPolicy, actions, load_builtin


def main():
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('classic_v4', ROOT/'training/v4/reference.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    rng = np.random.default_rng(8201)
    probes = rng.uniform(-2,2,(1024,6)).astype(np.float32)
    catalog = json.loads((ROOT/'training/catalog.json').read_text())
    verified_checkpoints = 0
    missing_checkpoints = []
    for model in catalog['models']:
        net = load_builtin(model['id'])
        np.testing.assert_array_equal(actions(net,probes),OnnxPolicy(ROOT/model['file'])(probes),err_msg=model['id'])
        if model.get('checkpoint') and not (ROOT/model['checkpoint']).exists():
            missing_checkpoints.append(model['id'])
        elif model.get('checkpoint'):
            verified_checkpoints += 1
            state = torch.load(ROOT/model['checkpoint'],weights_only=True)
            state = {key.removeprefix('core.'):value for key,value in state.items()}
            for key,value in net.state_dict().items():
                torch.testing.assert_close(state[key],value,rtol=0,atol=0,msg=f"{model['id']}: checkpoint must retain deployed weights")
    frames = [ref.serve(int(seed)) for seed in rng.integers(0,2**32,32)]
    arena = PaceArena(len(frames),CONFIGS['pace13-v9-scaled'],loop_detection=False)
    arena.load(frames)
    checked = 0
    for _ in range(400):
        selected = rng.integers(0,3,(len(frames),2))
        outcome,_ = arena.step(selected)
        reset = []
        for i,frame in enumerate(frames):
            scores = [[int(k==action) for k in range(3)] for action in selected[i]]
            frame = ref.advance(frame,scores)
            np.testing.assert_array_equal(arena.ball[i],[frame[k] for k in ('x','y','vx','vy')])
            np.testing.assert_array_equal(arena.paddles[i],frame['paddles'])
            assert arena.ticks[i] == frame['rallyTick']
            assert arena.returns[i] == frame['returns']
            expected = frame['actor'] if frame['event']=='point' else 2 if frame['event']=='draw' else -1
            assert outcome[i] == expected, 'Training outcomes must match the frozen v4 scoring rules'
            for side in (0,1):
                np.testing.assert_array_equal(arena.obs(side)[i],ref.observation(frame,side))
            frames[i] = frame
            if outcome[i]>=0:
                reset.append(i)
            checked += 1
        if reset:
            arena.reset(reset)
            for i in reset:
                # Reset's selected seed/rally must preserve the reference opening sequence.
                prior = {'rally':int(arena.rallies[i])-1,'tick':0,'points':[0,0]}
                frames[i] = ref.serve(int(arena.seeds[i]),None if arena.rallies[i]==1 else prior)
    print(f'PASS: {len(catalog["models"])} models × 1,024 native/Torch actions; {verified_checkpoints} exact checkpoints; {checked} v4 transitions and observations')

    if missing_checkpoints:
        print('SKIPPED local checkpoint checks (files not present): '+', '.join(missing_checkpoints))


if __name__ == '__main__':
    main()

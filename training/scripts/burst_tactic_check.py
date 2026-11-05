"""Replay all alternative tactical branches through the independent scalar port."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('burst_ref',ROOT/'training/burst/reference.py')
ref=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def main():
    out=ROOT/'training/results/burst-study'
    data=json.loads((out/'rules-tactics.json').read_text())
    record=data['pairs']['attack:safe']['replays'][0]
    branches=data['actual_tactic']['counter_branches']
    scores=lambda aa:[[int(a==i) for i in range(5)] for a in aa]
    total=0
    for branch in branches:
        frame,loop=ref.serve(record['seed']),ref.LoopDetector()
        for old in record['frames']:
            if old['tick']>=branch['start']['tick']:break
            frame=ref.advance(frame,scores(old['actions']),loop)
        for expected in branch['frames']:
            frame=ref.advance(frame,scores(expected['actions']),loop)
            assert [frame[k] for k in ('x','y','vx','vy')]==expected['ball']
            assert frame['energy']==expected['energy'] and frame['paddles']==expected['paddles']
            assert frame['returns']==expected['returns']
            total+=1
        outcome=2 if frame['event']=='draw' else frame['actor'] if frame['event']=='point' else -1
        assert outcome==branch['outcome']
    result=dict(branches=len(branches),branch_transitions=total,independent_scalar_parity=True,
        full_prefix_loop_history_restored=True,counter_preserves_draw=True)
    (out/'rules-tactic-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':main()

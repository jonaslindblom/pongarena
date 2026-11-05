"""Reconstruct the documented placement comparison through the scalar engine."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('classic_reference',ROOT/'training/v4/reference.py')
ref=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def compact(frame):
    return dict(tick=frame['rallyTick'],ball=[frame[k] for k in ('x','y','vx','vy')],
        paddles=frame['paddles'],returns=frame['returns'])


def replay(match, actions):
    frame=ref.serve(match['seed'],None if match['rally']==1 else
        dict(rally=match['rally']-1,tick=0,points=[0,0]))
    frames=[compact(frame)]
    for code in actions:
        left,right=divmod(int(code),3)
        scores=[[int(i==action) for i in range(3)] for action in (left,right)]
        frame=ref.advance(frame,scores)
        frames.append(compact(frame))
    outcome=frame['actor'] if frame['event']=='point' else 2 if frame['event']=='draw' else -1
    return dict(frames=frames,outcome=outcome,end_tick=frame['rallyTick'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['train','dev','final'],default='train')
    parser.add_argument('--snapshot',type=int,default=961)
    parser.add_argument('--target',type=int)
    parser.add_argument('--out',default='representative.json')
    parser.add_argument('--selection',choices=['precise','broad'],default='precise')
    args=parser.parse_args()
    data=json.loads((HERE/f'{args.phase}.json').read_text())
    snapshot=next(s for s in data['snapshots'] if s['id']==args.snapshot)
    match=next(m for m in data['matches'] if m['id']==snapshot['match_id'])
    choices=[b for b in data['candidates'] if b['snapshot_id']==snapshot['id']]
    original=next(b for b in choices if b['target'] is None)
    target=args.target if args.target is not None else snapshot['selected_target']
    alternative=next(b for b in choices if b['target']==target)
    assert original['value']==-1 and alternative['value']==1
    prefix=match['actions'][:snapshot['tick']]
    baseline=replay(match,prefix+original['actions'])
    changed=replay(match,prefix+alternative['actions'])
    assert baseline['frames'][snapshot['tick']]=={k:snapshot[k] for k in ('tick','ball','paddles','returns')}
    assert baseline['frames'][-1]==original['terminal']
    assert changed['frames'][-1]==alternative['terminal']
    result=dict(snapshot_id=snapshot['id'],phase=args.phase,seed=snapshot['seed'],rally=snapshot['rally'],
        side=snapshot['side'],kinds=snapshot['kinds'],start_tick=snapshot['tick'],
        baseline_contact=original['contact_center'],alternative_contact=alternative['contact_center'],
        contact_tick=snapshot['tick']+alternative['flight_ticks'],
        legal_centers=[b['target'] for b in choices if b['target'] is not None],
        winning_centers=[b['target'] for b in choices if b['target'] is not None and b['value']==1],
        baseline=baseline,alternative=changed,
        selection=('Training-set illustrative example selected for a small contact-position change' if args.selection=='precise'
            else 'Training-set illustrative example selected to show a wider set of winning contact centers')+
            f" after {snapshot['returns']} prior returns; not a random held-out example. Both paths independently checked, including full shared prefix.",
        semantics='Original 1600-tick rally deadline; no loop shortcut for either one-off intervention. Opponent remains the same fixed script; controlled side resumes its original script after the tested contact.')
    (HERE/args.out).write_text(json.dumps(result,separators=(',',':'))+'\n')
    print({k:v for k,v in result.items() if k not in ('baseline','alternative')})


if __name__=='__main__': main()

"""Measure placement tolerance among all retained strict improvements."""
from collections import Counter
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent


def measure(phase):
    data=json.loads((HERE/f'{phase}.json').read_text())
    groups={s['id']:[] for s in data['snapshots']}
    for branch in data['candidates']: groups[branch['snapshot_id']].append(branch)
    rows=[]
    for snapshot in data['snapshots']:
        branches=groups[snapshot['id']]
        if branches[0]['value']!=-1: continue
        legal=sorted(b['target'] for b in branches[1:])
        winners=sorted(b['target'] for b in branches[1:] if b['value']==1)
        if not winners: continue
        longest=current=1
        for a,b in zip(winners,winners[1:]):
            current=current+1 if b==a+1 else 1
            longest=max(longest,current)
        rows.append(dict(snapshot_id=snapshot['id'],seed=snapshot['seed'],rally=snapshot['rally'],
            side=snapshot['side'],kinds=snapshot['kinds'],tick=snapshot['tick'],returns=snapshot['returns'],
            baseline_contact=branches[0]['contact_center'],selected_contact=snapshot['selected_target'],
            legal_centers=legal,winning_centers=winners,winning_count=len(winners),
            winning_fraction=len(winners)/len(legal),longest_consecutive_integer_centers=longest))
    counts=Counter(row['winning_count'] for row in rows)
    fractions=[row['winning_fraction'] for row in rows]
    return dict(strict_opportunities=len(rows),
        winning_center_count_distribution={str(k):v for k,v in sorted(counts.items())},
        exactly_one_winning_center=sum(row['winning_count']==1 for row in rows),
        at_least_three_winning_centers=sum(row['winning_count']>=3 for row in rows),
        at_least_three_consecutive_integer_centers=sum(row['longest_consecutive_integer_centers']>=3 for row in rows),
        median_winning_fraction=statistics.median(fractions),
        mean_winning_fraction=statistics.mean(fractions),
        minimum_winning_fraction=min(fractions),maximum_winning_fraction=max(fractions),
        broad_mid_rally_candidates=sorted([r for r in rows if r['returns']>=4 and r['winning_count']>=3],
            key=lambda r:(-r['longest_consecutive_integer_centers'],-r['winning_fraction'],r['snapshot_id']))[:12],
        rows=rows)


def main():
    result=dict(interpretation='Conditional on enumerated strict loss-to-win opportunities. Correlated states; not independent human-success trials. Winning center means winning under the recorded scripted continuation, not guaranteed play against every response.',
        phases={phase:measure(phase) for phase in ('train','dev','final') if (HERE/f'{phase}.json').exists()})
    (HERE/'breadth.json').write_text(json.dumps(result,indent=2)+'\n')
    for phase,data in result['phases'].items():
        print(phase,{k:v for k,v in data.items() if k not in ('rows','broad_mid_rally_candidates')})
        print('Broad example:',data['broad_mid_rally_candidates'][0] if data['broad_mid_rally_candidates'] else None)


if __name__=='__main__': main()

"""Compact 19-band opponent forecast with a fixed cap-13 safety prior.

No fitting. Teacher coefficients .05 risk, .001 contact travel; first lower-vy
wins exact score ties. Outgoing real bands relax integer/paddle parity. The
cap-13 low/high curves each use30 sparse hinges; all19 destinations share them.
"""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
from pace_study import CONFIGS,decode,tables,NEXT
from env import reflect
RULES=CONFIGS['pace13-v9-scaled'];TAB=tables(RULES)
V=np.arange(-9,10,dtype=np.float64)[None,:]


def select_destination(scores,destinations):
    """Two max reductions + ReLU gates; no ArgMax op or score perturbation.

    Both reductions can use the same constant pooling matrices. The first gate
    identifies all float32 maximum scores; max over penalized integer indices
    then resolves their tie without changing any score order.
    """
    scores=scores.astype(np.float32)
    maximum=scores.max(1,keepdims=True)
    equal=np.maximum(np.float32(0),np.float32(1)-np.float32(2**48)*(maximum-scores))
    assert np.isin(equal,[0,1]).all(),equal
    indices=np.arange(19,dtype=np.float32)[None,:]
    index_scores=-indices-np.float32(32)*(np.float32(1)-equal)
    negative_index=index_scores.max(1,keepdims=True)
    selected=np.maximum(np.float32(0),np.float32(1)-np.abs(indices+negative_index))
    assert (selected.sum(1)==1).all()
    actual=(selected*destinations).sum(1)
    expected=destinations[np.arange(len(scores)),scores.argmax(1)]
    np.testing.assert_array_equal(actual,expected)
    return actual


def outgoing_target(obs,return_details=False):
    vx,own,other,ticks,height=decode(obs)
    stage=np.abs(vx);out=np.minimum(13,stage+1)
    F=np.ceil(330/out)
    h=height[:,None];p=other[:,None];t=ticks[:,None]
    scale=(65/(3*out))[:,None]
    low=np.maximum(np.maximum(h-(V+.5)*scale,h-15),15)
    high=np.minimum(np.minimum(h-(V-.5)*scale,h+15),145)
    low=np.maximum(low,p-2*t);high=np.minimum(high,p+2*t)
    gap=np.maximum(0,low-high)
    destinations=reflect(h+F[:,None]*V).astype(np.float32)
    a=TAB['low'][13,destinations.astype(int)]
    b=TAB['high'][13,destinations.astype(int)]
    floor=np.maximum(0,(a-b)/2)
    # Minimize safety risk, then contact travel, inside each real band.
    safe_low=a-2*F[:,None]-floor
    safe_high=b+2*F[:,None]+floor
    center=np.clip(np.clip(p,safe_low,safe_high),low,high)
    risk=np.maximum(floor,np.maximum(a-center-2*F[:,None],center-2*F[:,None]-b))
    travel=np.abs(center-p)
    expected=np.clip(TAB['mid'][stage,height],own-2*np.maximum(0,ticks-1),own+2*np.maximum(0,ticks-1))
    margin=np.abs(destinations-expected[:,None])-15-2*F[:,None]-2
    penalty=32768*np.maximum(0,gap-1/256)
    score=margin-.05*risk-.001*travel-penalty
    target=select_destination(score,destinations)
    if return_details:return target,dict(score=score,destinations=destinations,low=low,high=high,risk=risk,travel=travel,expected=expected,gap=gap)
    return target


def compose(incoming_policy):
    def policy(obs):
        result=incoming_policy(obs)
        vx,own,*_=decode(obs);ids=np.flatnonzero(vx>0)
        if len(ids):
            target=outgoing_target(obs[ids])
            result[ids]=np.argmin(np.abs(NEXT[own[ids]-15]-target[:,None])+[1e-6,0,1e-6],1)
        return result
    return policy

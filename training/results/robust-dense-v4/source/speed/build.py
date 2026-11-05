"""One analytic ordinary dense ReLU reciprocal-flight approximation; no fitting.

The network has exactly three fixed dense layers and two ReLU activations.
Six polynomial ridge pairs approximate a quadratic reciprocal on incoming speeds
5..13. Knots equidistribute square-root curvature of known polynomials; no game
states, actions, labels, optimizer or reward determine these constants.
"""
from pathlib import Path
import sys,json,importlib.util
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch,onnx
from torch import nn
from onnx import helper,numpy_helper
from pace_study import CONFIGS,make_policy,run_pair
from competition_eval import load_mlp
from policy import ROOT,load_builtin
OUT=Path('/tmp/rally-speed-dense')


def curvature_knots(poly,lower,upper,count=19):
    # Deterministic numerical integration of the analytic interpolation-error
    # density sqrt(|f''|), not fitting to sampled observations or labels.
    grid=np.linspace(lower,upper,100001)
    derivative=np.polyder(np.poly1d(poly),2)
    density=np.sqrt(abs(derivative(grid)))
    cumulative=np.r_[0,np.cumsum((density[1:]+density[:-1])*np.diff(grid)/2)]
    points=np.interp(np.linspace(0,cumulative[-1],count+2),cumulative,grid)
    values=np.polyval(poly,points)
    slopes=np.diff(values)/np.diff(points)
    return points,values,slopes,np.diff(slopes)


def make():
    n=125;net=nn.Sequential(nn.Linear(6,n),nn.ReLU(),nn.Linear(n,9),nn.ReLU(),nn.Linear(9,3)).eval()
    w1=np.zeros((n,6),np.float64);b1=np.zeros(n,np.float64);z=np.zeros(n,np.float64)
    # d=xnorm+11/12, v=(2/3)*obs_vy, w=−1.5*obs_vx−2.25.
    f0=[-3/130,0,9/16,0,0]
    fp=[3/260,-27/520,0,0,0]
    fm=[3/260,27/520,0,0,0]
    declarations=[('base',f0,-1,17/6,0),('plus',fp,-2,23/6,1),('minus',fm,-2,23/6,-1)]
    pos=0;linear_v=0.;records=[]
    for name,poly,lower,upper,wsign in declarations:
        points,values,slopes,coefficients=curvature_knots(poly,lower,upper)
        linear_v+=2*slopes[0]
        for vsign,output_sign in [(1,1),(-1,-1)]:
            ids=[]
            for knot,coefficient in zip(points[1:-1],coefficients):
                w1[pos,0]=1;w1[pos,3]=vsign*2/3;w1[pos,2]=-wsign*1.5
                b1[pos]=11/12-wsign*2.25-knot
                z[pos]=output_sign*coefficient;ids.append(pos);pos+=1
            records.append(dict(group=name,vertical_sign=vsign,output_sign=output_sign,units=ids,knots=points[1:-1].tolist(),slopes=slopes.tolist(),polynomial=poly,domain=[lower,upper]))
    assert pos==114
    # Raw signed carries. No skip connections exist in the exported graph.
    for feature in [1,3,4]:
        w1[pos,feature]=1;w1[pos+1,feature]=-1;pos+=2
    phase=pos;w1[pos,2]=1;pos+=1
    w1[pos,2]=-2;b1[pos]=1;w1[pos+1,2]=-2;pos+=2
    w1[pos,5]=10;b1[pos]=1;w1[pos+1,5]=10;b1[pos+1]=-1;pos+=2
    assert pos==125
    z[114]=1;z[115]=-1
    linear_vy=(linear_v+.05625)*2/3
    z[116]=linear_vy;z[117]=-linear_vy
    # Suppress the incoming-only unfolded feature for all legal outgoing
    # components, including extrapolated ridge arguments outside incoming range.
    lower=np.array([-11/12,-.95,5/6,-1.5,-.8125,-.8125])
    upper=np.array([11/12,.95,13/6,1.5,.8125,.8125])
    hmax=np.maximum(0,np.maximum(w1,0)@upper+np.minimum(w1,0)@lower+b1)
    bound=float(np.maximum(z,0)@hmax)
    phase_gate=float(2**np.ceil(np.log2(max(20,(bound+5)/(5/6)))))
    w2=np.zeros((9,n));b2=np.zeros(9)
    for i,knot in enumerate([-4.75,-2.85,-.95,.95,2.85]):
        w2[i]=z;w2[i,phase]-=phase_gate;b2[i]=-knot
    w2[5,121]=1;w2[5,122]=-1
    w2[6,118]=1;w2[7,119]=1
    w2[8,123]=1;w2[8,124]=-1;w2[8,phase]=-20
    aim=6/80
    down=np.r_[np.array([1,-2,2,-2,2]),-.95-aim,-1,1,aim]
    w3=np.stack([-down,np.zeros(9),down]);b3=np.array([0,1.1/80,0])
    with torch.no_grad():
        for layer,w,b in zip([net[0],net[2],net[4]],[w1,w2,w3],[b1,b2,b3]):
            layer.weight.copy_(torch.from_numpy(w.astype(np.float32)));layer.bias.copy_(torch.from_numpy(b.astype(np.float32)))
    description=dict(architecture=[6,125,9,3],parameters=sum(p.numel() for p in net.parameters()),macs=6*125+125*9+9*3,
        reciprocal='q(s)=(s*s−27*s+227)/585; interpolates1/s at5,9,13; maximum relative error4.21%',
        identity='Yn+9/4*d*v−81/65*d*v*w+36/65*d*v*w*w+.05625*v, d=xnorm+11/12,v=rawvy/9,w=(speed−9)/4',
        ridge_groups=records,linear_v=linear_v,linear_vy=linear_vy,phase_gate=phase_gate,outgoing_positive_preactivation_bound=bound,
        limitations=['No exact integer flight ceiling; uses continuous flight plus.5ticks','Polynomial reciprocal and PWL ridge approximation','Fixed6unit opponent-side aim','Not learned-source transfer; explicit analytic prior encoded only through ordinary dense weights'])
    return net,description,z


def export(net,path,description):
    nodes=[];weights=[];current='observation'
    for i,layer in enumerate(net):
        output='action_scores' if i==len(net)-1 else f'h{i}'
        if isinstance(layer,nn.Linear):
            weights.extend([numpy_helper.from_array(layer.weight.detach().numpy(),f'w{i}'),numpy_helper.from_array(layer.bias.detach().numpy(),f'b{i}')])
            nodes.append(helper.make_node('Gemm',[current,f'w{i}',f'b{i}'],[output],transB=1))
        else:nodes.append(helper.make_node('Relu',[current],[output]))
        current=output
    model=helper.make_model(helper.make_graph(nodes,'Analytic dense variable-speed approximation',[helper.make_tensor_value_info('observation',1,[1,6])],[helper.make_tensor_value_info('action_scores',1,[1,3])],weights),opset_imports=[helper.make_opsetid('',18)],ir_version=10)
    helper.set_model_props(model,dict(rules='pong-v4',architecture='6-125-9-3 ReLU',method='Analytic fixed dense polynomial-ridge initialization; no training',limitation='Approximate reciprocal/flight and PWL polynomial ridges; no exact trajectory program or learned-source transfer'))
    onnx.checker.check_model(model);onnx.save(model,path)


def numerical_check(net,description,z):
    x=np.tile(np.arange(15,346),9*19);speed=np.repeat(np.arange(5,14),331*19);vy=np.tile(np.repeat(np.arange(-9,10),331),9)
    # Component-valid probes are not all reachable. Also report shot-law range.
    obs=np.zeros((len(x),6),np.float32);obs[:,0]=(x-180)/180;obs[:,2]=-speed/6;obs[:,3]=vy/6
    with torch.no_grad():h1=torch.cat([net[:2](p) for p in torch.from_numpy(obs).split(4096)]).numpy()
    actual=h1@z.astype(np.float32)
    distance=x-15;fraction=distance*(speed*speed-27*speed+227)/585
    analytic=vy*(fraction+.5)/80
    exact_continuous=vy*(distance/speed+.5)/80
    exact_tick=vy*np.maximum(1,np.ceil(distance/speed))/80
    law_max=np.floor(9*speed/13+.5).astype(int);law_max[speed==5]=5
    compatible=(abs(vy)<=law_max)&((speed!=5)|(x<=180))
    report={'component_probes':len(x),'shot_range_probes':int(compatible.sum()),'all':{},'shot_range':{},'per_speed':{}}
    for label,mask in [('all',np.ones(len(x),bool)),('shot_range',compatible)]:
        report[label]={'pwl_error_units_max':float(np.max(abs(actual[mask]-analytic[mask]))*80),'pwl_error_units_mean':float(np.mean(abs(actual[mask]-analytic[mask]))*80),'total_continuous_error_units_max':float(np.max(abs(actual[mask]-exact_continuous[mask]))*80),'total_tick_error_units_mean':float(np.mean(abs(actual[mask]-exact_tick[mask]))*80),'total_tick_error_units_p95':float(np.quantile(abs(actual[mask]-exact_tick[mask]),.95)*80)}
    for s in range(5,14):
        mask=compatible&(speed==s);error=abs(actual[mask]-exact_tick[mask])*80
        report['per_speed'][str(s)]={'count':int(mask.sum()),'mean_units':float(error.mean()),'p95_units':float(np.quantile(error,.95)),'max_units':float(error.max())}
    return report


if __name__=='__main__':
    torch.set_num_threads(1);net,description,z=make();torch.save(net.state_dict(),OUT/'variable-speed.pt');export(net,OUT/'variable-speed.onnx',description)
    checks=numerical_check(net,description,z)
    (OUT/'construction.json').write_text(json.dumps(description,indent=2)+'\n');(OUT/'numerical.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('BUDGET',description['parameters'],description['macs'],'GATE',description['phase_gate'])
    print('NUMERIC',json.dumps(checks,indent=2),flush=True)
    seeds=json.loads(Path('/tmp/rally-causal-failures/seeds.json').read_text())['seeds'];rules=CONFIGS['pace13-v9-scaled']
    policies={name:load_builtin(name) for name in ['tracker','anticipator','challenger']}
    policies['pacer']=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt');policies.update({name:make_policy(rules,name) for name in ['attack','defend']})
    results=dict(purpose='One analytically fixed genuine dense variable-speed approximation; no fitting/training/sweep',seeds=seeds,architecture=description['architecture'],pairs={})
    for name,other in policies.items():
        row=run_pair(rules,net,other,seeds);results['pairs'][name]=row;(OUT/'results.json').write_text(json.dumps(results,separators=(',',':'))+'\n')
        s=row['summary'];print(name,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
    results['complete']=True;(OUT/'results.json').write_text(json.dumps(results,separators=(',',':'))+'\n')

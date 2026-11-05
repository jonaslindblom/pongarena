"""Target-regression distillation with a fixed linear paddle-position feedback.

The ONNX graph still takes only six observations and emits three action scores.
Its small feedback branch is disclosed; all predictive target weights are trained.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper
import torch
from torch import nn
from env import Arena,scripted
from policy import Actor,ROOT,actions
from reference import serve
from teacher_experiment import teacher,teacher_targets,target_teacher,evaluate_pair
from train_distill import make_actor


class TargetActor(nn.Module):
    def __init__(self,widths=(6,128,128,64,1),features=None):
        super().__init__()
        self.net=make_actor(widths)
        self.features=features
        self.widths=list(widths)
        feedback=torch.zeros(6,3);feedback[4,0]=1;feedback[4,2]=-1
        self.register_buffer('feedback',feedback)
        self.register_buffer('target_scores',torch.tensor([[-1.,0.,1.]]))
        self.register_buffer('deadband',torch.tensor([0.,1.001/80,0.]))

    def target(self,obs):
        return self.net(self.features(obs) if self.features is not None else obs)

    def forward(self,obs):
        return self.target(obs)@self.target_scores+obs@self.feedback+self.deadband


def export_target(actor,path):
    if actor.features is not None:
        from trajectory_features import export_with_features
        export_with_features(actor.features,actor.net,path,target_feedback=True,deadband=1.001/80)
        return
    nodes=[];weights=[];current='observation'
    for i,layer in enumerate(actor.net):
        output=f'h{i}'
        if isinstance(layer,nn.Linear):
            weights += [numpy_helper.from_array(layer.weight.detach().numpy(),f'w{i}'),numpy_helper.from_array(layer.bias.detach().numpy(),f'b{i}')]
            nodes.append(helper.make_node('Gemm',[current,f'w{i}',f'b{i}'],[output],transB=1))
        else:nodes.append(helper.make_node('Tanh',[current],[output]))
        current=output
    for name in ['feedback','target_scores','deadband']:
        weights.append(numpy_helper.from_array(getattr(actor,name).numpy(),name))
    nodes.extend([helper.make_node('MatMul',[current,'target_scores'],['target_action']),
                  helper.make_node('MatMul',['observation','feedback'],['feedback_action']),
                  helper.make_node('Add',['target_action','feedback_action'],['error_action']),
                  helper.make_node('Add',['error_action','deadband'],['action_scores'])])
    graph=helper.make_graph(nodes,'Distilled target with own-paddle feedback',
        [helper.make_tensor_value_info('observation',1,[1,6])],[helper.make_tensor_value_info('action_scores',1,[1,3])],weights)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)],ir_version=10)
    helper.set_model_props(model,{'rules':'pong-v3','architecture':'branched: target '+ '-'.join(map(str,actor.widths))+' Tanh + linear own-paddle feedback',
        'training':'PyTorch target-regression DAgger; fixed disclosed feedback branch'})
    onnx.checker.check_model(model);onnx.save(model,str(path))


def probes(seeds,old,max_offset):
    frames=[]
    for seed in seeds:
        f=None
        for _ in range(5):f=serve(int(seed),f);frames.append(f)
    env=Arena(len(frames));env.load(frames);rows=[];active=np.ones(len(frames),bool)
    for tick in range(1600):
        if not active.any():break
        obs=env.obs()
        if tick%8==0:rows.append(obs[active].copy())
        other=env.obs(1)
        b=np.where(np.arange(env.n)%2==0,scripted(other,True),actions(old,other))
        out,_=env.step(np.stack([target_teacher(obs,max_offset),b],1),active)
        active&=out<0
    data=np.concatenate(rows)
    data=data[np.random.default_rng(803).permutation(len(data))[:8192]]
    return data,teacher_targets(data,max_offset)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=81);p.add_argument('--rounds',type=int,default=24)
    p.add_argument('--out',type=Path,default=ROOT/'training/runs/target-81');p.add_argument('--envs',type=int,default=256)
    p.add_argument('--steps',type=int,default=256);p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--contact-weight',type=float,default=3)
    p.add_argument('--classification-weight',type=float,default=0)
    p.add_argument('--max-offset',type=int,default=15,choices=[13,14,15])
    p.add_argument('--features',action='store_true');args=p.parse_args()
    if (args.out/'metrics.jsonl').exists():raise FileExistsError('Use a fresh output directory')
    args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    if args.features:
        from trajectory_features import TrajectoryFeatures
        actor=TargetActor((8,128,128,64,1),features=TrajectoryFeatures())
    else:actor=TargetActor()
    optimizer=torch.optim.Adam(actor.parameters(),lr=.001)
    old=Actor();old.load_state_dict(torch.load(ROOT/'training/results/v2/ppo.pt',weights_only=True));old.eval()
    validation=np.random.default_rng(1459).choice(100_000_000,32,replace=False)+1_200_000_000
    probe_x,probe_y=probes(validation,old,args.max_offset)
    np.testing.assert_array_equal(target_teacher(probe_x,args.max_offset),np.where(probe_y-np.rint(80+80*probe_x[:,4]) < -1,0,np.where(probe_y-np.rint(80+80*probe_x[:,4])>1,2,1)))
    probe_x_t=torch.from_numpy(probe_x)
    env=Arena(args.envs,args.seed,loop_detection=False);opponents=rng.integers(0,4,args.envs)
    config=dict(**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},rules='pong-v3',
        architecture='branched target MLP plus own-paddle feedback',target_widths=actor.widths,activation='Tanh',
        validation_seeds=validation.tolist(),training_loop_detection=False,teacher=f'minimax recovery; incoming offset <= {args.max_offset}',
        selection='minimum (wins+half draws) against attacker and v2; tie mean',
        loss=f'weighted normalized target MSE + {args.classification_weight} cross entropy on scores*20',deadband_physical=1.001,
        random_action_probability=.01)
    (args.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    started=time.monotonic();best=(-1,-1);buffers=[]
    def opposition(obs):
        out=np.empty(env.n,int)
        for code in range(4):
            mask=opponents==code
            if mask.any():out[mask]=scripted(obs[mask],True) if code==0 else actions(old,obs[mask]) if code==1 else scripted(obs[mask]) if code==2 else teacher(obs[mask])
        return out
    for iteration in range(1,args.rounds+1):
        inputs=[];targets=[];weights=[];labels=[];beta=max(0.,1-(iteration-1)/8)
        for _ in range(args.steps):
            obs=env.obs();target=teacher_targets(obs,args.max_offset);expert=target_teacher(obs,args.max_offset)
            chosen=np.where(rng.random(env.n)<beta,expert,actions(actor,obs))
            chosen=np.where(rng.random(env.n)<.01,rng.integers(0,3,env.n),chosen)
            flight=np.ceil((180+180*obs[:,0]-15)/np.maximum(1,-6*obs[:,2]))
            inputs.append(obs);targets.append(((target-80)/80).astype(np.float32));labels.append(expert)
            weights.append(np.where((obs[:,2]<0)&(flight<=12),args.contact_weight,1).astype(np.float32))
            out,_=env.step(np.stack([chosen,opposition(env.obs(1))],1));done=np.flatnonzero(out>=0);env.reset(done);opponents[done]=rng.integers(0,4,len(done))
        buffers.append(tuple(np.concatenate(v) for v in [inputs,targets,weights,labels]));buffers=buffers[-4:]
        x,y,w,label=[torch.from_numpy(np.concatenate([b[i] for b in buffers])) for i in range(4)]
        with torch.no_grad():
            encoded=torch.cat([actor.features(part) for part in x.split(4096)]) if actor.features is not None else x
        optimizer.param_groups[0]['lr']=.001*(1-.7*(iteration-1)/max(1,args.rounds-1))
        for _ in range(args.epochs):
            for ix in torch.randperm(len(x)).split(2048):
                prediction=actor.net(encoded[ix])
                error=prediction.squeeze(-1)-y[ix]
                scores=prediction@actor.target_scores+x[ix]@actor.feedback+actor.deadband
                per=error.square()+args.classification_weight*nn.functional.cross_entropy(scores*20,label[ix],reduction='none')
                loss=(per*w[ix]).sum()/w[ix].sum();optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step()
        evals={name:evaluate_pair(lambda o:actions(actor,o),op,validation) for name,op in [('attacker',lambda o:scripted(o,True)),('v2',lambda o:actions(old,o))]}
        scores=[(v['wins']+.5*v['draws'])/v['rallies'] for v in evals.values()];key=(min(scores),float(np.mean(scores)))
        with torch.no_grad():pred=actor.target(probe_x_t).squeeze(-1).numpy()*80+80
        error=np.abs(pred-probe_y);flight=np.ceil((180+180*probe_x[:,0]-15)/np.maximum(1,-6*probe_x[:,2]));near=(probe_x[:,2]<0)&(flight<=12)
        row=dict(round=iteration,env_steps=iteration*args.steps*env.n,seconds=time.monotonic()-started,loss=float(loss.detach()),
            selection_min=key[0],selection_mean=key[1],teacher_probe_mae=float(error.mean()),near_contact_mae=float(error[near].mean()),
            teacher_probe_action_agreement=float((actions(actor,probe_x)==target_teacher(probe_x,args.max_offset)).mean()),opponents=evals)
        with (args.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        torch.save(actor.state_dict(),args.out/'last.pt')
        if key>best:
            best=key;torch.save(actor.state_dict(),args.out/'best.pt');export_target(actor,args.out/'best.onnx')
            (args.out/'selection.json').write_text(json.dumps(row,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    print('Finished',args.out,'best',best,flush=True)


if __name__=='__main__':main()

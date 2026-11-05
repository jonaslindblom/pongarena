from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,onnx
from onnx import helper
from compact_handcrafted_v4 import Graph
from pace_study import CONFIGS,tables
from policy import OnnxPolicy

class G(Graph):
 def __init__(self):
  super().__init__();self.shapes={'observation':(1,6)}
 def constant(self,value):
  name=super().constant(value);self.shapes[name]=np.asarray(value).shape;return name
 def op(self,kind,*inputs,**attrs):
  name=super().op(kind,*inputs,**attrs);a=self.shapes[inputs[0]]
  if kind=='Add':shape=np.broadcast_shapes(a,self.shapes[inputs[1]])
  elif kind in ('Gemm','MatMul'):
   b=self.shapes[inputs[1]]
   if attrs.get('transA'):a=a[::-1]
   if attrs.get('transB'):b=b[::-1]
   shape=(a[0],b[1])
  else:shape=a
  self.shapes[name]=shape;return name
 def neg(self,x,n=1):return self.affine(x,np.eye(n),1,alpha=-1.,beta=0.)
 def sub(self,a,b,n=1):
  ash,bsh=self.shapes[a],self.shapes[b];out=np.broadcast_shapes(ash,bsh)
  if ash[0]==out[0]:x,c,alpha,beta=a,b,1.,-1.
  else:x,c,alpha,beta=b,a,-1.,1.
  cols=self.shapes[x][1];weight=np.eye(cols) if cols==out[1] else np.ones((1,out[1]))
  return self.op('Gemm',x,self.constant(weight),c,alpha=alpha,beta=beta)
 def maximum(self,a,b,n=1):return self.op('Add',b,self.relu(self.sub(a,b,n)))
 def minimum(self,a,b,n=1):return self.sub(a,self.relu(self.sub(a,b,n)),n)
 def absolute(self,a,n=1):return self.op('Gemm',self.relu(a),self.constant(np.eye(n)),a,alpha=2.,beta=-1.)
 def clip(self,x,lo,hi,n=1):return self.sub(self.op('Add',lo,self.relu(self.sub(x,lo,n))),self.relu(self.sub(x,hi,n)),n)
 def col(self,row):return self.affine(row,[[1]],1,alpha=1.,beta=0.,transA=1)
 def pool(self,x,rows):
  while rows>1:
   n=(rows+1)//2;diff=np.zeros((n,rows));odd=np.zeros_like(diff)
   for i in range(n):
    a=2*i;b=min(a+1,rows-1);diff[i,a]+=1;diff[i,b]-=1;odd[i,b]=1
   x=self.op('Add',self.relu(self.product(self.constant(diff),x)),self.product(self.constant(odd),x));rows=n
  return x
 def fold(self,x):
  h=self.relu(self.add(x,[604,452,300,148,-4,-156,-308,-460,-612]))
  return self.affine(h,np.array([1,-2,2,-2,2,-2,2,-2,2])[:,None],4)

def trajectory(g):
 vx=g.project(2,6);speed=g.op('Add',g.relu(vx),g.relu(g.neg(vx)))
 sign=g.scalar(g.sub(g.relu(g.add(vx,[1])),g.relu(g.add(vx,[-1]))),1,-1)
 basis=g.relu(g.add(speed,-np.arange(5,13)));slopes=np.diff(1/np.arange(5,14,dtype=float))
 inv=g.affine(basis,np.r_[slopes[0],np.diff(slopes)][:,None],1/5)
 d=g.scalar(g.product(sign,g.project(0,180)),-1,165);ft=g.product(d,inv)
 low=g.relu(g.add(ft,-np.arange(1,66)-1/64));hi=g.relu(g.add(ft,-np.arange(1,66)-1/32))
 ticks=g.affine(g.op('Add',low,g.product(g.constant([[-1]]),hi)),np.ones((65,1)),1,alpha=64.)
 height=g.fold(g.op('Add',g.project(1,80,80),g.product(ticks,g.project(3,6))))
 return height,ticks,speed,basis,sign

def midpoint(g,h,basis,cap=False):
 tab=tables(CONFIGS['pace13-v9-scaled'])
 if cap:return g.curve(h,np.arange(4,157),tab['mid'][13,4:])
 pieces=[]
 for stage in range(5,13):
  co=np.zeros(8)
  if stage==5:co[:2]=[-1,1]
  elif stage==12:co[-2:]=[1,-1]
  else:co[stage-6:stage-3]=[1,-2,1]
  gate=g.affine(basis,co[:,None],int(stage==5));curve=g.curve(h,np.arange(4,157),tab['mid'][stage,4:]);pieces.append(g.product(gate,curve))
 return g.sum(pieces)

def export(path,forecast='endpoint',cap=False):
 g=G();h,t,s,b,sign=trajectory(g);own=g.project(4,80,80);other=g.project(5,80,80)
 rec=midpoint(g,h,b,cap);speedheights=np.arange(5,14)
 ig=g.scalar(sign,-.5,.5);og=g.scalar(sign,.5,.5)
 actor=g.op('Add',g.product(ig,own),g.product(og,other));receiver=g.op('Add',g.product(ig,other),g.product(og,own))
 # Offset bands for each legal outgoing integer vertical velocity.
 scale=g.curve(s,speedheights,65/(3*np.minimum(13,speedheights+1)))
 flight=g.curve(s,speedheights,np.ceil(330/np.minimum(13,speedheights+1)))
 k=np.arange(-9,10)[:,None]
 lo=g.maximum(g.maximum(g.add(h,[-15]),g.constant([[15]])),g.sub(h,g.affine(scale,(k+.5).T,0))) if False else None
 # Produce candidate columns from scalar times each velocity bound.
 dlo=g.product(g.constant(k+.5),scale);dhi=g.product(g.constant(k-.5),scale)
 lo=g.maximum(g.maximum(g.sub(h,dlo),g.add(h,[-15])),g.constant([[15]]))
 hi=g.minimum(g.minimum(g.sub(h,dhi),g.add(h,[15])),g.constant([[145]]))
 # Keep the ordinary movement parity. Wall-clamp parity changes are a
 # conservative omission in this compact candidate.
 fp=g.affine(g.op('Tanh',g.add(g.scalar(actor,4096),-8192*(np.arange(8,73)-1/512))),np.ones((65,1)),39.5,alpha=.5)
 parity=g.sub(actor,g.scalar(fp,2))
 z=g.scalar(g.sub(lo,parity),4096)
 q=g.affine(g.op('Tanh',g.add(z,-8192*(np.arange(7,73)+1/512))),np.ones((66,1)),40,alpha=.5)
 lo=g.op('Add',parity,g.scalar(q,2))
 dest=g.fold(g.op('Add',h,g.product(g.constant(k),flight)))
 radius=g.scalar(t,2,-2)
 expected=g.clip(rec,g.sub(receiver,radius),g.op('Add',receiver,radius))
 nextp=g.clip(g.add(actor,[-2,0,2]),g.constant([[15,15,15]]),g.constant([[145,145,145]]),3)
 lower=g.sub(g.sub(lo,nextp,3),radius)
 upper=g.sub(g.sub(nextp,hi),radius)
 gap=g.relu(g.maximum(g.maximum(lower,upper,3),g.sub(lo,hi)))
 immediate=g.maximum(g.relu(g.sub(lo,nextp,3)),g.relu(g.sub(nextp,hi)),3)
 margin=g.absolute(g.sub(dest,expected))
 quality=g.op('Gemm',gap,g.constant(np.eye(3)),margin,alpha=-1000.,beta=1.);quality=g.op('Gemm',immediate,g.constant(np.eye(3)),quality,alpha=-.001,beta=1.)
 tab=tables(CONFIGS['pace13-v9-scaled'])
 safe_lo=g.curve(dest,np.arange(4,157),tab['low'][13,4:]);safe_hi=g.curve(dest,np.arange(4,157),tab['high'][13,4:])
 slack=g.scalar(g.sub(safe_lo,safe_hi),.5)
 reachf=g.scalar(flight,2)
 risk=g.relu(g.maximum(g.maximum(slack,g.sub(g.sub(safe_lo,hi),reachf)),g.sub(g.sub(lo,reachf),safe_hi)))
 quality=g.op('Gemm',quality,g.constant(np.eye(3)),risk,alpha=1.,beta=-.05)
 attack=g.pool(quality,19)
 attack=g.add(attack,[-1e-6,0,-1e-6])
 # While outgoing, forecast which reachable endpoint is farther from the
 # recovery point that the opposing teacher expects us to approach.
 reach=g.scalar(t,2)
 qgap=g.relu(g.maximum(g.maximum(g.sub(g.sub(lo,actor),reach),g.sub(g.sub(actor,hi),reach)),g.sub(lo,hi)))
 penalty=g.scalar(g.relu(g.add(qgap,[-1/256])),32768)
 top=g.pool(g.sub(dest,penalty),19);bottom=g.neg(g.pool(g.sub(g.neg(dest),penalty),19))
 expected_own=g.clip(rec,g.sub(receiver,radius),g.op('Add',receiver,radius))
 diff=g.sub(g.op('Add',top,bottom),g.scalar(expected_own,2))
 gate=g.scalar(g.relu(diff),64);gate=g.sub(gate,g.relu(g.add(gate,[-1])))
 target=g.op('Add',bottom,g.product(gate,g.sub(top,bottom)))
 err=g.sub(target,own);defense=g.affine(err,[[-1,0,1]],[0,1.1,0])
 # Only the selected direction's score scale matters; score multiplication is
 # scalar-on-left to broadcast over three action scores.
 final=g.op('Add',g.product(ig,attack),g.product(og,defense));g.nodes[-1].output[0]='action_scores'
 model=helper.make_model(helper.make_graph(g.nodes,'Compact predictive attacker',
 [helper.make_tensor_value_info('observation',1,[1,6])],[helper.make_tensor_value_info('action_scores',1,[1,3])],g.weights),opset_imports=[helper.make_opsetid('',18)],ir_version=10)
 helper.set_model_props(model,dict(rules='pong-v4',method='Handcrafted19-angle attack and predictive recovery; no training',limitation='Real-valued future contact bands; endpoint forecast; no exact long-horizon parity'))
 onnx.checker.check_model(model);onnx.save(model,path)
 return {'params':sum(np.prod(w.dims).item() for w in model.graph.initializer),'nodes':len(g.nodes)}
if __name__=='__main__':
 p=Path('/tmp/rally-counter-root/counter.onnx');print(export(p,cap=True));session=OnnxPolicy(p).session;print(session.run(None,{'observation':np.zeros((1,6),np.float32)}))

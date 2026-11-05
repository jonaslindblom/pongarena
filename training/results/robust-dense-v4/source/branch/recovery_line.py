from sys import path; path.insert(0,str(__import__("pathlib").Path.cwd()/"training/scripts")); from env import reflect

exec(open('/tmp/rally-safe-branch/analyze.py').read().split('report=dict')[0])
L=T['low'][13,H];U=T['high'][13,H];good=L<=U
print('nonempty',good.sum(),'lowhigh',[(int(y),int(l),int(u))for y,l,u in zip(H,L,U)if y%4==0])
x=H-80;lo=np.full(len(x),-np.inf);hi=np.full(len(x),np.inf)
lo[x>0]=(80-U[x>0])/x[x>0];hi[x>0]=(80-L[x>0])/x[x>0]
lo[x<0]=(80-L[x<0])/x[x<0];hi[x<0]=(80-U[x<0])/x[x<0]
print('slopeinterval',float(lo[good].max()),float(hi[good].min()))
for k in [0,.25,.5]:
 p=80-k*x;miss=np.maximum(L-p,p-U)
 print(k,'unsafe',int((miss>0).sum()),'nonemptyunsafe',int(((miss>0)&good).sum()),'max',float(miss.max()))
t=H+11.5+np.clip(H-50,0,2)-27*np.clip((H-78)/4,0,1)+np.clip(H-108,0,2)
possible=(abs(C[None]-t[:,None])<=1.101)&(R<=0)
y=np.broadcast_to(H[:,None],possible.shape)[possible];c=np.broadcast_to(C[None,:],possible.shape)[possible];vv=np.floor(.6*(y-c)+.5);dest=np.unique(reflect(y+26*vv).astype(int));print('branchsafe destinations',dest.tolist())
mask=np.isin(H,dest);g=mask&good;print('branch slopeinterval',float(lo[g].max()),float(hi[g].min()))
for k in [0,.25,.125]:
 p=80-k*x;miss=np.maximum(L-p,p-U);print('branch',k,'unsafe',int(((miss>0)&g).sum()),[(int(y),float(m))for y,m in zip(H[g],miss[g])if m>0])

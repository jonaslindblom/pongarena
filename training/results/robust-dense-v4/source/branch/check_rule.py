exec(open('/tmp/rally-safe-branch/analyze.py').read().split('report=dict')[0])
def worst(t):
 possible=(abs(C[None]-t[:,None])<=1.101)
 # Boundaries always clamp tovalidcenters; however tgeneratedinsidelegaldomain.
 costs=np.where(possible,R,-np.inf);return costs.max(1),costs
for off in [9.5,10,10.5,11.5]:
 t=np.clip(H+off,15,145);v,_=worst(t)
 print('offset',off,[(int(y),float(x))for y,x in zip(H[:14],v[:14])])
# One declaredrule:lowoffset11.5→13.5between50and52;centercrossover78→82.
# Keepedgebranchsameasinterior, useheight4clamp15.
t=H+11.5+np.clip(H-50,0,2)-27*np.clip((H-78)/4,0,1)+np.clip(H-108,0,2)
v,_=worst(t)
print('full',int((v>0).sum()),float(v.mean()),float(v.max()),[(int(y),float(p),float(r))for y,p,r in zip(H,t,v)if r>0])
print('knots',[4,50,52,78,82,108,110,156])
print('values',(np.array([4,50,52,78,82,108,110,156])+11.5+np.clip(np.array([4,50,52,78,82,108,110,156])-50,0,2)-27*np.clip((np.array([4,50,52,78,82,108,110,156])-78)/4,0,1)+np.clip(np.array([4,50,52,78,82,108,110,156])-108,0,2)).tolist())

import sys,numpy as np
sys.path.insert(0,str(__import__('pathlib').Path.cwd()/'training/scripts'))
from env import reflect
h=np.array([4,50,52,78,82,108,110,156.]);v=np.array([15.5,61.5,65.5,91.5,68.5,94.5,98.5,144.5])
raw=[]
for n in range(-3,4):
 for a in h:raw.extend([a+304*n,312-a+304*n])
raw=np.unique(np.r_[-240,raw,400]);raw=raw[(raw>=-240)&(raw<=400)]
val=np.interp(reflect(raw),h,v);sl=np.diff(val)/np.diff(raw);co=np.r_[sl[0],np.diff(sl)];use=abs(co)>1e-9
print(raw.tolist());print(val.tolist());print('count',len(raw),use.sum(),'width2',use.sum()+2,'params',7*42+46*(use.sum()+2)+3)
print('withoutfirstaffine',7*40+44*(use.sum()+2)+3,'plus12',7*40+44*(use.sum()+14)+3)

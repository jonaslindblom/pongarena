exec(open('/tmp/rally-safe-branch/analyze.py').read().split('report=dict')[0])
M=np.maximum(R[:,:-1],R[:,1:]);m=M.min(1)
for i,y in enumerate(H):
 if y<=18 or 77<=y<=83 or y>=142:
  ix=np.flatnonzero(M[i]==m[i]);print(y,m[i],(C[:-1][ix]+.5).tolist())

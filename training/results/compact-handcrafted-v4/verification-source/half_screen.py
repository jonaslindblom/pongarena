exec(open('/tmp/rally-compact-targets/screen.py').read().split('result={')[0])
def half(obs):
 vx,own,_,_,h=decode(obs)
 r=h-80;a=abs(r)
 upper=incoming[13,80+a]-80
 # Exact signed reflection, with center assigned the lower-half convention.
 sign=np.where(r>0,1,-1)
 target=np.where(vx<0,80+sign*upper,80)
 return np.where(target<own-1.1,0,np.where(target>own+1.1,2,1))
result={'purpose':'One Nano target compression: upper-half cap13 table mirrored with sign, outgoingcenter80; exact scalar landing','seeds':seeds,'pairs':{}}
for name,op in opponents.items():
 row=run_pair(rules,half,op,seeds);result['pairs'][name]=row
 Path('/tmp/rally-compact-targets/half-results.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
 s=row['summary'];print(name,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)

"""Requested decomposition: retain CraftSmall incoming; replace recovery only."""
from screen_endpoint import *
centers=T['centers']
quality=-T['risk']-.0001*np.abs(centers-np.arange(157)[:,None])[None]
best=np.argmax(np.where(T['legal'][None],quality,-1e12),axis=2)
small_targets=np.take_along_axis(np.broadcast_to(centers,T['risk'].shape),best[:,:,None],axis=2)[:,:,0]

def mixed(mode):
    outgoing=recovery_policy(mode,'attack')
    def policy(obs):
        result=outgoing(obs)
        vx,own,other,ticks,height=decode(obs)
        mask=vx<0
        target=small_targets[np.abs(vx[mask]),height[mask]]
        result[mask]=np.where(target<own[mask]-1.1,0,np.where(target>own[mask]+1.1,2,1))
        return result
    return policy

def main():
    torch.set_num_threads(1)
    seeds=(np.random.default_rng(2026110511).choice(50_000_000,24,replace=False)+4_100_000_000).tolist()
    opponents={'attack':make_policy(RULES,'attack'),'defend':make_policy(RULES,'defend'),
      'pacer':load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt'),
      'craft_small':OnnxPolicy(ROOT/'training/results/compact-handcrafted-v4/handcrafted-small-v4.onnx')}
    candidates={'small_predicted_return':mixed('predicted_attack'),'small_predicted_endpoint':mixed('predicted_endpoint')}
    result=dict(method='Requested design decomposition; CraftSmall incoming unchanged; forecast outgoing recovery only; no fitting',seeds=seeds,pairs={})
    for name,candidate in candidates.items():
        for opponent_name,opponent in opponents.items():
            row=run_pair(RULES,candidate,opponent,seeds)
            result['pairs'][name+':'+opponent_name]=row
            (OUT/'mixed-screen.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
            s=row['summary'];print(name,opponent_name,s['wins'],s['losses'],s['draws'],'CI',s['point_margin_ci95'],'paired',s['paired_match_wins'],s['paired_match_losses'],s['paired_match_draws'],flush=True)
    result['complete']=True
    (OUT/'mixed-screen.json').write_text(json.dumps(result,separators=(',',':'))+'\n')

if __name__=='__main__':main()

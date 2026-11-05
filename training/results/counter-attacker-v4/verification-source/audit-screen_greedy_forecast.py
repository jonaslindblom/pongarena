"""Requested single compact greedy-forecast composition; no search or fitting."""
from screen_band_mixed import band,OUT,ROOT,RULES,make_policy,load_mlp,OnnxPolicy,run_pair,np,torch,json
from greedy_forecast import compose,outgoing_target
import hashlib

def main():
    torch.set_num_threads(1)
    seeds=(np.random.default_rng(2026110511).choice(50_000_000,24,replace=False)+4_100_000_000).tolist()
    candidate=compose(band.band_policy)
    opponents={'attack':make_policy(RULES,'attack'),'defend':make_policy(RULES,'defend'),
      'pacer':load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt'),
      'craft_small':OnnxPolicy(ROOT/'training/results/compact-handcrafted-v4/handcrafted-small-v4.onnx')}
    result=dict(method='One requested compact forecast: real bands, cap13 safety, teacher coefficients, exact max/tie extraction; band incoming unchanged',seeds=seeds,
      forecast_source_sha256=hashlib.sha256((OUT/'greedy_forecast.py').read_bytes()).hexdigest(),pairs={})
    for name,op in opponents.items():
        row=run_pair(RULES,candidate,op,seeds);result['pairs'][name]=row
        (OUT/'greedy-forecast-screen.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
        s=row['summary'];print(name,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],'paired',s['paired_match_wins'],s['paired_match_losses'],s['paired_match_draws'],flush=True)
    result['complete']=True;(OUT/'greedy-forecast-screen.json').write_text(json.dumps(result,separators=(',',':'))+'\n')

if __name__=='__main__':main()

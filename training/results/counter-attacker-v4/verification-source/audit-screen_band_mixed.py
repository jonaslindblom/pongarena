"""Essential decomposition: fixed 19-vy-band incoming + forecast recovery."""
from screen_endpoint import *
import hashlib,importlib.util,shutil
SOURCE=Path('/tmp/rally-attacker-targets/band_policy.py')
shutil.copyfile(SOURCE,OUT/'band_policy_source.py')
spec=importlib.util.spec_from_file_location('band_policy_fixed',OUT/'band_policy_source.py')
band=importlib.util.module_from_spec(spec);spec.loader.exec_module(band)

def mixed(mode):
    outgoing=recovery_policy(mode,'attack')
    def policy(obs):
        result=outgoing(obs)
        vx=decode(obs)[0];mask=vx<0
        result[mask]=band.band_policy(obs)[mask]
        return result
    return policy

def main():
    torch.set_num_threads(1)
    seeds=(np.random.default_rng(2026110511).choice(50_000_000,24,replace=False)+4_100_000_000).tolist()
    opponents={'attack':make_policy(RULES,'attack'),'defend':make_policy(RULES,'defend'),
      'pacer':load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt'),
      'craft_small':OnnxPolicy(ROOT/'training/results/compact-handcrafted-v4/handcrafted-small-v4.onnx')}
    candidates={'band_predicted_return':mixed('predicted_attack'),'band_predicted_endpoint':mixed('predicted_endpoint')}
    result=dict(method='Requested design decomposition; fixed real-band incoming unchanged; forecast outgoing recovery only; no fitting',seeds=seeds,
       band_source_sha256=hashlib.sha256((OUT/'band_policy_source.py').read_bytes()).hexdigest(),pairs={})
    for name,candidate in candidates.items():
        for opponent_name,opponent in opponents.items():
            row=run_pair(RULES,candidate,opponent,seeds)
            result['pairs'][name+':'+opponent_name]=row
            (OUT/'band-mixed-screen.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
            s=row['summary'];print(name,opponent_name,s['wins'],s['losses'],s['draws'],'CI',s['point_margin_ci95'],'paired',s['paired_match_wins'],s['paired_match_losses'],s['paired_match_draws'],flush=True)
    result['complete']=True
    (OUT/'band-mixed-screen.json').write_text(json.dumps(result,separators=(',',':'))+'\n')

if __name__=='__main__':main()

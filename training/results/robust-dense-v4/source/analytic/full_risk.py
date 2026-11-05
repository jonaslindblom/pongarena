"""One Standard-size control retaining the complete physical target curve."""
from pathlib import Path
import sys,json
sys.path.insert(0,'/tmp/rally-analytic-dense')
from risk_prior import make_risk
from build import export,OUT
import torch,numpy as np,onnx
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
from competition_eval import load_mlp
from pace_study import CONFIGS,make_policy,run_pair
from handcrafted_v4 import target_tables
from env import reflect
from policy import ROOT,load_builtin

if __name__=='__main__':
 torch.set_num_threads(1);incoming,_=target_tables();raw=np.arange(-240,401);values=incoming[13,reflect(raw).astype(int)].astype(float)
 prior=dict(purpose='Complete physical cap13targetcurve on everyinteger rawforecastcoordinate; no observedstate/reward fitting',raw_knots=raw.tolist(),target_values=values.tolist(),forecast_limitation='NN cap13 half-tickforecast iscontinuous and approximate; quantized optimalcontacttargets are sensitive to small landing errors')
 (OUT/'full-risk-feasibility.json').write_text(json.dumps(prior,indent=2)+'\n')
 net=make_risk('full-risk-feasibility.json');torch.save(net.state_dict(),OUT/'standard-full-risk.pt');export(net,OUT/'standard-full-risk.onnx')
 width=net[2].out_features;params=sum(x.numel() for x in net.parameters());macs=6*27+27*width+width*3
 assert params<=17795 and macs<=17536
 p=OUT/'standard-full-risk.onnx';m=onnx.load(p);onnx.helper.set_model_props(m,dict(rules='pong-v4',architecture=f'6-27-{width}-3 ReLU',method='Transparent analytic dense fullresolutioncap13physicalcontactprior; no training',target='ExactPWLextensionofcap13minriskcontacttableatallintegerrawforecastcoordinates;outgoingcenter80',limitation=prior['forecast_limitation']));onnx.save(m,p)
 seeds=json.loads(Path('/tmp/rally-causal-failures/seeds.json').read_text())['seeds'];opponents={k:load_builtin(k) for k in ['tracker','anticipator','challenger']};opponents['pacer']=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt');opponents.update({k:make_policy(CONFIGS['pace13-v9-scaled'],k) for k in ['attack','defend']})
 result=dict(purpose='One Standard budget completephysicaltargetcurve control; no fitting/sweeps',architecture=[6,27,width,3],parameters=params,macs=macs,seeds=seeds,pairs={})
 for name,other in opponents.items():
  row=run_pair(CONFIGS['pace13-v9-scaled'],net,other,seeds);result['pairs'][name]=row
  (OUT/'full-risk-results.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
  s=row['summary'];print(name,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
 print('Done',result['architecture'],params,macs)

from pathlib import Path
import torch,json,psutil,numpy as np
from v9_policy import Controller,CONTEXT_FIELDS
from gate_ppo import terms,update
R=Path(__file__).resolve().parent.parent
psutil.Process().cpu_affinity([6,7]);torch.set_num_threads(1)
gate=Controller().cuda();gate.load_state_dict(torch.load(R/'checkpoints/initial.pt',weights_only=False));rows=[]
for n in (50,150,250):rows.extend(torch.load(R/f'smoke/N{n}/session_rollout.pt',weights_only=False)['rows'])
# Recompute the new normalization from recorded scores on the same physical trajectories.
for n in (50,150,250):
 result=json.loads((R/f'smoke/N{n}/results/gate_trained.json').read_text());lookup={(r['batch'],r['direction'],r['step']-1):r for r in result['records']}
 for row in rows:
  if row['n']!=n:continue
  score=lookup[(row['batch'],row['expert'],row['local_step'])]['best_before'];factor=max(abs(score),1e-6)/(1+abs(score));row['session_return']*=factor
with torch.no_grad():lr,_=terms(gate,rows,'cuda');error=float(lr.abs().max());assert error<1e-4,error
critic=torch.nn.Sequential(torch.nn.Linear(len(CONTEXT_FIELDS),32),torch.nn.Tanh(),torch.nn.Linear(32,1)).cuda();opt=torch.optim.Adam(list(gate.parameters())+list(critic.parameters()),lr=3e-4)
stat=update(gate,critic,opt,rows);stat.update(passed=True,real_rollout_logratio_error=error,diagnostic_only=True)
(R/'PPO_SMOKE.json').write_text(json.dumps(stat,indent=2));print(json.dumps(stat),flush=True)

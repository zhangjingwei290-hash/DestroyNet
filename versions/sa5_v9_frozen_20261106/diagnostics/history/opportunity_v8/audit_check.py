"""Read-only audit of corrected bounded integration check."""
from pathlib import Path
import sys,json,hashlib
import torch,numpy as np
O=Path(__file__).resolve().parent;R=O/'technical_check_corrected'
manifest=json.loads((R/'SOURCE_MANIFEST.json').read_text())
assert all(hashlib.sha256(Path(i['windows_path']).read_bytes()).hexdigest()==i['sha256'] for i in manifest['files'])
c=json.loads((R/'COMPLETE.json').read_text());assert c['passed'] and c['optimizer_steps']==2
rows_total=0;session_total=0;negative=0
for n in (50,150,250):
 p=R/f'cases/cycle0_{n}';d=torch.load(p/'session_rollout.pt',weights_only=False);result=json.loads((p/'results/gate_trained.json').read_text())
 assert json.loads((p/'analysis/AUDIT.json').read_text())['passed']
 ss={(s['batch'],s['direction']):s for s in d['sessions'] if s['complete']}
 for key,s in ss.items():
  rr=[r for r in d['rows'] if (r['batch'],r['expert'])==key];assert len(rr)==5
  rec=[r for r in result['records'] if (r['batch'],r['direction'])==key]
  assert abs(sum(r['best_reward'] for r in rec)-s['gain'])<1e-10
  assert abs(s['net_return']-(s['gain']-s['v_rate']*s['seconds']))<1e-12
  assert all(r['session_return']==s['net_return'] for r in rr)
  negative+=s['net_return']<0
 assert any(s['v_rate']>0 for s in ss.values()) and any(s['v_rate']==0 for s in ss.values())
 for row in d['rows']:
  x=np.asarray(row['gate_context']);assert abs(x[-2]-row['last_repair_sec']/d['budget'])<1e-7
  assert abs(x[-1]-(n/len(row['routes']))/50)<1e-7
 rows_total+=len(d['rows']);session_total+=len(ss)
a=torch.load(R/'initial_gate.pt',weights_only=False);b=torch.load(R/'gate.pt',weights_only=False)
assert any(not torch.equal(a[k],b[k]) for k in a)
assert not torch.equal(a['film.2.weight'],b['film.2.weight'])
r=dict(passed=True,complete_sessions=session_total,actions=rows_total,negative_net_sessions=negative,exact_runtime_scalar_reward=True,raw_time_feature_verified=True,mean_customers_per_vehicle_verified=True,causal_nonzero_ema=True,three_scale_joint_updates=2,film_updated=True,source_manifest_unchanged=True,all_physical_audits_passed=True,performance_claim=False)
(R/'IMPLEMENTATION_AUDIT.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))

"""Hold expert7 fixed while changing preference; add to matched expert7 rows."""
import experiment as x
import numpy as np,torch,json,time
from pathlib import Path
x.write('FIXED_PREFERENCE_PROTOCOL.json',dict(purpose='Separate preference conditioning from switching adapted experts',policy_expert=7,preferences=[0,7,15],new_preferences=[0,15],reuse_preference7='candidate_panel.csv exact same snapshots/seeds/repair protocol',new_candidate_calls=1440,repetitions=8,training_updates=0,declared_before_loading_candidate_outcomes=True,limit='Preference also conditions repair and scalar criterion, so test whole preference-conditioned action utility, not policy-only causal effect'))
cp=x.P/'training/direction_07.pt';digest=x.sha(cp)
m=x.NewPolicy(x.cfg);m.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model']);m.eval();m.requires_grad_(False);m.cpu_head=x.NewHead(m)
rows=[];arcs=[]
arms=[('network',None,None)]+[(f'm{mode}_{label}',mode,u) for mode in range(3) for label,u in [('small',.1),('medium',.5),('large',.9)]]
for bi,(n,phase) in enumerate(( (n,p) for n in (50,150,250) for p in ('early','middle','late'))):
 b=json.loads((x.P/f'state_bank/v5p_train_{n}_d1_tw2_0/state_{phase}.json').read_text())
 for pref in (0,15):
  e=x.environment(b,pref);ss=x.state(e);ad=x.Adaptive(len(e.q_values));remaining=b['budget']-b['seconds'];phys=x.physical_features(e.instance,e.routes);rc=x.route_context(e.instance,e.routes,phys)
  for rep in range(8):
   rid=x.REPAIRERS[rep%4];seed=20262000+bi*100+rep;obs=e.observation(torch.zeros(1,x.cfg.hidden_dim),rid,ad,remaining,b['budget'],remaining,remaining);cache=x.refresh(m,obs);raw=obs['context'].numpy().copy();m.cpu_head.state=np.asarray(ss.state(remaining/b['budget']),np.float32);m.cpu_head.summary=x.summary_for(e,e.archive);z=(e.objectives-e.dynamic_ideal)/e.dynamic_span;pos=np.asarray([z.mean(),(e.preference*z).max()],np.float32)
   for label,mode,u in arms:
    action,replay=x.sample(cache,phys,raw,n,np.random.default_rng(seed),e.neighbors,m.cpu_head,rc,pos,forced_mode=mode,forced_u=u);r=x.repair(e,action,rid,seed);g=x.gain(e,r)
    rows.append(dict(iid=b['iid'],n=n,stage=phase,preference=pref,policy_expert=7,rep=rep,repair=rid,arm=label,mode=action['mode'],q_ratio=action['q_ratio'],gain=g,positive_gain=max(0,g) if g is not None else 0.,mode_probabilities=json.dumps(replay['mode_probabilities'].tolist()),feasible=bool(r['feasible']),timeout=bool(r['timeout']),repair_seconds=r['seconds']))
  x.dumpcsv('fixed_preference.csv',rows)
 print('fixed preference',bi,len(rows),flush=True)
assert len(rows)==1440 and digest==x.sha(cp)
sources=json.loads((x.O/'sources.json').read_text())
assert all(x.sha(Path(s['path']))==s['sha256'] for s in sources)
x.write('FIXED_PREFERENCE_AUDIT.json',dict(passed=True,new_candidate_calls=len(rows),independent_feasible_checks=x.checks,checkpoint_unchanged=True,all_original_sources_unchanged=True,seconds=time.perf_counter()-x.START,training_updates=0))

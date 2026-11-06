"""Frozen action-value panel and paired continuation; no parameter updates."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,copy,hashlib,csv,traceback
import numpy as np
import torch,psutil
O=Path(__file__).resolve().parent
P=Path('D:/workspace/fuxian/DPR/development/sa5_coverage_v2_20261104')
F=Path('D:/workspace/destroy-repair-V0.01-publication/versions/v5_step5_r2_frozen_20261102')
sys.path[:0]=[str(P),str(F/'evaluation_source'),str(F/'evaluation_source/stage2_repo')]
from training_core import Pool,load_config
from network import NewPolicy,NewHead,summary_for
from session_contract import Session,calibrated_temperature
from aligned_cpu import sample
from solver.environment import Environment,Adaptive,LocalSession
from solver.config import directions,REPAIRERS
from solver.outer import union
from stage1_mechanisms import dynamic_coordinates,physical_features,refresh_required
from stage2_structured import route_context
from stage2_session import raw_context
psutil.Process().cpu_affinity([14,15]);torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.manual_seed(20261104)
START=time.perf_counter();END=START+1500
cfg=load_config(F/'evaluation_source/eval_config.json');cfg.project_root=str(F/'physical');cfg.cpu_workers=0;cfg.step5_reference_penalty=1.
pool=Pool(cfg);weights=np.asarray(directions());sources=[];checks=0
def write(name,d):
 p=O/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8');os.replace(t,p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source(p):sources.append(dict(path=str(p),sha256=sha(p)))
def guard():
 if time.perf_counter()>END:raise RuntimeError('Diagnostic 1500 second cap reached; partial rows preserved')
def dumpcsv(name,rows):
 with (O/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
@torch.no_grad()
def refresh(m,obs):
 h,nodes,x,raw=m.encode([obs]);lo,w=m.node_logits_parts(nodes,x,raw,raw.new_full((1,),.04),[0]);hi,_=m.node_logits_parts(nodes,x,raw,raw.new_full((1,),.25),[0])
 return dict(hidden=h[0].numpy().copy(),values_lo=lo[0].numpy().copy(),values_hi=hi[0].numpy().copy(),physical_weights=w[0].numpy().copy(),refresh_raw=obs['context'].numpy().copy())
def environment(bank,k):
 # Same host slot across preferences isolates preference from starting-route quality.
 e=Environment(pool,bank['iid'],cfg,weights[k],routes=bank['population'][7]['routes']);e.archive=copy.deepcopy(bank['archive'])
 ref=e.offset+e.scales*np.asarray(e.box['reference']);e.dynamic_ideal,e.dynamic_span=dynamic_coordinates([p['objectives'] for p in bank['archive']],e.scales,ref)
 return e
def state(e):
 v=e.score(e.objectives);return Session(v,v,calibrated_temperature(.01,.5),calibrated_temperature(.0001,.5))
def repair(e,action,rid,seed):
 global checks
 guard();assert len(set(action['removed']))==action['q'] and all(1<=c<=e.instance.n_customers for c in action['removed'])
 result=e.api.repair(e.instance,e.routes,e.deletion(action),rid,e.preference,e.scalar_context(),int(seed),time.perf_counter()+cfg.repair_seconds)
 if result['feasible']:
  checked=e.evaluate(result['routes']);assert np.allclose(checked,result['objectives'],atol=1e-6,rtol=0)
  assert pool.api.feasible(e.solution(result['routes']),e.instance)
  assert sorted(c for route in result['routes'] for c in route)==list(range(1,e.instance.n_customers+1))
  checks+=1
 return result
def gain(e,result):return e.score(e.objectives)-e.score(np.asarray(result['objectives'])) if result['feasible'] and not result['timeout'] else None
def nd_addition(e,result,archive):
 if not result['feasible'] or result['timeout']:return False
 p=np.asarray(result['objectives']);pts=np.asarray([x['objectives'] for x in archive])
 return bool(not np.any(np.all(pts<=p,axis=1)))
def candidate_features(e,arc):
 pts=np.asarray([p['objectives'] for p in arc]);dd=(pts-e.objectives)/e.dynamic_span;ww=np.maximum(e.preference,1e-8)
 projected=dd@ww;return dict(summary=summary_for(e,arc).tolist(),count=len(arc),objective_spread=np.ptp(pts/e.dynamic_span,axis=0).tolist(),preferred_projection_min=float(projected.min()))
def main():
 source(P/'SOURCE_MANIFEST.json')
 for name in ('network.py','learning.py','session_contract.py','aligned_cpu.py','training_core.py'):source(P/name)
 source(F/'evaluation_source/stage2_repo/solver/repair.py');source(F/'evaluation_source/stage2_structured.py')
 banks=[]
 for n in (50,150,250):
  for phase in ('early','middle','late'):
   p=P/f'state_bank/v5p_train_{n}_d1_tw2_0/state_{phase}.json';source(p);banks.append(json.loads(p.read_text()))
 models={};frozen={}
 for k in (0,7,15):
  p=P/f'training/direction_{k:02d}.pt';source(p);m=NewPolicy(cfg);m.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model']);m.eval();m.requires_grad_(False);m.cpu_head=NewHead(m);models[k]=m;frozen[k]={a:b.clone() for a,b in m.state_dict().items()}
 write('sources.json',sources)
 arms=[('network',None,None)]+[(f'm{mode}_{label}',mode,u) for mode in range(3) for label,u in [('small',.1),('medium',.5),('large',.9)]]
 rows=[];archive_rows=[];duplicate_checks=0
 for bi,b in enumerate(banks):
  for k,m in models.items():
   e=environment(b,k);ss=state(e);ad=Adaptive(len(e.q_values));remaining=b['budget']-b['seconds']
   phys=physical_features(e.instance,e.routes);rc=route_context(e.instance,e.routes,phys)
   pts=np.asarray([x['objectives'] for x in e.archive]);order=np.argsort(np.linalg.norm((pts-e.objectives)/e.dynamic_span,axis=1))
   arcs={'full':e.archive,'near16':[e.archive[i] for i in order[:16]],'far16':[e.archive[i] for i in order[-16:]]}
   for label,arc in arcs.items():archive_rows.append(dict(iid=b['iid'],stage=b['stage'],expert=k,variant=label,**candidate_features(e,arc)))
   for rep in range(8):
    rid=REPAIRERS[rep%4];seed=20262000+bi*100+rep
    obs=e.observation(torch.zeros(1,cfg.hidden_dim),rid,ad,remaining,b['budget'],remaining,remaining);cache=refresh(m,obs);raw=obs['context'].numpy().copy()
    m.cpu_head.state=np.asarray(ss.state(remaining/b['budget']),np.float32);m.cpu_head.summary=summary_for(e,e.archive)
    z=(e.objectives-e.dynamic_ideal)/e.dynamic_span;pos=np.asarray([z.mean(),(e.preference*z).max()],np.float32)
    for label,mode,u in arms:
     t=time.perf_counter();action,replay=sample(cache,phys,raw,e.instance.n_customers,np.random.default_rng(seed),e.neighbors,m.cpu_head,rc,pos,forced_mode=mode,forced_u=u)
     r=repair(e,action,rid,seed);g=gain(e,r)
     rows.append(dict(iid=b['iid'],n=e.instance.n_customers,stage=b['stage'],expert=k,rep=rep,repair=rid,arm=label,mode=action['mode'],q=action['q'],q_ratio=action['q_ratio'],gain=g,positive_gain=max(0,g) if g is not None else 0.,feasible=bool(r['feasible']),timeout=bool(r['timeout']),repair_seconds=r['seconds'],online_cpu_seconds=time.perf_counter()-t,nd_add_full=nd_addition(e,r,e.archive),nd_add_near=nd_addition(e,r,arcs['near16']),nd_add_far=nd_addition(e,r,arcs['far16']),mode_probabilities=json.dumps(replay['mode_probabilities'].tolist()),removed=json.dumps(action['removed']),objectives=json.dumps(r['objectives'])))
     if k==7 and rep==0 and label=='network':
      r2=repair(e,action,rid,seed);assert r2['feasible']==r['feasible'] and r2['timeout']==r['timeout'];assert r2['routes']==r['routes'];duplicate_checks+=1
   dumpcsv('candidate_panel.csv',rows);write('progress.json',dict(phase='candidate_panel',blocks_done=bi*3+list(models).index(k)+1,total_blocks=27,rows=len(rows),seconds=time.perf_counter()-START));print('panel',bi,k,len(rows),flush=True)
 dumpcsv('archive_features.csv',archive_rows)
 # Extension is explicitly counterfactual: after step5, hold step fraction=0 and final temperature. No claim of a trained 15-step policy.
 trajectories=[]
 for bi,b in enumerate(banks):
  k=7;m=models[k]
  for rep in range(3):
   seed=20263000+100*bi+rep
   for arm,firstmode,firstu in [('network',None,None),('local_small',0,.1),('route_large',2,.9)]:
    e=environment(b,k);ad=Adaptive(len(e.q_values));ss=state(e);ss0=ss.best;cache=None;phys=None;rc=None
    remaining=b['budget']-b['seconds'];s=LocalSession(e,ad,seed,b['budget'],remaining,remaining);s.budget=remaining;s.repair_seconds_since_refresh=0.;s.last_gnn_refresh_seconds=0.;s.dirty_since_refresh=False
    for step in range(15):
     guard();s.dirty_since_refresh|=e.last_accepted;rf=refresh_required(s);obs=s.prepare(include_observation=rf)
     if rf:
      t=time.perf_counter();cache=refresh(m,obs);s.cached_distribution=cache;s.last_gnn_refresh_seconds=time.perf_counter()-t;s.repair_seconds_since_refresh=0.;s.dirty_since_refresh=False;s.cache_age=0
     if rf or e.last_accepted:phys=physical_features(e.instance,e.routes);rc=route_context(e.instance,e.routes,phys)
     raw=raw_context(s);s.cache_age+=1;ss.closed=False;ss.attempts=min(step,4)
     features=np.asarray(ss.state(remaining/b['budget']),np.float32)
     if step>=5:features[0]=0.
     m.cpu_head.state=features;m.cpu_head.summary=summary_for(e,e.archive)
     z=(e.objectives-e.dynamic_ideal)/e.dynamic_span
     action,replay=sample(cache,phys,raw,e.instance.n_customers,np.random.default_rng(seed+step*10000),e.neighbors,m.cpu_head,rc,np.asarray([z.mean(),(e.preference*z).max()],np.float32),forced_mode=firstmode if step==0 else None,forced_u=firstu if step==0 else None)
     rid=REPAIRERS[s.pending[1]];r=repair(e,action,rid,seed+step*10000);valid=bool(r['feasible'] and not r['timeout']);before=e.score(e.objectives);candidate=e.score(np.asarray(r['objectives'])) if valid else None
     outcome=ss.transition(candidate,float(np.random.default_rng(seed+step*10000+777).random()),feasible=bool(r['feasible']),on_time=not r['timeout'])
     if valid:e.archive=union(e.archive,[dict(routes=r['routes'],objectives=r['objectives'])])
     if outcome['accepted']:e.routes=copy.deepcopy(r['routes']);e.objectives=np.asarray(r['objectives'])
     e.last_removed=action['removed'];e.steps+=1;e.last_accepted=outcome['accepted'];e.last_reward=outcome['reward'];e.stagnation=ss.stagnation
     ad.update(*s.pending,dict(accepted=outcome['reward']>0,reward=outcome['reward'],environment_seconds=r['seconds'],feasible=bool(r['feasible']),timeout=bool(r['timeout'])),ss.best)
     s.pending=None;s.hidden=torch.from_numpy(cache['hidden'].copy()).reshape(1,-1);s.step_in_session+=1;s.repair_seconds_since_refresh+=r['seconds']
     trajectories.append(dict(iid=b['iid'],n=e.instance.n_customers,stage=b['stage'],expert=k,rep=rep,arm=arm,step=step+1,mode=action['mode'],q_ratio=action['q_ratio'],repair=rid,feasible=bool(r['feasible']),timeout=bool(r['timeout']),gain_before_accept=before-candidate if valid else None,accepted=outcome['accepted'],accept_probability=outcome['probability'],temperature=outcome['temperature'],best_gain=ss0-ss.best,reward=outcome['reward'],repair_seconds=r['seconds'],nonzero_hidden=bool(np.linalg.norm(s.hidden.numpy())>0),cache_refresh=rf))
   dumpcsv('continuations.csv',trajectories);write('progress.json',dict(phase='continuation',states_done=bi,repetition=rep,rows=len(trajectories),seconds=time.perf_counter()-START))
  print('continuation',bi,len(trajectories),flush=True)
 assert len(rows)==2160 and len(trajectories)==1215 and duplicate_checks==9
 for k,m in models.items():assert all(torch.equal(v,m.state_dict()[a]) for a,v in frozen[k].items())
 for item in sources:assert sha(Path(item['path']))==item['sha256']
 write('AUDIT.json',dict(passed=True,candidate_rows=len(rows),continuation_rows=len(trajectories),repair_calls=len(rows)+len(trajectories)+duplicate_checks,independent_feasible_objective_checks=checks,repeatability_checks=duplicate_checks,source_hashes_unchanged=True,weights_unchanged=True,training_updates=0,diagnostic_only=True,seconds=time.perf_counter()-START))
 write('COMPLETE.json',dict(passed=True,seconds=time.perf_counter()-START));print('COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:write('FAILURE.json',dict(error=traceback.format_exc()));raise

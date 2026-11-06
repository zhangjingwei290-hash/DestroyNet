import os,pickle,base64
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,time,json,copy,hashlib,math,traceback,collections
O=Path(__file__).resolve().parent;P=Path('D:/workspace/fuxian/DPR/development/sa5_coverage_v2_20261104');F=Path('D:/workspace/destroy-repair-V0.01-publication/versions/v5_step5_r2_frozen_20261102')
sys.path[:0]=[str(O),str(F/'evaluation_source'),str(F/'evaluation_source/stage2_repo')]
import numpy as np,torch,psutil
from network import NewPolicy,NewHead,configure,summary_for
from learning import ppo_loss,joint_logprob,returns_and_advantages,reptile_update
from session_contract import Session,calibrated_temperature
from aligned_cpu import sample
from solver.environment import Environment,Adaptive,LocalSession
from solver.config import directions,REPAIRERS,q_values
from solver.outer import union
from solver.data import project_api
from stage1_mechanisms import dynamic_coordinates,physical_features,refresh_required
from stage2_structured import route_context
from stage2_session import raw_context
from run import load_config
def write(name,data):
 p=O/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2),encoding='utf-8');os.replace(t,p)
class Pool:
 def __init__(self):self.api=project_api(F/'physical');self.manifest=json.loads((F/'data/FIT_MANIFEST.json').read_text());self.cache={}
 def load(self,iid):
  assert iid in self.manifest['fit_ids']
  if iid not in self.cache:
   row=json.loads((F/f'data/fit/{iid}.json').read_text());self.cache[iid]=(self.api.instance(row),row,self.manifest['boxes'][iid],q_values(row['n_customers']))
  return self.cache[iid]
def episodes(m,pool,cfg,bank,w,seed,deadline,groups=4):
 slot=int(bank['direction']);startpoint=bank['population'][slot];out=[];m.cpu_head=NewHead(m);m.cpu_head.gate_enabled=True;m.cpu_head.cross_stall=bank['cross_stall']
 for rep in range(groups):
  if time.perf_counter()+.2>=deadline:break
  env=Environment(pool,bank['iid'],cfg,w,routes=startpoint['routes']);archive=copy.deepcopy(bank['archive']);env.archive=copy.deepcopy(archive)
  ref=env.offset+env.scales*np.asarray(env.box['reference']);env.dynamic_ideal,env.dynamic_span=dynamic_coordinates([a['objectives'] for a in archive],env.scales,ref)
  env.dynamic_ideal=np.asarray(bank['dynamic_ideal']);env.dynamic_span=np.asarray(bank['dynamic_span']);ad=pickle.loads(base64.b64decode(bank['adaptive_pickle']));initial=env.score(env.objectives);ad.best=initial;ss=Session(initial,initial,calibrated_temperature(.01,.5),calibrated_temperature(.0001,.5))
  # Identical initial adaptive/context state; independent stochastic trajectories.
  remaining=bank['budget']-bank['seconds'];s=LocalSession(env,ad,seed+rep*1000003,bank['budget'],remaining,remaining);s.budget=remaining;s.repair_seconds_since_refresh=0.;s.last_gnn_refresh_seconds=0.;s.dirty_since_refresh=False
  rng=np.random.default_rng(seed+rep*1000003);cache=None;ep=[];started=time.perf_counter()
  for step in range(5):
   if time.perf_counter()+.05>=deadline:break
   s.elapsed=time.perf_counter()-started;s.dirty_since_refresh|=env.last_accepted;rf=refresh_required(s);obs=s.prepare(include_observation=rf)
   if rf:
    t=time.perf_counter();cache=m.refresh(obs,retain_nodes=True);s.cached_distribution=cache;s.last_gnn_refresh_seconds=time.perf_counter()-t;s.repair_seconds_since_refresh=0.;s.dirty_since_refresh=False;s.cache_age=0
   if rf or env.last_accepted:phys=physical_features(env.instance,env.routes);rc=route_context(env.instance,env.routes,phys)
   raw=raw_context(s);features=np.asarray(ss.state(max(0,remaining-s.elapsed)/bank['budget']),np.float32);summ=summary_for(env,archive);m.cpu_head.state=features;m.cpu_head.summary=summ
   z=(env.objectives-env.dynamic_ideal)/env.dynamic_span
   action,replay=sample(cache,phys,raw,env.instance.n_customers,s.cache_rng,env.neighbors,m.cpu_head,rc,np.asarray([z.mean(),(env.preference*z).max()],np.float32));s.cache_age+=1
   r=env.api.repair(env.instance,env.routes,env.deletion(action),REPAIRERS[s.pending[1]],env.preference,env.scalar_context(),int(rng.integers(2**31)),min(deadline,time.perf_counter()+cfg.repair_seconds))
   if r['feasible']:assert np.allclose(env.evaluate(r['routes']),r['objectives'],rtol=0,atol=1e-6)
   valid=r['feasible'] and not r['timeout'] and time.perf_counter()<deadline;candidate=env.score(np.asarray(r['objectives'])) if valid else None
   bonus=0.;result=ss.transition(candidate,float(rng.random()),feasible=bool(r['feasible']),on_time=not r['timeout'] and time.perf_counter()<deadline)
   if result['archive_eligible']:env.archive=union(env.archive,[dict(routes=r['routes'],objectives=r['objectives'])]);archive=union(archive,env.archive)
   if result['accepted']:env.routes=copy.deepcopy(r['routes']);env.objectives=np.asarray(r['objectives'])
   env.last_removed=action['removed'];env.steps+=1;env.last_accepted=result['accepted'];env.last_reward=result['reward'];env.stagnation=ss.stagnation
   ad.update(*s.pending,dict(accepted=result['reward']>0,reward=result['reward'],environment_seconds=r['seconds'],feasible=bool(r['feasible']),timeout=not valid),ss.best)
   s.pending=None;s.hidden=torch.from_numpy(cache['hidden'].copy()).reshape(1,-1);s.step_in_session+=1;s.repair_seconds_since_refresh+=r['seconds']
   replay.update(gate_context=m.cpu_head.gate_context.copy(),cross_stall=bank['cross_stall'],search_state=features,set_summary=summ,reward=result['reward']+.01*bonus,scalar_reward=result['reward'],archive_reward=.01*bonus,value=m.cpu_head.value,terminated=result['terminated'],truncated=result['truncated'],iid=bank['iid'],phase=bank['stage'],slot=slot,group=rep)
   ep.append(replay)
   if result['truncated']:break
  if len(ep)==5 and ep[-1]['terminated']:out.append(ep)
 return out

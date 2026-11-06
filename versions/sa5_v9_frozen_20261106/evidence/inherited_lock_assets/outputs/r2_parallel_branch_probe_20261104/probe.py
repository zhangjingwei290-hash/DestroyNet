"""Bounded engineering probe, 16 directions x {1,2} branches x 2 transitions x3 scales.
Not a full solver, SA calibration, training run or scientific performance comparison.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys,time,json,copy,hashlib,math,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
F=Path('D:/workspace/destroy-repair-V0.01-publication/versions/v5_step5_r2_frozen_20261102');O=Path(__file__).resolve().parent
sys.path.insert(0,str(F/'evaluation_source/stage2_repo'));sys.path.insert(0,str(F/'evaluation_source'))
import numpy as np,psutil
WORK=None

def write(name,d):
 p=O/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else x.item()),encoding='utf-8');os.replace(t,p)
def worker_init(q):
 global WORK
 psutil.Process().cpu_affinity(q.get());from solver.data import project_api
 api=project_api(F/'physical');WORK=(api,{})
 write('worker_'+str(os.getpid())+'.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),ppid=os.getppid(),cmd=psutil.Process().cmdline(),affinity=psutil.Process().cpu_affinity()))
def repair_job(task):
 api,cache=WORK;iid=task['iid']
 if iid not in cache:cache[iid]=api.instance(json.loads((F/f'data/test/{iid}.json').read_text()))
 from solver.repair import repair
 begin=time.perf_counter();r=repair(cache[iid],task['routes'],task['removed'],task['repair'],task['weight'],task['scalar_context'],task['seed'],begin+2.)
 return dict(result=r,wall=time.perf_counter()-begin,pid=os.getpid(),identity=(task['direction'],task['branch'],task['round']))
class Pool:
 def __init__(self,cfg):
  from solver.data import project_api
  self.api=project_api(F/'physical');self.cache={};self.boxes=json.loads((F/'evaluation_source/boxes.json').read_text())
 def load(self,iid):
  from solver.config import q_values
  if iid not in self.cache:
   row=json.loads((F/f'data/test/{iid}.json').read_text());self.cache[iid]=(self.api.instance(row),row,self.boxes[iid],q_values(row['n_customers']))
  return self.cache[iid]

def main():
 import torch
 from run import load_config
 from solver.model import Policy
 from stage2_cpu import CpuHead,sample
 from solver.config import directions,REPAIRERS
 from solver.environment import Environment,Adaptive,LocalSession
 from solver.outer import Search,Genome,union
 from stage1_mechanisms import dynamic_coordinates,refresh_required,physical_features
 from stage2_structured import route_context
 from stage2_session import raw_context
 psutil.Process().cpu_affinity([6,7]);torch.set_num_threads(1);torch.set_num_interop_threads(1);assert torch.cuda.is_available()
 cfg=load_config(F/'evaluation_source/eval_config.json');cfg.project_root=str(F/'physical');cfg.cpu_workers=0;cfg.step5_reference_penalty=1.;pool=Pool(cfg);models=[];digests=[];before={}
 for k in range(16):
  p=F/f'checkpoints/direction_{k:02d}.pt';digests.append(hashlib.sha256(p.read_bytes()).hexdigest());m=Policy(cfg).cuda();m.load_state_dict(torch.load(p,map_location='cuda',weights_only=False)['model']);m.eval();m.requires_grad_(False);m.cpu_head=CpuHead(m,legacy_feature_reductions=True);models.append(m);before[k]={n:v.detach().clone() for n,v in m.state_dict().items()}
 # Three workers use distinct CPUs outside the active random-q0..5 and monitor14/15.
 ctx=mp.get_context('spawn');q=ctx.Queue()
 for a in [[8,9],[10,11],[12,13]]:q.put(a)
 ex=ProcessPoolExecutor(3,mp_context=ctx,initializer=worker_init,initargs=(q,));cases=[];allchecks=[]
 try:
  for iid in ['50-0-0','150-1-2','250-2-4']:
   search=Search(pool,iid,cfg,20261104,policies=models);alg=search.algorithm;snapshot=copy.deepcopy(alg.pop);snapshotF=np.array([p.F for p in snapshot]);snapshotRoutes=[copy.deepcopy(p.X[0].routes) for p in snapshot]
   children=[];rng=np.random.RandomState(20261104);old_rng=np.random.get_state();np.random.set_state(rng.get_state())
   try:
    for k in range(16):
     P=alg.selection.do(alg.problem,snapshot,1,alg.mating.crossover.n_parents,neighbors=[alg.neighbors[k]])
     off=np.random.choice(alg.mating.do(alg.problem,snapshot,1,parents=P,n_max_iterations=1));g=off.X[0];fallback=not pool.api.feasible(search.base.solution(g.routes),search.base.instance);routes=g.fallback if fallback else g.routes
     children.append(dict(routes=copy.deepcopy(routes),fallback=fallback))
   finally:np.random.set_state(old_rng)
   assert snapshotRoutes==[p.X[0].routes for p in alg.pop] and np.array_equal(snapshotF,np.array([p.F for p in alg.pop]))
   refs=directions();ref=search.base.offset+search.base.scales*np.asarray(search.base.box['reference']);ideal,span=dynamic_coordinates([p['objectives'] for p in search.archive],search.base.scales,ref)
   for B in [1,2]:
    lanes=[];timing=[];audit=[];bad_accept=0;accepts=0;handoffs=0;cachehits=0;replayequal=0;temps=None
    for k in range(16):
     for b in range(B):
      env=Environment(pool,iid,cfg,refs[k],routes=children[k]['routes']);env.dynamic_ideal=ideal.copy();env.dynamic_span=span.copy();ad=Adaptive(len(env.q_values));ad.best=env.score(env.objectives);ses=LocalSession(env,ad,20261104+k*101+b,300.,300.,10.)
      ses.repair_seconds_since_refresh=0.;ses.last_gnn_refresh_seconds=0.;ses.dirty_since_refresh=False
      lanes.append(dict(k=k,b=b,session=ses,best=copy.deepcopy(env.routes),bestscore=env.score(env.objectives),rng=np.random.default_rng(20261104+k*103+b),previous_hidden=None))
    assert len({id(v['session'].env.routes) for v in lanes})==16*B
    batchstart=time.perf_counter()
    for roundidx in range(2):
     tasks=[];actions=[];neuralbegin=time.perf_counter()
     for lane in lanes:
      k=lane['k'];s=lane['session'];env=s.env;s.dirty_since_refresh|=bool(env.last_accepted);refresh=refresh_required(s);obs=s.prepare(include_observation=refresh)
      if refresh:
       incoming=obs['state_'].detach().cpu().numpy().reshape(-1)
       if lane['previous_hidden'] is not None:assert np.array_equal(incoming,lane['previous_hidden']);handoffs+=int(np.linalg.norm(incoming)>0)
       else:assert np.all(incoming==0)
       t=time.perf_counter()
       with torch.no_grad():cache=models[k].refresh(obs)
       s.cached_distribution=cache;s.cache_hidden=torch.from_numpy(cache['hidden'].copy()).reshape(1,-1);lane['previous_hidden']=cache['hidden'].copy();lane['phys']=obs['x'][1:,[18,17,19,12,13]].numpy().copy();lane['routectx']=route_context(env.instance,env.routes,lane['phys']);s.cache_age=0;s.last_gnn_refresh_seconds=time.perf_counter()-t;s.repair_seconds_since_refresh=0.;s.dirty_since_refresh=False
      else:
       cachehits+=1
       if env.last_accepted:lane['phys']=physical_features(env.instance,env.routes);lane['routectx']=route_context(env.instance,env.routes,lane['phys'])
      raw=raw_context(s);z=(env.objectives-env.dynamic_ideal)/env.dynamic_span
      action,replay=sample(s.cached_distribution,lane['phys'],raw,env.instance.n_customers,s.cache_rng,env.neighbors,models[k].cpu_head,lane['routectx'],np.array([z.mean(),(refs[k]*z).max()],np.float32));s.cache_age+=1
      _,repairindex=s.pending;task=dict(iid=iid,direction=k,branch=lane['b'],round=roundidx,routes=copy.deepcopy(env.routes),removed=list(action['removed']),repair=REPAIRERS[repairindex],weight=refs[k].copy(),scalar_context=env.scalar_context(),seed=int(s.rng.integers(0,2**31-1)));tasks.append(task);actions.append(action)
     neural_seconds=time.perf_counter()-neuralbegin
     # Same finite tasks first serialized on one worker, then on all three workers.
     started=time.perf_counter();serial=[ex.submit(repair_job,t).result(timeout=30) for t in tasks];serialseconds=time.perf_counter()-started
     started=time.perf_counter();futures=[ex.submit(repair_job,t) for t in tasks];parallel=[f.result(timeout=30) for f in futures];parallelseconds=time.perf_counter()-started
     for t,a,b in zip(tasks,serial,parallel):
      assert a['identity']==b['identity']==(t['direction'],t['branch'],t['round'])
      ra,rb=a['result'],b['result'];assert not ra['timeout'] and not rb['timeout'],'Probe hit2s repair guard; timing replay comparison invalid'
      assert ra['feasible']==rb['feasible'] and ra.get('routes')==rb.get('routes')
      if ra['feasible']:np.testing.assert_allclose(ra['objectives'],rb['objectives'],rtol=0,atol=1e-7)
      replayequal+=1
     gaps=[]
     for lane,rr in zip(lanes,parallel):
      if rr['result']['feasible']:gaps.append(abs(lane['session'].env.score(np.array(rr['result']['objectives']))-lane['session'].env.score(lane['session'].env.objectives)))
     if temps is None:
      # Diagnostic-only dimensionless scale. Not a selected production cooling schedule.
      temp=max(float(np.median(gaps))/math.log(2),1e-6);temps=[temp,temp*.1]
     for lane,task,action,rr in zip(lanes,tasks,actions,parallel):
      s=lane['session'];env=s.env;previous_stag=env.stagnation;ad_before=copy.deepcopy(s.adaptive);pending_before=s.pending;before_score=env.score(env.objectives);candidate=rr['result'];repairid=task['repair'];previous=env.api.repair
      env.api.repair=lambda *args,_r=candidate,**kwargs:copy.deepcopy(_r)
      try:result=s.finish(action,s.cache_hidden,0.)
      finally:env.api.repair=previous
      # super.finish preserves original validation/ND handling; explicitly replace strict
      # acceptance transition with diagnostic Metropolis acceptance where candidate is worse.
      if candidate['feasible']:
       f=np.asarray(candidate['objectives']);score=env.score(f);delta=score-before_score;prob=1. if delta<=0 else math.exp(-delta/temps[roundidx]);u=float(lane['rng'].random());accepted=u<prob
       if accepted:
        env.routes=copy.deepcopy(candidate['routes']);env.objectives=f.copy();bad_accept+=int(delta>0);accepts+=1
       env.last_accepted=accepted;env.last_reward=max(0.,before_score-score) if accepted else 0.
       # stagnation means failure to improve best, not failure to accept.
       improved=score<lane['bestscore']
       if improved:lane['bestscore']=score;lane['best']=copy.deepcopy(candidate['routes'])
       env.stagnation=0 if improved else previous_stag+1
       audit.append(dict(direction=lane['k'],branch=lane['b'],round=roundidx,temperature=temps[roundidx],delta=delta,probability=prob,u=u,accepted=accepted,feasible=True))
      s.adaptive=ad_before;ad_result=dict(result,accepted=bool(env.last_accepted and env.last_reward>0),reward=env.last_reward);s.adaptive.update(*pending_before,ad_result,env.score(env.objectives))
      s.step_in_session+=1;s.repair_seconds_since_refresh+=candidate['seconds'];assert pool.api.feasible(env.solution(env.routes),env.instance);np.testing.assert_allclose(env.evaluate(env.routes),env.objectives,rtol=0,atol=1e-7)
     timing.append(dict(round=roundidx,neural_serial_dispatch_seconds=neural_seconds,serialized_repair_rpc_seconds=serialseconds,three_worker_repair_rpc_seconds=parallelseconds,repair_only_speedup=serialseconds/parallelseconds,worker_ids=sorted(set(r['pid'] for r in parallel))))
    # Read-only staging until barrier; commit selected branch per direction, deterministic k order.
    assert snapshotRoutes==[p.X[0].routes for p in alg.pop]
    candidates=[];merged=copy.deepcopy(search.archive)
    for k in range(16):
     selected=min([v for v in lanes if v['k']==k],key=lambda v:(v['bestscore'],v['b']));env=selected['session'].env;routes=selected['best'];obj=env.evaluate(routes);c=copy.deepcopy(snapshot[k]);c.set('X',np.asarray([Genome(copy.deepcopy(routes),copy.deepcopy(routes))],object));c.set('F',obj);candidates.append(c)
     for lane in lanes:
      if lane['k']==k:merged=union(merged,lane['session'].env.archive)
    updated=copy.deepcopy(snapshot);alg.pop=updated;alg.decomposition.dynamic_ideal=ideal.copy();alg.decomposition.dynamic_span=span.copy();alg.decomposition.reference=ref;alg.decomposition.penalty=cfg.step5_reference_penalty
    for k,c in enumerate(candidates):alg.ideal=np.minimum(alg.ideal,c.F);alg._replace(k,c)
    for p in alg.pop:assert pool.api.feasible(search.base.solution(p.X[0].routes),search.base.instance);np.testing.assert_allclose(search.base.evaluate(p.X[0].routes),p.F,rtol=0,atol=1e-7)
    for point in merged:assert pool.api.feasible(search.base.solution(point['routes']),search.base.instance);np.testing.assert_allclose(search.base.evaluate(point['routes']),point['objectives'],rtol=0,atol=1e-7)
    write(f'{iid}_B{B}_transitions.json',audit);case=dict(iid=iid,branches=B,directions=16,rounds=2,transitions=len(tasks)*2,sequential_parallel_exact_repair_replays=replayequal,accepted=accepts,worse_accepted=bad_accept,nonzero_history_handoffs=handoffs,cache_hits=cachehits,parent_snapshot_unchanged_until_barrier=True,branch_routes_state_isolated=True,physical_and_objectives_passed=True,archive_points=len(merged),total_probe_seconds=time.perf_counter()-batchstart,timing=timing,temperature_scope='diagnostic Metropolis only, not validated cooling or training',gpu_inference_scope='16 expert weights dispatched sequentially on GPU, not true batched forward');cases.append(case);write('progress.json',dict(completed=len(cases),total=6,last=case))
    alg.pop=copy.deepcopy(snapshot)
  assert all(torch.equal(v,models[k].state_dict()[n]) for k,sd in before.items() for n,v in sd.items())
  write('summary.json',dict(passed=True,cases=cases,weights_unchanged=True,training=False,formal_searches=0,scope='Bounded two-level scheduler/repair parallelism and SA transition probe; not full time-based SA, full solver or learned adaptation',active_campaign_untouched=True,repair_worker_affinities=[[8,9],[10,11],[12,13]],gpu_main_affinity=[6,7]));write('AUDIT.json',dict(passed=True,checks=['snapshot child generation','independent branch RNG/routes/hidden/cache','same-task sequential vs3worker repair exact outputs','Metropolis state update','feasible5objective recomputation','barrier deterministic population merge','frozen weights torch.equal'],cases=len(cases),transitions=sum(c['transitions'] for c in cases)));print(json.dumps(dict(passed=True,cases=cases)))
 finally:ex.shutdown(wait=True,cancel_futures=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback;write('FAILURE.json',dict(error=traceback.format_exc()));raise

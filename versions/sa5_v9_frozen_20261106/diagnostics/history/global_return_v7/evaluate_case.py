"""Finite16-direction x5-attempt SA transfer throughput measurement; no training/full solver evaluation."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys,json,time,math,copy,hashlib,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np,psutil,pickle,base64
F=Path('D:/workspace/destroy-repair-V0.01-publication/versions/v5_step5_r2_frozen_20261102');O=Path(os.environ['CASE_ROOT']);O.mkdir(parents=True,exist_ok=True);(O/'results').mkdir(exist_ok=True);(O/'analysis').mkdir(exist_ok=True)
sys.path.insert(0,str(F/'evaluation_source/stage2_repo'));sys.path.insert(0,str(F/'evaluation_source'));sys.path.insert(0,'D:/workspace/outputs/r2_parallel_branch_probe_20261104')
class Pool:
 def __init__(self,cfg):
  from solver.data import project_api
  self.api=project_api(F/'physical');self.cache={};self.boxes=json.loads((F/'data/FIT_MANIFEST.json').read_text())['boxes'] if os.environ.get('TRAIN_CAPTURE') else json.loads((F/'evaluation_source/boxes.json').read_text())
 def load(self,iid):
  from solver.config import q_values
  if iid not in self.cache:
   row=json.loads((F/f"data/{'fit' if os.environ.get('TRAIN_CAPTURE') else 'test'}/{iid}.json").read_text());self.cache[iid]=(self.api.instance(row),row,self.boxes[iid],q_values(row['n_customers']))
  return self.cache[iid]

WORK=None

def write(name,d):
 p=O/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else x.item()),encoding='utf-8');os.replace(t,p)
def worker_init(q,barrier):
 global WORK
 affinity=q.get();psutil.Process().cpu_affinity(affinity)
 from solver.data import project_api
 WORK=(project_api(F/'physical'),{},barrier)
 write(f'worker_{os.getpid()}.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),ppid=os.getppid(),cmd=psutil.Process().cmdline(),affinity=affinity))
def ready(_):
 WORK[2].wait(timeout=60);return os.getpid()
def repair_job(t):
 api,cache,barrier=WORK
 if t['iid'] not in cache:cache[t['iid']]=api.instance(json.loads((F/f"data/test/{t['iid']}.json").read_text()))
 from solver.repair import repair
 started=time.perf_counter();r=repair(cache[t['iid']],t['routes'],t['removed'],t['repair'],t['weight'],t['scalar_context'],t['seed'],t['deadline']);finished=time.perf_counter()
 return dict(result=r,direction=t['k'],pid=os.getpid(),started=started,finished=finished)

def physical(row,sol):
 routes=sol['routes'];n=row['n_customers']
 assert sorted(c for r in routes for c in r)==list(range(1,n+1)) and all(routes)
 f=[len(routes),0.,0.,0.,0.]
 for route in routes:
  assert sum(row['customer_demands'][c-1] for c in route)<=row['vehicle_capacity']+1e-8
  t=0.;p=0
  for c in route:
   t+=row['duration_matrix'][p][c];f[1]+=row['distance_matrix'][p][c]
   lo,hi=row['customer_time_windows'][c-1]
   assert t<=hi+row['max_delay']+1e-6
   f[3]+=max(0,lo-t);f[4]+=max(0,t-hi);t=max(t,lo)+row['customer_service_times'][c-1];p=c
  t+=row['duration_matrix'][p][0];f[1]+=row['distance_matrix'][p][0]
  assert t<=row['depot_time_window'][1]+1e-6
  f[2]=max(f[2],t)
 assert all(abs(a-b)<=1e-6 for a,b in zip(f,sol['objectives']))

def warm_worker(iid):
 api,cache,barrier=WORK
 cache[iid]=api.instance(json.loads((F/f"data/{'fit' if os.environ.get('TRAIN_CAPTURE') else 'test'}/{iid}.json").read_text()))
 barrier.wait(timeout=90)
 return os.getpid()

def main():
 import torch
 from run import load_config
 from network import NewPolicy as Policy,NewHead,summary_for,load_base
 from session_contract import calibrated_temperature,Session
 from aligned_cpu import CpuHead,sample
 from solver.config import directions,REPAIRERS
 from solver.environment import Environment,Adaptive,LocalSession
 from solver.outer import Search,Genome,union
 from stage1_mechanisms import dynamic_coordinates,refresh_required,physical_features
 from stage2_structured import route_context
 from stage2_session import raw_context
 psutil.Process().cpu_affinity([6,7]);torch.set_num_threads(1);torch.set_num_interop_threads(1);assert torch.cuda.is_available();cfg=load_config(F/'evaluation_source/eval_config.json');cfg.project_root=str(F/'physical');cfg.cpu_workers=0;cfg.step5_reference_penalty=1.;pool=Pool(cfg)
 models=[];frozen=[]
 for k in range(16):
  m=Policy(cfg).cuda();load_base(m,Path(os.environ['CHECKPOINT_ROOT'])/f'direction_{k:02d}.pt');m.eval();m.requires_grad_(False);m.cpu_head=NewHead(m,legacy_feature_reductions=False);models.append(m);frozen.append({n:v.detach().clone() for n,v in m.state_dict().items()})
 for k in range(1,16):assert all(torch.equal(v,models[k].encoder.state_dict()[n]) for n,v in models[0].encoder.state_dict().items()),'Cannot share different encoders'
 # GPU batched shared encoder is exact because all expert encoders are identical.
 # Expert-specific film/node matrices are stacked, not averaged.
 def linear_expert(x,ids,name):
  w=torch.stack([models[k].state_dict()[name+'.weight'] for k in ids]);b=torch.stack([models[k].state_dict()[name+'.bias'] for k in ids]);return torch.bmm(x,w.transpose(1,2))+b[:,None]
 @torch.no_grad()
 def batch_refresh(obs,ids):
  hidden,nodes,x,raw=models[0].encode(obs);outputs=[]
  for frac in [.04,.25]:
   inp=torch.cat((raw[:,:5],raw.new_full((len(ids),1),frac),raw[:,14:15]),-1)[:,None]
   film=linear_expert(torch.nn.functional.elu(linear_expert(inp,ids,'film.0')),ids,'film.2').squeeze(1);gamma,beta=film.chunk(2,-1)
   z=torch.tanh((1+gamma[:,None])*nodes+beta[:,None]);z=torch.nn.functional.elu(linear_expert(z,ids,'heads.0.node.0'));outputs.append(linear_expert(z,ids,'heads.0.node.2').squeeze(-1))
  w=torch.stack([models[k].heads[0].physical_bilinear for k in ids]);physicalw=torch.bmm(raw[:,:5,None].transpose(1,2),w).squeeze(1)
  packed=torch.cat((hidden,*outputs,physicalw),1).cpu().numpy();h=cfg.hidden_dim;n=len(obs[0]['x'])-1
  return [dict(hidden=row[:h].copy(),values_lo=row[h:h+n].copy(),values_hi=row[h+n:h+2*n].copy(),physical_weights=row[h+2*n:].copy(),refresh_raw=o['context'].numpy().copy(),nodes=node.copy()) for row,o,node in zip(packed,obs,nodes.detach().cpu().numpy())]


 BUDGET=float(os.environ['CASE_SECONDS']);CAP=5;SEED=int(os.environ['CASE_SEED']);IID=os.environ['CASE_IID']
 from solver.environment import classical_removal
 import moocore
 def verify_manifest():
  manifest=json.loads(Path(os.environ['STAGE_MANIFEST']).read_text(encoding='utf-8'))
  for item in manifest['files']:
   p=Path(item['windows_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],str(p)
 verify_manifest()
 # Deterministic acceptance checks, no extra optimization searches.
 def metropolis(delta,temp,u):
  assert temp>0 and math.isfinite(temp) and 0<=u<1
  prob=1. if delta<=0 else math.exp(-delta/temp)
  return u<prob,prob
 assert metropolis(-1.,.1,.999)[0]
 assert metropolis(1.,1.,.1)[0] and not metropolis(1.,1.,.9)[0]
 assert metropolis(0.,1.,.99)[0]
 ctx=mp.get_context('spawn');q=ctx.Queue()
 for k in range(16):q.put([8+k%6])
 barrier=ctx.Barrier(16);ex=ProcessPoolExecutor(16,mp_context=ctx,initializer=worker_init,initargs=(q,barrier));reports=[]
 try:
  fs=[ex.submit(ready,k) for k in range(16)];pids=[f.result(timeout=120) for f in fs];assert len(set(pids))==16
  inst,row,box,_=pool.load(IID)
  fs=[ex.submit(warm_worker,IID) for _ in range(16)];assert len(set(f.result(timeout=120) for f in fs))==16
  # One frozen forward warmup and batch-vs-reference check, no extra search.
  ob=[]
  for k in range(16):
   env=Environment(pool,IID,cfg,directions()[k]);ad=Adaptive(len(env.q_values));ob.append(env.observation(torch.zeros(1,cfg.hidden_dim),'greedy',ad,30.,30.,30.,30.))
  packed=batch_refresh(ob,list(range(16)));maxerr=0.
  for k,x in enumerate(ob):
   original=models[k].refresh(x)
   for name in ['hidden','values_lo','values_hi','physical_weights']:maxerr=max(maxerr,float(np.max(np.abs(original[name]-packed[k][name]))))
  assert maxerr<5e-5;torch.cuda.synchronize()
  direct=[]
  for pid in pids:
   p=psutil.Process(pid);assert p.ppid()==os.getpid();direct.append(dict(pid=pid,created=p.create_time(),ppid=p.ppid(),cmd=p.cmdline(),affinity=p.cpu_affinity()))
  write('READY.json',dict(parent_pid=os.getpid(),created=psutil.Process().create_time(),parent_command=psutil.Process().cmdline(),worker_count=16,workers=direct,batch_forward_error=maxerr,memory_available=psutil.virtual_memory().available))
  print('READY:16 persistent workers, both methods parallel',flush=True)
  initial_reference=None
  for method in json.loads(os.environ['METHODS']):
   verify_manifest();neural=method!='roulette_sa5'
   roots=json.loads(os.environ.get('METHOD_WEIGHTS','{}'));root=Path(roots.get(method,'D:/workspace/fuxian/DPR/development/sa5_coverage_v2_20261104/training'))
   for k,m in enumerate(models):
    load_base(m,root/f'direction_{k:02d}.pt')
    if method=='gate_trained':m.stall_gate.load_state_dict(torch.load(Path(os.environ['GATE_PATH']),map_location='cuda',weights_only=False))
    else:
     torch.nn.init.zeros_(m.stall_gate.body[-1].weight);torch.nn.init.zeros_(m.stall_gate.body[-1].bias)
    m.cpu_head=NewHead(m)
   frozen=[{n:v.detach().clone() for n,v in m.state_dict().items()} for m in models];stall=[0]*16;historical=[None]*16;bias_calls=0
   rng=np.random.default_rng(np.random.SeedSequence([SEED,717]));records=[];batches=[];stats=[dict(calls=0,completed5=0,attempts=0,accepted=0,improvements=0,worse=0,late=0,cache_hits=0,history=0) for _ in range(16)]
   timer=time.perf_counter();deadline=timer+BUDGET
   search=Search(pool,IID,cfg,SEED,policies=models if neural else None,seconds=BUDGET);alg=search.algorithm
   initial=[dict(routes=copy.deepcopy(p.X[0].routes),objectives=p.F.tolist()) for p in alg.pop]
   if initial_reference is None:initial_reference=copy.deepcopy(initial)
   else:assert initial_reference==initial,'Common initial population differs'
   archive=copy.deepcopy(search.archive);initial_archive=copy.deepcopy(archive);training_rows=[];
   def quality(points):
    z=(np.asarray([p['objectives'] for p in points])-search.base.offset)/search.base.scales
    return -float(np.min(np.max(np.asarray(directions())[:,None,:]*z[None,:,:],axis=2),axis=1).mean())
   initial_seconds=time.perf_counter()-timer
   point_events=[dict(seconds=initial_seconds,objectives=p['objectives']) for p in archive];snapshots=[];last_log=0.;physical_count=0;capture_next=0.;captured=[]
   print(method,'START',json.dumps({'pid':os.getpid(),'initial_seconds':initial_seconds}),flush=True)
   while time.perf_counter()<deadline:
    batch_start=time.perf_counter();source=copy.deepcopy(alg.pop)
    # Original pymoo selection/crossover, one shared population snapshot per batch.
    def vary():
     order=np.random.permutation(16).tolist();children=[]
     for k in order:
      if time.perf_counter()>=deadline:break
      parents=alg.selection.do(alg.problem,source,1,alg.mating.crossover.n_parents,neighbors=[alg.neighbors[k]])
      off=np.random.choice(alg.mating.do(alg.problem,source,1,parents=parents,n_max_iterations=1));g=off.X[0];routes=g.routes
      if not pool.api.feasible(search.base.solution(routes),search.base.instance):routes=g.fallback
      off.set('X',np.asarray([Genome(copy.deepcopy(routes),copy.deepcopy(routes))],dtype=object));off.set('F',search.base.evaluate(routes));children.append((int(k),off))
     return order,children
    order,children=search._random_call(vary)
    if not children or time.perf_counter()>=deadline:break
    ref=search.base.offset+search.base.scales*np.asarray(search.base.box['reference']);ideal,span=dynamic_coordinates([p['objectives'] for p in archive],search.base.scales,ref)
    lanes=[]
    for k,off in children:
     if time.perf_counter()>=deadline:break
     env=Environment(pool,IID,cfg,directions()[k],routes=off.X[0].routes);env.dynamic_ideal=ideal.copy();env.dynamic_span=span.copy()
     now=time.perf_counter()
     if now>=deadline:break
     physical(row,dict(routes=env.routes,objectives=env.objectives));physical_count+=1
     if time.perf_counter()>=deadline:break
     admitted=time.perf_counter();archive=union(archive,env.archive);point_events.append(dict(seconds=admitted-timer,objectives=env.objectives.tolist()))
     remaining=max(1e-9,deadline-time.perf_counter());s=LocalSession(env,search.adaptive[k],int(SEED+1009*(len(batches)*16+k)),BUDGET,remaining,remaining);s.budget=remaining;s.outer_deadline=deadline;s.solver_task_deadline=deadline
     s.repair_seconds_since_refresh=0.;s.last_gnn_refresh_seconds=0.;s.dirty_since_refresh=False;s.adaptive.best=env.score(env.objectives)
     lanes.append(dict(k=k,s=s,off=off,step=0,previous_hidden=None,bestscore=env.score(env.objectives),best_routes=copy.deepcopy(env.routes),best_obj=env.objectives.copy(),temperature0=calibrated_temperature(.01,.5),rng=np.random.default_rng(np.random.SeedSequence([SEED,len(batches),k,719])),physical=None,rc=None))
     lanes[-1]['transition']=Session(env.score(env.objectives),env.score(env.objectives),calibrated_temperature(.01,.5),calibrated_temperature(.0001,.5))
     stats[k]['calls']+=1
    if not lanes:break
    if os.environ.get('CAPTURE_STATES') and time.perf_counter()-timer>=capture_next:
     for v in lanes:
      k=v['k'];captured.append(dict(iid=IID,stage='real_cross_session',budget=BUDGET,seconds=time.perf_counter()-timer,direction=k,cross_stall=stall[k],dynamic_ideal=ideal.tolist(),dynamic_span=span.tolist(),adaptive_pickle=base64.b64encode(pickle.dumps(search.adaptive[k])).decode(),population=[dict(routes=copy.deepcopy(x['s'].env.routes),objectives=x['s'].env.objectives.tolist()) for x in sorted(lanes,key=lambda x:x['k'])],archive=copy.deepcopy(archive)))
     # Only complete16-lane states have slot-index correspondence.
     if len(lanes)!=16:captured=captured[:-len(lanes)]
     capture_next=time.perf_counter()-timer+10.
    waves=0
    for j in range(5):
     if time.perf_counter()>=deadline:break
     obs=[];refresh_lanes=[];active=[]
     for v in lanes:
      s=v['s'];s.elapsed=max(0.,s.global_start-(deadline-time.perf_counter()))
      if s.remaining()<=0:continue
      if neural:
       s.dirty_since_refresh|=s.env.last_accepted;v['refresh']=refresh_required(s);o=s.prepare(include_observation=v['refresh'])
       if v['refresh']:
        hidden=o['state_'].numpy().reshape(-1)
        if v['previous_hidden'] is None:assert np.all(hidden==0)
        else:assert np.array_equal(hidden,v['previous_hidden']);stats[v['k']]['history']+=int(np.linalg.norm(hidden)>0)
        obs.append(o);refresh_lanes.append(v)
      else:s.prepare(include_observation=False)
      active.append(v)
     if obs:
      begin=time.perf_counter();cache=batch_refresh(obs,[v['k'] for v in refresh_lanes]);gpu_time=time.perf_counter()-begin
      for v,o,c in zip(refresh_lanes,obs,cache):
       s=v['s'];s.cached_distribution=c;s.cache_hidden=torch.from_numpy(c['hidden'].copy()).reshape(1,-1);v['previous_hidden']=c['hidden'].copy();v['physical']=o['x'][1:,[18,17,19,12,13]].numpy().copy();v['rc']=route_context(s.env.instance,s.env.routes,v['physical']);s.cache_age=0;s.last_gnn_refresh_seconds=gpu_time/len(obs);s.repair_seconds_since_refresh=0.;s.dirty_since_refresh=False
     tasks=[];used=[]
     for v in active:
      s=v['s'];env=s.env;k=v['k'];s.elapsed=max(0.,s.global_start-(deadline-time.perf_counter()))
      if s.remaining()<=0:s.pending=None;continue
      if neural:
       if not v['refresh']:
        stats[k]['cache_hits']+=1
        if env.last_accepted:v['physical']=physical_features(env.instance,env.routes);v['rc']=route_context(env.instance,env.routes,v['physical'])
       raw=raw_context(s);z=(env.objectives-env.dynamic_ideal)/env.dynamic_span
       models[k].cpu_head.q_bias=.10 if method=='v2_stallq' and stall[k]>=3 else 0.;models[k].cpu_head.cross_stall=stall[k];models[k].cpu_head.gate_enabled=method in ('gate_untrained','gate_trained')
       models[k].cpu_head.state=np.asarray(v['transition'].state(max(0.,deadline-time.perf_counter())/BUDGET),np.float32);models[k].cpu_head.summary=summary_for(env,archive)
       action,replay=sample(s.cached_distribution,v['physical'],raw,env.instance.n_customers,s.cache_rng,env.neighbors,models[k].cpu_head,v['rc'],np.array([z.mean(),(env.preference*z).max()],np.float32));s.cache_age+=1
       removed=env.deletion(action)
       if os.environ.get('GLOBAL_ROLLOUT'):
        replay.update(gate_context=models[k].cpu_head.gate_context.copy(),cross_stall=stall[k],search_state=models[k].cpu_head.state.copy(),set_summary=models[k].cpu_head.summary.copy(),expert=k,global_quality=quality(archive),global_seconds=time.perf_counter()-timer);training_rows.append(replay)
      else:
       removed=classical_removal(env,s.pending[0],s.rng);action=dict(q=len(removed),removed=removed)
      assert len(set(removed))==len(removed) and 1<=len(removed)<=env.instance.n_customers
      v['q_bias']=float(models[k].cpu_head.q_bias) if neural else 0.;v['gate_shift']=float(models[k].cpu_head.gate_shift) if neural else 0.;bias_calls+=int(v['q_bias']>0)
      v['action']=action;v['before_routes']=copy.deepcopy(env.routes);v['before_obj']=env.objectives.copy();v['before_score']=env.score(env.objectives);v['before_best']=v['bestscore'];v['before_stagnation']=env.stagnation
      if v['step']==0:v['historical_before']=None if historical[k] is None else env.score(historical[k])
      if time.perf_counter()>=deadline:s.pending=None;continue
      repairer=REPAIRERS[s.pending[1]];task=dict(iid=IID,k=k,destroy_index=int(s.pending[0]),repair_index=int(s.pending[1]),routes=copy.deepcopy(env.routes),removed=list(removed),repair=repairer,weight=env.preference.copy(),scalar_context=env.scalar_context(),seed=int(s.rng.integers(0,2**31-1)),deadline=min(deadline,time.perf_counter()+cfg.repair_seconds));tasks.append(task);used.append(v)
     if not tasks:break
     fs=[ex.submit(repair_job,t) for t in tasks];returned=[f.result(timeout=40) for f in fs];waves+=1
     for v,task,res in zip(used,tasks,returned):
      k=v['k'];s=v['s'];env=s.env;r=res['result'];stats[k]['attempts']+=1;v['step']+=1
      old_best=v['bestscore'];temperature=v['temperature0']*math.exp(math.log(.01)*(v['step']-1)/4);candidate_score=None;delta=None;u=None;prob=0.;accepted=False;improved=False;late=bool(r['timeout'] or res['finished']>deadline);t_checked=None
      if not late and time.perf_counter()<deadline:
       fn=env.api.repair;env.api.repair=lambda *args,_r=r,**kw:copy.deepcopy(_r)
       try:out=env.step(v['action'],task['repair'],task['seed'],deadline,removal_override=task['removed'] if not neural else None)
       finally:env.api.repair=fn
       late=bool(out['timeout']);candidate_score=out['candidate_score']
       if r['feasible'] and not late:
        physical(row,dict(routes=r['routes'],objectives=r['objectives']));physical_count+=1;t_checked=time.perf_counter();late=t_checked>=deadline
       if candidate_score is not None and not late:
        delta=candidate_score-v['before_score'];u=float(v['rng'].random())
       decision=v['transition'].transition(candidate_score,u if u is not None else float(v['rng'].random()),feasible=bool(r['feasible'] and candidate_score is not None),on_time=not late)
       accepted=decision['accepted'];prob=decision['probability']
       if accepted:
        env.routes=copy.deepcopy(r['routes']);env.objectives=np.asarray(r['objectives'],float);stats[k]['accepted']+=1;stats[k]['worse']+=int(delta>0)
       else:env.routes=v['before_routes'];env.objectives=v['before_obj']
       if accepted and candidate_score<old_best:
        improved=True;v['bestscore']=candidate_score;v['best_routes']=copy.deepcopy(env.routes);v['best_obj']=env.objectives.copy();stats[k]['improvements']+=1
       env.last_accepted=accepted;env.last_reward=max(0.,old_best-v['bestscore']);env.stagnation=0 if improved else v['before_stagnation']+1
       feedback=dict(out,accepted=improved,reward=env.last_reward);s.adaptive.update(*s.pending,feedback,v['bestscore'])
       if neural:s.hidden=s.cache_hidden.clone()
       s.step_in_session+=1;s.repair_seconds_since_refresh+=r['seconds']
       if not late and time.perf_counter()<deadline:
        # Only physically validated on-time discoveries enter the shared online archive.
        archive=union(archive,env.archive)
        if r['feasible']:point_events.append(dict(seconds=t_checked-timer,objectives=r['objectives']))
      else:late=True
      stats[k]['late']+=int(late);s.pending=None
      records.append(dict(gate_shift=v.get('gate_shift',0.),q_bias=v.get('q_bias',0.),direction_stall=stall[k],batch=len(batches),direction=k,step=v['step'],completed_seconds=None if t_checked is None else t_checked-timer,repair_finished_seconds=res['finished']-timer,worker_pid=res['pid'],q=v['action']['q'],mode=v['action'].get('mode'),destroy_index=None if neural else task['destroy_index'],repair=task['repair'],accepted=accepted,improved_best=improved,worse_accepted=bool(accepted and delta>0),before_score=v['before_score'],candidate_score=candidate_score,current_score=env.score(env.objectives),best_before=old_best,best_after=v['bestscore'],temperature=temperature,delta=delta,uniform=u,probability=prob,feasible=r['feasible'],late=late,best_reward=max(0.,old_best-v['bestscore']),repair_seconds=r['seconds'],candidate_objectives=r['objectives'] if r['feasible'] and not late else None))
     if time.perf_counter()-timer-last_log>=5:
      last_log=time.perf_counter()-timer;write('progress.json',dict(method=method,elapsed=last_log,batches=len(batches),attempts=len(records),archive_size=len(archive),workers=16));print(method,'elapsed',round(last_log,2),'attempts',len(records),'archive',len(archive),flush=True)
    # Deterministic parent/neighbor update in that batch's sampled direction order.
    decomp=alg.decomposition;decomp.dynamic_ideal=ideal;decomp.dynamic_span=span;decomp.reference=ref;decomp.penalty=1.
    committed=[]
    for v in lanes:
     if v['step']==5:
      k=v['k'];stats[k]['completed5']+=1
      if historical[k] is None or v['bestscore']<v.get('historical_before',float('inf')):
       historical[k]=v['best_obj'].copy();stall[k]=0
      else:stall[k]+=1
    for v in lanes:
     if time.perf_counter()>=deadline:break
     off=v['off'];off.set('X',np.asarray([Genome(copy.deepcopy(v['best_routes']),copy.deepcopy(v['best_routes']))],dtype=object));off.set('F',v['best_obj']);alg.ideal=np.minimum(alg.ideal,off.F);alg._replace(v['k'],off);committed.append(v['k'])
    batches.append(dict(index=len(batches),start_seconds=batch_start-timer,end_seconds=time.perf_counter()-timer,order=order,active_directions=[v['k'] for v in lanes],committed_directions=committed,waves=waves,local_steps={str(v['k']):v['step'] for v in lanes}))
    if time.perf_counter()<deadline:snapshots.append(dict(seconds=time.perf_counter()-timer,archive_objectives=[p['objectives'] for p in archive]))
   online_seconds=time.perf_counter()-timer
   audit_begin=time.perf_counter()
   for p in archive:physical(row,p)
   for p in alg.pop:physical(row,dict(routes=p.X[0].routes,objectives=p.F))
   assert all(r['completed_seconds'] is None or r['completed_seconds']<=BUDGET for r in records if r['candidate_objectives'] is not None)
   assert all(e['seconds']<=BUDGET for e in point_events),'Late archive point event'
   report=dict(method=method,bias_action_attempts=bias_calls,final_direction_stall=stall,iid=IID,seed=SEED,budget_seconds=BUDGET,online_seconds=online_seconds,cleanup_overrun_seconds=max(0.,online_seconds-BUDGET),initial_seconds=initial_seconds,initial_population=initial,initial_archive=initial_archive,archive=archive,population=[dict(routes=p.X[0].routes,objectives=p.F.tolist()) for p in alg.pop],records=records,batches=batches,per_direction=stats,adaptive_weights=[dict(direction=k,destroy=a.selector.destroy_weights.tolist(),repair=a.selector.repair_weights.tolist(),counts=a.count.tolist()) for k,a in enumerate(search.adaptive)],point_events=point_events,archive_snapshots=snapshots,physical_audit_passed=True,physical_checks_online=physical_count,offline_audit_seconds=time.perf_counter()-audit_begin,training=True,workers=16)
   
   if os.environ.get('TRAIN_CAPTURE'):write('training_states.json',captured)
   
   if os.environ.get('GLOBAL_ROLLOUT'):
    torch.save(dict(rows=training_rows,final_quality=quality(archive),budget=BUDGET,actual=online_seconds,batches=len(batches)),O/'global_rollout.pt')
   write('results/'+method+'.json',report);reports.append(report);print(method,'DONE',json.dumps(dict(online_seconds=online_seconds,attempts=len(records),batches=len(batches),archive=len(archive))),flush=True)
  verify_manifest();assert all(torch.equal(v,models[k].state_dict()[name]) for k,sd in enumerate(frozen) for name,v in sd.items())
  analyze(reports,pool,physical)
 finally:ex.shutdown(wait=True,cancel_futures=True)

def analyze(reports,pool,physical):
 import moocore,csv
 allpoints=np.unique(np.concatenate([np.asarray([p['objectives'] for p in r['archive']]) for r in reports]),axis=0);pf=allpoints[moocore.is_nondominated(allpoints)];ideal=pf.min(0);nadir=pf.max(0);span=nadir-ideal
 from solver.environment import Environment
 from solver.config import directions
 from run import load_config
 cfg=load_config(F/'evaluation_source/eval_config.json');cfg.project_root=str(F/'physical');cfg.step5_reference_penalty=1.
 env=Environment(pool,os.environ['CASE_IID'],cfg,directions()[15]);zero=span/env.scales<=1e-12
 def norm(x):return np.where(zero,0.,(np.asarray(x)-ideal)/np.where(zero,1.,span))
 reference=norm(pf);metrics=[]
 for r in reports:
  f=np.asarray([p['objectives'] for p in r['archive']]);x=norm(f)
  curves=[]
  for sec in np.linspace(0,r['budget_seconds'],7):
   points=[e['objectives'] for e in r['point_events'] if e['seconds']<=sec]
   h=0. if not points else float(moocore.hypervolume(norm(points),ref=[1.1]*5));curves.append(dict(seconds=sec,hv=h))
  hv=float(moocore.hypervolume(x,ref=[1.1]*5));assert abs(curves[-1]['hv']-hv)<1e-10
  z=(f-env.offset)/env.scales
  m=dict(method=r['method'],hv=hv,igd_plus=float(moocore.igd_plus(x,ref=reference)),fixed_box_hv=float(moocore.hypervolume(z,ref=env.box['reference'])),archive_size=len(f),online_seconds=r['online_seconds'],attempts=len(r['records']),batches=len(r['batches']),accepted=sum(d['accepted'] for d in r['per_direction']),best_improvements=sum(d['improvements'] for d in r['per_direction']),worse_accepted=sum(d['worse'] for d in r['per_direction']),late=sum(d['late'] for d in r['per_direction']),direction_calls=[d['calls'] for d in r['per_direction']],direction_attempts=[d['attempts'] for d in r['per_direction']],hv_curve=curves)
  metrics.append(m)
 write('analysis/reference.json',dict(ideal=ideal,nadir=nadir,zero_coordinates=np.flatnonzero(zero),reference_pf=pf,normalized_reference=[1.1]*5,definition='All stage-method single-run feasible archives common empirical ND; no old results mixed'))
 write('analysis/summary.json',dict(iid=os.environ['CASE_IID'],seed=int(os.environ['CASE_SEED']),metrics=metrics,scope='one-instance one-seed preliminary, newly trained aligned policy, no pvalue',reference='本阶段方法共同经验ND,HV(1.1)^5,shared normalization,zero span ratio<=1e-12'))
 audit=dict(passed=True,common_initial_population=all(r['initial_population']==reports[0]['initial_population'] for r in reports),all_archives_and_population_physical=True,no_late_candidates_in_archive=True,expected_searches=len(reports)==len(json.loads(os.environ['METHODS'])),same_workers16=True,weights_torch_equal=True,source_manifest_unchanged=True,empirical_hv_matches_event_curve=True,training=True)
 write('analysis/AUDIT.json',audit)
 (O/'analysis/REPORT.md').write_text('Three-IID single-seed development check; do not mix references between stages. See root PROTOCOL.md.',encoding='utf-8')
 with (O/'analysis/metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
  fields=['method','hv','igd_plus','fixed_box_hv','archive_size','online_seconds','attempts','batches','accepted','best_improvements','worse_accepted','late'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:m[k] for k in fields} for m in metrics)
 write('COMPLETE.json',dict(passed=True,searches=len(reports),audit_passed=True,report='analysis/REPORT.md',training=True))
 print('COMPLETE',json.dumps(metrics),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback;write('FAILURE.json',dict(error=traceback.format_exc()));raise

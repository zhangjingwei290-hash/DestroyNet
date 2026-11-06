import os
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
def novelty(env,archive,result):
 if not result['feasible'] or result['timeout']:return 0.
 p=np.asarray(result['objectives']);pts=np.asarray([x['objectives'] for x in archive])
 if np.any(np.all(pts<=p,axis=1)):return 0.
 return float(min(1.,np.linalg.norm((pts-p)/env.dynamic_span,axis=1).min()/math.sqrt(5)))
def episodes(m,pool,cfg,bank,w,seed,deadline,groups=4):
 slot=int(seed%16);startpoint=bank['population'][slot];out=[];m.cpu_head=NewHead(m)
 for rep in range(groups):
  if time.perf_counter()+.2>=deadline:break
  env=Environment(pool,bank['iid'],cfg,w,routes=startpoint['routes']);archive=copy.deepcopy(bank['archive']);env.archive=copy.deepcopy(archive)
  ref=env.offset+env.scales*np.asarray(env.box['reference']);env.dynamic_ideal,env.dynamic_span=dynamic_coordinates([a['objectives'] for a in archive],env.scales,ref)
  ad=Adaptive(len(env.q_values));initial=env.score(env.objectives);ad.best=initial;ss=Session(initial,initial,calibrated_temperature(.01,.5),calibrated_temperature(.0001,.5))
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
   bonus=novelty(env,archive,r) if valid else 0.;result=ss.transition(candidate,float(rng.random()),feasible=bool(r['feasible']),on_time=not r['timeout'] and time.perf_counter()<deadline)
   if result['archive_eligible']:env.archive=union(env.archive,[dict(routes=r['routes'],objectives=r['objectives'])]);archive=union(archive,env.archive)
   if result['accepted']:env.routes=copy.deepcopy(r['routes']);env.objectives=np.asarray(r['objectives'])
   env.last_removed=action['removed'];env.steps+=1;env.last_accepted=result['accepted'];env.last_reward=result['reward'];env.stagnation=ss.stagnation
   ad.update(*s.pending,dict(accepted=result['reward']>0,reward=result['reward'],environment_seconds=r['seconds'],feasible=bool(r['feasible']),timeout=not valid),ss.best)
   s.pending=None;s.hidden=torch.from_numpy(cache['hidden'].copy()).reshape(1,-1);s.step_in_session+=1;s.repair_seconds_since_refresh+=r['seconds']
   replay.update(search_state=features,set_summary=summ,reward=result['reward']+.01*bonus,scalar_reward=result['reward'],archive_reward=.01*bonus,value=m.cpu_head.value,terminated=result['terminated'],truncated=result['truncated'],iid=bank['iid'],phase=bank['stage'],slot=slot,group=rep)
   ep.append(replay)
   if result['truncated']:break
  if len(ep)==5 and ep[-1]['terminated']:out.append(ep)
 return out
def update(m,opt,eps):
 if len(eps)!=4:return None
 rows=[];advs=[];targets=[];starts=[]
 for ep in eps:
  v=torch.tensor([r['value'] for r in ep],device='cuda');r=torch.tensor([r['reward'] for r in ep],device='cuda')
  target,adv=returns_and_advantages(r,v,[t['terminated'] for t in ep],[t['truncated'] for t in ep],v.new_tensor(0.),gamma=1.,lam=1.)
  rows+=ep;advs.append(adv);targets.append(target);starts.append(float(target[0]-v[0]))
 for j in range(4):advs[j]-=sum(starts[k] for k in range(4) if k!=j)/3
 adv=torch.cat(advs);adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8);target=torch.cat(targets)
 old=torch.tensor([r['logprob_macro']+r['logprob_micro'] for r in rows],device='cuda');error=0.
 for epoch in range(2):
  terms=m.replay_terms(rows);lp=joint_logprob(terms)
  if epoch==0:
   error=float((lp.detach()-old).abs().max());assert error<2e-3,f'CPU/GPU joint density mismatch {error}'
  loss=ppo_loss(lp,old,adv,terms['value'],target,terms['entropy']);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
 return dict(transitions=len(rows),loss=float(loss.detach()),replay_error=error,scalar_gain=sum(r['scalar_reward'] for r in rows)/4,archive_gain=sum(r['archive_reward'] for r in rows)/4,value_mse=float((terms['value'].detach()-target).square().mean()),iid=rows[0]['iid'],stage=rows[0]['phase'])
def main():
 psutil.Process().cpu_affinity([0,1,2,3,4,5]);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261104);np.random.seed(20261104)
 # Wait for the predecessor's healthy active case to complete, without restarting it.
 old=json.loads((O/'PREDECESSOR_DRAIN.json').read_text());child=old['active_child'];before=time.time()
 while True:
  try:
   p=psutil.Process(child['pid'])
   if abs(p.create_time()-child['created'])>.02:break
  except psutil.NoSuchProcess:break
  write('progress.json',dict(phase='waiting_predecessor_case_boundary',seconds=time.time()-before));time.sleep(3)
 stage=child['stage'];case=P/'cases'/stage.split('_')[1]/stage.split('_')[2]
 assert json.loads((case/'analysis/AUDIT.json').read_text())['passed']
 write('PREDECESSOR_DRAINED.json',dict(case=str(case),passed=True,time=time.time(),coordinator_remains_suspended=True))
 start=time.perf_counter()-30.;end=start+900;cfg=load_config(F/'evaluation_source/eval_config.json');cfg.project_root=str(F/'physical');cfg.cpu_workers=0;cfg.step5_reference_penalty=1.;pool=Pool()
 banks={}
 for split,d,tw in [('train',0,0),('gate',1,2),('holdout',2,4)]:
  banks[split]=[json.loads((P/f'state_bank/v5p_train_{n}_d{d}_tw{tw}_0/state_{ph}.json').read_text()) for n in (50,150,250) for ph in ('early','middle','late')]
 m=NewPolicy(cfg).cuda();load=m.load_state_dict(torch.load(P/'training/meta.pt',map_location='cuda',weights_only=False)['model'],strict=False);assert all(k.startswith('rich_') for k in load.missing_keys) and not load.unexpected_keys
 def clone(actor):
  other=NewPolicy(cfg).cuda();other.load_state_dict(actor.state_dict());return other
 initial=clone(m);torch.save(dict(model=m.state_dict()),O/'training/initial.pt');history=[];valid=[]
 def validate(actor,name,split,expert,groups=2):
  result=[]
  for index,b in enumerate(banks[split]):
   seed=20264000+index*16+expert;np.random.seed(seed);torch.manual_seed(seed)
   eps=episodes(actor,pool,cfg,b,directions()[expert],seed,end,groups=groups);assert len(eps)==groups
   for ep in eps:
    returns=np.cumsum([r['reward'] for r in ep][::-1])[::-1]
    result.append(dict(checkpoint=name,split=split,expert=expert,iid=b['iid'],stage=b['stage'],rep=ep[0]['group'],scalar_gain=sum(r['scalar_reward'] for r in ep),archive_gain=sum(r['archive_reward'] for r in ep),return_total=sum(r['reward'] for r in ep),value_mse=float(np.mean([(r['value']-target)**2 for r,target in zip(ep,returns)]))))
  return result
 # Technical preflight uses fit states only; no extra solver run.
 eps=episodes(m,pool,cfg,banks['train'][0],directions()[7],20264007,end)
 opt=configure(m);metric=update(m,opt,eps);assert metric
 write('preflight/PASSED.json',dict(passed=True,joint_density_error=metric['replay_error'],rich_summary_dimension=len(eps[0][0]['set_summary']),same_start_group4=True,critic_full_condition=True,physical_objective_checks=True,training_device='cuda',probe_update=True))
 m.load_state_dict(initial.state_dict());rounds=0;meta_end=start+420;rng=np.random.default_rng(20261104)
 while time.perf_counter()+10<meta_end:
  base=rng.dirichlet(np.ones(5));actors=[]
  for task in range(5):
   actor=clone(m);opt=configure(actor);idx=(rounds*5+task)%9;eps=episodes(actor,pool,cfg,banks['train'][idx],np.roll(base,task),20265000+rounds*17+task,end);row=update(actor,opt,eps)
   if row:actors.append(actor);history.append(dict(phase='meta',round=rounds,**row))
  assert len(actors)==5;reptile_update(m,actors,.2);rounds+=1;write('progress.json',dict(phase='meta',rounds=rounds,seconds=time.perf_counter()-start))
 torch.save(dict(model=m.state_dict()),O/'training/meta.pt');actors=[clone(m) for _ in range(16)];opts=[configure(a) for a in actors];counts=[0]*16;adapt_end=min(time.perf_counter()+180,end-240)
 for cycle in range(27):
  if time.perf_counter()+8>=adapt_end:break
  for k,w in enumerate(directions()):
   if time.perf_counter()+1>=adapt_end:break
   eps=episodes(actors[k],pool,cfg,banks['train'][cycle%9],w,20266000+cycle*16+k,end);row=update(actors[k],opts[k],eps)
   if row:counts[k]+=1;history.append(dict(phase='adapt',direction=k,cycle=cycle,**row))
  write('progress.json',dict(phase='adapt',cycle=cycle,counts=counts,seconds=time.perf_counter()-start))
 gates=[];selected={}
 for k in range(16):
  write('progress.json',dict(phase='adapt_validation_gate',direction=k,seconds=time.perf_counter()-start))
  meta_rows=validate(m,'meta','gate',k);adapt_rows=validate(actors[k],'adapt','gate',k);valid+=meta_rows+adapt_rows
  mg=float(np.mean([r['return_total'] for r in meta_rows]));ag=float(np.mean([r['return_total'] for r in adapt_rows]));accept=ag>=mg
  selected[k]=actors[k] if accept else m;torch.save(dict(model=selected[k].state_dict(),selected='adapt' if accept else 'meta',gate_meta=mg,gate_adapt=ag),O/f'training/direction_{k:02d}.pt')
  gates.append(dict(direction=k,meta_return=mg,adapt_return=ag,selected='adapt' if accept else 'meta'))
 for k in (0,7,15):
  valid+=validate(initial,'initial','holdout',k);valid+=validate(m,'meta','holdout',k);valid+=validate(selected[k],'selected','holdout',k)
  write('training/validation.json',valid)
 frozen=dict(model={k:v.detach().clone() for k,v in initial.encoder.state_dict().items()})
 assert all(torch.equal(v,m.encoder.state_dict()[k]) for k,v in frozen['model'].items())
 assert time.perf_counter()<=end
 write('training/history.json',history);write('training/gates.json',gates);write('training/COMPLETE.json',dict(passed=True,seconds=time.perf_counter()-start,limit=900,meta_rounds=rounds,adapt_updates=counts,validation_rows=len(valid),gate_meta_retained=sum(g['selected']=='meta' for g in gates),encoder_unchanged=True))
 print('TRAINING COMPLETE',json.dumps(gates),flush=True)
if __name__=='__main__':
 try:
  (O/'training').mkdir(exist_ok=True);(O/'preflight').mkdir(exist_ok=True);main()
 except BaseException:write('FAILURE.json',dict(error=traceback.format_exc()));raise

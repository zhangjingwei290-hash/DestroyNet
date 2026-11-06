"""Frozen, counterfactual forward tests. No repair, search, gradient or weight update."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib,copy,math,csv,time
import numpy as np
import torch,psutil
from scipy.special import betainc
from scipy.stats import spearmanr
F=Path('D:/workspace/destroy-repair-V0.01-publication/versions/v5_step5_r2_frozen_20261102')
P=Path('D:/workspace/fuxian/DPR/development/sa5_richstate_v3_20261104')
STATE_ROOT=Path('D:/workspace/fuxian/DPR/development/sa5_aligned_9iid3seed_20261104')
O=Path(os.environ.get('DIAG_OUTPUT',str(Path(__file__).resolve().parent)));O.mkdir(parents=True,exist_ok=True)
EXPERTS=[int(x) for x in os.environ.get('DIAG_EXPERTS',','.join(map(str,range(16)))).split(',')];DRAWS=int(os.environ.get('DIAG_DRAWS','1024'))
sys.path[:0]=[str(P),str(F/'evaluation_source'),str(F/'evaluation_source/stage2_repo')]
from run import load_config
from network import NewPolicy,NewHead,summary_for
from session_contract import Session,calibrated_temperature
from aligned_cpu import struct_state
from stage1_mechanisms import dynamic_coordinates,normalize_physical
from stage2_structured import route_context,softmax,interpolate,route_logits
from solver.environment import Environment,Adaptive
from solver.config import directions,q_values
psutil.Process().cpu_affinity([14,15]);torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.manual_seed(20261104);np.random.seed(20261104)
cfg=load_config(F/'evaluation_source/eval_config.json');cfg.project_root=str(F/'physical');cfg.cpu_workers=0;cfg.step5_reference_penalty=1.
class Pool:
 def __init__(self):
  from solver.data import project_api
  self.api=project_api(F/'physical');self.boxes=json.loads((F/'evaluation_source/boxes.json').read_text());self.cache={}
 def load(self,iid):
  if iid not in self.cache:
   row=json.loads((F/f'data/test/{iid}.json').read_text());self.cache[iid]=(self.api.instance(row),row,self.boxes[iid],q_values(row['n_customers']))
  return self.cache[iid]
pool=Pool();weights=np.asarray(directions());sources=[];records=[];comparisons=[]
@torch.no_grad()
def refresh_cpu(m,obs):
 hidden,nodes,x,raw=m.encode([obs]);lo,w=m.node_logits_parts(nodes,x,raw,raw.new_full((1,),.04),[0]);hi,_=m.node_logits_parts(nodes,x,raw,raw.new_full((1,),.25),[0])
 return dict(hidden=hidden[0].numpy().copy(),values_lo=lo[0].numpy().copy(),values_hi=hi[0].numpy().copy(),physical_weights=w[0].numpy().copy(),refresh_raw=obs['context'].numpy().copy())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source(p):
 s=dict(path=str(p),sha256=sha(p));sources.append(s);return s
for p in [P/'SOURCE_MANIFEST.json',*[P/n for n in ('network.py','learning.py','session_contract.py','aligned_cpu.py')]]:source(p)
states=[]
for iid in ('50-2-4','150-2-4','250-0-0'):
 path=STATE_ROOT/f'cases/{iid}/20261104/results/neural_sa5_retrained.json';source(path);data=json.loads(path.read_text())
 arc=data['archive'];pts=np.asarray([a['objectives'] for a in arc]);base=data['initial_population'][15]
 scale=np.asarray(pool.load(iid)[2]['scale'] if 'scale' in pool.load(iid)[2] else np.ones(5))
 lo=pts.min(0);sp=np.maximum(pts.max(0)-lo,1.);idx=int(np.argmin(((pts-lo)/sp).mean(1)))
 states.extend([dict(iid=iid,label='initial_slot15',point=base,archive=data['initial_archive']),dict(iid=iid,label='final_balanced',point=arc[idx],archive=arc)])
 del data
def expected_q(mode,a,b,n,routes,rlog,draws):
 if mode in (0,1):
  lower=2 if mode==0 else 3;upper=min(max(12,math.floor(.22*n)),45) if mode==0 else min(max(10,math.floor(.35*n)),75)
  qs=np.arange(lower,upper+1);cuts=np.clip(np.log((qs[:-1]+.5)/lower)/np.log(upper/lower),0,1)
  probs=np.diff(np.r_[0.,betainc(a,b,cuts),1.]);return float(qs@probs)
 v=len(routes);upper=min(v,max(1,min(4,math.floor(.25*v))));pk=np.diff(betainc(a,b,np.arange(upper+1)/upper))
 means=[];lengths=np.asarray([len(r) for r in routes]);cap=min(int(np.rint(.45*n)),100)
 for k in range(1,upper+1):
  order=np.argsort(-(rlog[k-1][None,:]-np.log(draws)),axis=1,kind='stable')[:,:k]
  sums=lengths[order].sum(1);values=np.where(lengths[order[:,0]]>cap,lengths[order[:,0]],np.minimum(sums,cap));means.append(values.mean())
 return float(pk@means)
def extract(m,env,archive,state,raw,cache,physical,rc,draws):
 h=m.cpu_head;h.state=np.asarray(state,np.float32);h.summary=summary_for(env,archive)
 z=(env.objectives-env.dynamic_ideal)/env.dynamic_span
 with torch.no_grad():
  mode,a,b,_=h.outputs(cache['hidden'],struct_state(physical),raw,env.instance.n_customers,np.asarray([z.mean(),(env.preference*z).max()],np.float32))
  def t(x):return torch.as_tensor(np.asarray(x,dtype=np.float32)).reshape(1,-1)
  delta=m.customer_condition(t(cache['hidden']),t(h.state),t(h.summary))[0].numpy()
 n=env.instance.n_customers
 # Customer scores standardized to the same q/N=.10 in every condition.
 logits=(interpolate(cache['values_lo'],cache['values_hi'],.10*n,n)+normalize_physical(physical)@(cache['physical_weights']+delta)).astype(np.float32)
 upper=min(len(env.routes),max(1,min(4,math.floor(.25*len(env.routes)))))
 rlog=[]
 for k in range(1,upper+1):
  qc=int(np.clip(np.rint(k*n/len(env.routes)),2,n));cl=(interpolate(cache['values_lo'],cache['values_hi'],qc,n)+normalize_physical(physical)@(cache['physical_weights']+delta)).astype(np.float32)
  rlog.append(route_logits(cl,rc,raw[:5],h.params['heads.0.route_bilinear'],h.params['heads.0.route_score_scale']))
 probs=softmax(mode);eq=[expected_q(k,float(a[k]),float(b[k]),n,env.routes,rlog,draws) for k in range(3)]
 # Mode2 averages 1024 common PL draws per route-count bin; other mode expectations use exact Beta bin masses.
 return dict(p=probs,eq=float(probs@eq)/n,eq_by_mode=np.asarray(eq)/n,alpha=a,beta=b,cust=softmax(logits),logits=logits,summary=h.summary.copy())
def compare(base,other):
 top=max(1,len(base['cust'])//10);a=set(np.argsort(base['logits'])[-top:]);b=set(np.argsort(other['logits'])[-top:])
 return dict(mode_tv=float(np.abs(base['p']-other['p']).sum()/2),q_delta_pp=float((other['eq']-base['eq'])*100),customer_tv=float(np.abs(base['cust']-other['cust']).sum()/2),top10_overlap=len(a&b)/top,customer_spearman=float(spearmanr(base['logits'],other['logits']).statistic))
start=time.perf_counter()
for expert in EXPERTS:
 cp=P/f'training/direction_{expert:02d}.pt';source(cp);m=NewPolicy(cfg);m.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model']);m.eval();m.requires_grad_(False);m.cpu_head=NewHead(m)
 frozen={k:v.clone() for k,v in m.state_dict().items()}
 for si,st in enumerate(states):
  env=Environment(pool,st['iid'],cfg,weights[expert],routes=st['point']['routes']);env.archive=copy.deepcopy(st['archive'])
  ref=env.offset+env.scales*np.asarray(env.box['reference']);env.dynamic_ideal,env.dynamic_span=dynamic_coordinates([p['objectives'] for p in st['archive']],env.scales,ref)
  ad=Adaptive(len(env.q_values));ss=Session(env.score(env.objectives),env.score(env.objectives),calibrated_temperature(.01,.5),calibrated_temperature(.0001,.5))
  base_state=np.asarray(ss.state(.5),np.float32);obs=env.observation(torch.zeros(1,cfg.hidden_dim),'greedy',ad,150.,300.,150.,150.)
  with torch.no_grad():cache=refresh_cpu(m,obs)
  raw=obs['context'].numpy().copy();physical=obs['x'][1:,[18,17,19,12,13]].numpy().copy();rc=route_context(env.instance,env.routes,physical)
  draws=np.maximum(np.random.default_rng(1700+si).exponential(size=(DRAWS,len(env.routes))),np.finfo(float).tiny)
  base=extract(m,env,st['archive'],base_state,raw,cache,physical,rc,draws)
  def save(group,condition,out,**extra):
   row=dict(expert=expert,iid=st['iid'],state=st['label'],group=group,condition=str(condition),mode0=float(out['p'][0]),mode1=float(out['p'][1]),mode2=float(out['p'][2]),expected_q_ratio=out['eq'],summary=out['summary'].tolist(),**extra)
   records.append(row);comparisons.append(dict(expert=expert,iid=st['iid'],state=st['label'],group=group,condition=str(condition),**compare(base,out)))
  save('control','repeat',extract(m,env,st['archive'],base_state,raw,cache,physical,rc,draws))
  for k,w in enumerate(weights):
   env.preference=w.copy();r=raw.copy();r[:5]=w
   # Deployment-equivalent fresh-cache path with the same zero history at session entry.
   o=copy.deepcopy(obs);o['context']=torch.as_tensor(r)
   with torch.no_grad():c=refresh_cpu(m,o)
   out=extract(m,env,st['archive'],base_state,r,c,physical,rc,draws);save('preference',k,out,weight=w.tolist())
  env.preference=weights[expert].copy()
  pts=np.asarray([p['objectives'] for p in st['archive']]);dist=np.linalg.norm((pts-env.objectives)/env.dynamic_span,axis=1);order=np.argsort(dist)
  current=dict(routes=env.routes,objectives=env.objectives.tolist())
  sets={'singleton':[current],'full':st['archive'],'near16':[st['archive'][i] for i in order[:16]]+[current],'far16':[st['archive'][i] for i in order[-16:]]+[current]}
  for name,arc in sets.items():
   save('archive_summary',name,extract(m,env,arc,base_state,raw,cache,physical,rc,draws),archive_count=len(arc))
  for attempts in range(5):
   s=base_state.copy();s[0]=(5-attempts)/5
   save('steps_only',5-attempts,extract(m,env,st['archive'],s,raw,cache,physical,rc,draws))
   s[1]=.01**(attempts/4)
   save('steps_temperature',5-attempts,extract(m,env,st['archive'],s,raw,cache,physical,rc,draws))
  for fraction in (.9,.5,.1):
   s=base_state.copy();s[-1]=fraction;r=raw.copy();r[10]=fraction;r[11]=1-fraction
   save('global_stage',fraction,extract(m,env,st['archive'],s,r,cache,physical,rc,draws))
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in frozen.items())
 print('expert',expert,'complete',len(records),flush=True)
 del m
for item in sources:assert sha(Path(item['path']))==item['sha256']
(O/'responses.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
with (O/'comparisons.csv').open('w',newline='',encoding='utf-8-sig') as f:
 writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
summary={}
for group in sorted(set(r['group'] for r in comparisons)):
 rows=[r for r in comparisons if r['group']==group];summary[group]={k:{'median':float(np.median([r[k] for r in rows])),'max':float(np.max([r[k] for r in rows])),'min':float(np.min([r[k] for r in rows]))} for k in ('mode_tv','q_delta_pp','customer_tv','top10_overlap')}
(O/'summary.json').write_text(json.dumps(dict(groups=summary,rows=len(records),states=len(states),experts=EXPERTS,mode2_samples=DRAWS,seconds=time.perf_counter()-start),indent=2))
(O/'sources.json').write_text(json.dumps(sources,indent=2))
assert summary['control']['mode_tv']['max']==0 and summary['control']['customer_tv']['max']==0
(O/'AUDIT.json').write_text(json.dumps(dict(passed=True,forward_conditions=len(records),source_hashes_unchanged=True,all16_weights_unchanged=True,repair_calls=0,optimization_searches=0,training_updates=0,device='cpu',affinity=[14,15],history='zero at session entry; other step values are counterfactual, no fabricated GRU history',mode2_q='1024 common PL samples per route-count bin with the actual k-conditioned customer scores; Monte Carlo diagnostic estimate',archive_test='only three-number summary varied; dynamic coordinates and raw count held fixed to isolate new set pathway',global_test='cache held fixed, live global raw and adapter stage fields changed; local budget held fixed'),indent=2))
print(json.dumps(summary),flush=True)

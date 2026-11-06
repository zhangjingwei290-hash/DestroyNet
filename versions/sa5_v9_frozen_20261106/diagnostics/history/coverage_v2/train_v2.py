import training_core as t
import os,time,json,copy,random,collections
O=t.O

def main():
 t.psutil.Process().cpu_affinity([6,7]);t.torch.set_num_threads(1);t.torch.set_num_interop_threads(1);t.torch.manual_seed(20261104);t.np.random.seed(20261104);random.seed(20261104)
 end=time.perf_counter()+max(0,float(os.environ['TRAINING_DEADLINE_EPOCH'])-time.time());begin=time.perf_counter();assert end-begin>600
 cfg=t.load_config(t.F/'evaluation_source/eval_config.json');cfg.project_root=str(t.F/'physical');cfg.cpu_workers=0;cfg.step5_reference_penalty=1.;cfg.training_context_seconds=60.;cfg.collection_segments=4
 pool=t.Pool(cfg);banks=[json.loads(p.read_text()) for p in sorted((O/'state_bank').glob('*/state_*.json'))];assert len(banks)==27
 layers={}
 for bank in banks:
  n=pool.load(bank['iid'])[0].n_customers;layers.setdefault((n,bank['stage']),[]).append(bank)
 keys=[(n,stage) for n in (50,150,250) for stage in ('early','middle','late')];assert set(keys)==set(layers) and all(len(v)==3 for v in layers.values())
 for v in layers.values():v.sort(key=lambda x:x['iid'])
 meta=t.NewPolicy(cfg).cuda();meta.load_state_dict(t.torch.load(t.F/'checkpoints/direction_15.pt',map_location='cuda',weights_only=False)['model'],strict=False);t.configure(meta)
 initial={k:v.detach().cpu().clone() for k,v in meta.state_dict().items()};t.torch.save(dict(model=initial),O/'training/initial.pt')
 rng=t.np.random.default_rng(20261104);history=[];coverage=[]
 def choose(index,offset=0):
  bank=layers[keys[index%9]][(index//9+offset)%3];pool.current_bank=bank;return bank
 def collect(actor,index,w,cut,offset=0):
  b=choose(index,offset);eps=t.collect(actor,pool,cfg,b['iid'],w,int(rng.integers(2**31)),cut)
  for ep in eps:
   assert len(ep)==5 and ep[-1]['terminated']
   coverage.append(dict(iid=b['iid'],n=pool.load(b['iid'])[0].n_customers,stage=b['stage'],steps=len(ep),nonzero_history=sum(bool(t.np.linalg.norm(x['hidden'])>0) for x in ep)))
  return eps
 meta_cut=min(begin+540,end-600);rounds=0
 while time.perf_counter()+15<meta_cut:
  base=rng.dirichlet(t.np.ones(5));actors=[]
  for task in range(5):
   actor=copy.deepcopy(meta);opt=t.configure(actor);eps=collect(actor,rounds*5+task,t.np.roll(base,task),meta_cut);metric=t.update(actor,opt,eps)
   if metric:actors.append(actor);history.append(dict(phase='meta',round=rounds,task=task,**metric))
  if len(actors)!=5:break
  t.reptile_update(meta,actors,.2);rounds+=1;t.write('training/progress.json',dict(phase='meta',rounds=rounds,seconds=time.perf_counter()-begin))
 assert rounds>=2
 t.torch.save(dict(model=meta.state_dict(),rounds=rounds),O/'training/meta.pt')
 actors=[copy.deepcopy(meta) for _ in range(16)];opts=[t.configure(a) for a in actors];counts=[0]*16;acov=[set() for _ in range(16)];adapt_cut=min(time.perf_counter()+420,end-180)
 # Round-robin prevents the final experts losing all their adaptation budget.
 for cycle in range(27):
  if time.perf_counter()+15>=adapt_cut:break
  for k,w in enumerate(t.directions()):
   if time.perf_counter()+3>=adapt_cut:break
   eps=collect(actors[k],cycle,w,adapt_cut,offset=k);metric=t.update(actors[k],opts[k],eps)
   if metric:counts[k]+=1;acov[k].add(keys[cycle%9]);history.append(dict(phase='adapt',direction=k,cycle=cycle,**metric))
  t.write('training/progress.json',dict(phase='adapt',cycle=cycle,counts=counts,coverage_layers=[len(s) for s in acov],seconds=time.perf_counter()-begin))
 for k,a in enumerate(actors):t.torch.save(dict(model=a.state_dict(),updates=counts[k]),O/f'training/direction_{k:02d}.pt')
 t.write('training/COVERAGE.json',dict(meta_rounds=rounds,direction_updates=counts,direction_layers=[sorted(s) for s in acov],episodes=coverage))
 assert all(len(s)==9 for s in acov),'Every expert must cover all three scales x three real phases; no evaluation if incomplete'
 # Fixed fit-state validation, same seeds and restored RNG per checkpoint. No optimizer or gradients.
 cfg.collection_segments=1;validation=[]
 for name in ('initial','meta','adapt'):
  for k in (0,7,15):
   cp='initial.pt' if name=='initial' else ('meta.pt' if name=='meta' else f'direction_{k:02d}.pt')
   m=t.NewPolicy(cfg).cuda();m.load_state_dict(t.torch.load(O/'training'/cp,map_location='cuda',weights_only=False)['model']);m.eval();m.requires_grad_(False)
   for index in range(9):
    assert time.perf_counter()+2<end,'Validation exceeded remaining total training cap'
    b=choose(index,offset=1);seed=20261104+index*16+k;t.np.random.seed(seed);t.torch.manual_seed(seed)
    before=time.perf_counter();eps=t.collect(m,pool,cfg,b['iid'],t.directions()[k],seed,end)
    assert len(eps)==1 and len(eps[0])==5,'Fixed validation requires a complete five-step episode'
    validation.append(dict(checkpoint=name,direction=k,iid=b['iid'],stage=b['stage'],best_gain=sum(x['reward'] for x in eps[0]),improved=any(x['reward']>0 for x in eps[0]),seconds=time.perf_counter()-before,steps=5))
  t.write('training/checkpoint_validation.json',dict(rows=validation,fit_only=True,common_start_states=True,not_unseen_generalization=True))
 t.write('training/COMPLETE.json',dict(passed=True,meta_rounds=rounds,direction_updates=counts,validation_rows=len(validation),history=history,seconds=time.perf_counter()-begin,fit_only=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback;t.write('training/FAILURE.json',dict(error=traceback.format_exc()));raise

import collector as t
import copy,time,json,hashlib,os,traceback
from learning import ppo_loss,joint_logprob,returns_and_advantages
def update(m,opt,eps,kind):
 assert len(eps)==4 and all(len(ep)==5 for ep in eps)
 rows=sum(eps,[]);old=t.torch.tensor([r['logprob_macro']+r['logprob_micro'] for r in rows],device='cuda');targets=[];advs=[]
 if kind=='rloo':
  returns=t.torch.tensor([sum(r['scalar_reward'] for r in ep) for ep in eps],device='cuda');group_adv=(4*returns-returns.sum())/3
  adv=group_adv.repeat_interleave(5);target=t.torch.zeros_like(adv)
 else:
  for ep in eps:
   values=t.torch.tensor([r['value'] for r in ep],device='cuda');rewards=t.torch.tensor([r['scalar_reward'] for r in ep],device='cuda')
   target0,adv0=returns_and_advantages(rewards,values,[r['terminated'] for r in ep],[r['truncated'] for r in ep],values.new_tensor(0.));targets.append(target0);advs.append(adv0)
  target=t.torch.cat(targets);adv=t.torch.cat(advs)
 raw_variance=float(adv.var(unbiased=False));adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
 error=0.
 for epoch in range(2):
  terms=m.replay_terms(rows);lp=joint_logprob(terms)
  if epoch==0:error=float((lp.detach()-old).abs().max());assert error<2e-3,f'Joint density error {error}'
  loss=ppo_loss(lp,old,adv,terms['value'],target,terms['entropy'],value_weight=0. if kind=='rloo' else .5);opt.zero_grad();loss.backward();t.torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
 return dict(kind=kind,transitions=20,loss=float(loss.detach()),sampling_replay_error=error,mean_return=sum(r['scalar_reward'] for r in rows)/4,raw_advantage_variance=raw_variance,positive_trajectories=sum(any(r['scalar_reward']>0 for r in ep) for ep in eps),critic_used=kind!='rloo')
def main():
 O=t.O;t.psutil.Process().cpu_affinity([0,1,2,3,4,5]);t.torch.set_num_threads(1);t.torch.set_num_interop_threads(1);t.torch.manual_seed(20261104);t.np.random.seed(20261104)
 start=time.perf_counter();end=start+900;cfg=t.load_config(t.F/'evaluation_source/eval_config.json');cfg.project_root=str(t.F/'physical');cfg.cpu_workers=0;cfg.step5_reference_penalty=1.;pool=t.Pool()
 banks=[json.loads((t.P/f'state_bank/v5p_train_{n}_d0_tw0_0/state_{phase}.json').read_text()) for n in (50,150,250) for phase in ('early','middle','late')]
 holdout=[json.loads((t.P/f'state_bank/v5p_train_{n}_d2_tw4_0/state_{phase}.json').read_text()) for n in (50,150,250) for phase in ('early','middle','late')]
 actors={};opts={};frozen={};counts={k:[0]*16 for k in ('ppo','rloo')};history=[]
 for kind in ('ppo','rloo'):
  actors[kind]=[];opts[kind]=[];(O/f'training/{kind}').mkdir(parents=True,exist_ok=True)
  for k in range(16):
   m=t.NewPolicy(cfg).cuda();m.load_state_dict(t.torch.load(t.P/f'training/direction_{k:02d}.pt',map_location='cuda',weights_only=False)['model']);opt=t.configure(m)
   if kind=='rloo':
    for p in m.adapter.value.parameters():p.requires_grad_(False)
   actors[kind].append(m);opts[kind].append(opt)
   if kind=='rloo':frozen[k]={name:v.detach().clone() for name,v in m.state_dict().items() if name.startswith(('encoder.','adapter.value.'))}
 def persist(cycle):
  for kind in actors:
   for k,m in enumerate(actors[kind]):t.torch.save(dict(model=m.state_dict(),kind=kind,updates=counts[kind][k],cycle=cycle),O/f'training/{kind}/direction_{k:02d}.pt')
  t.write('training/history.json',history);t.write('training/counts.json',counts)
 try:
  for cycle,b in enumerate(banks):
   # Complete paired cycle or preserve failure; never quietly compare unequal update counts.
   if time.perf_counter()+20>end-120:raise RuntimeError('Training time ceiling reached before matched9 cycles')
   for k in range(16):
    seed=20268000+cycle*16+k
    for kind in ('ppo','rloo'):
     eps=t.episodes(actors[kind][k],pool,cfg,b,t.directions()[k],seed,end-120);assert len(eps)==4
     row=update(actors[kind][k],opts[kind][k],eps,kind);counts[kind][k]+=1;history.append(dict(cycle=cycle,direction=k,iid=b['iid'],stage=b['stage'],**row))
   persist(cycle);t.write('progress.json',dict(phase='matched_gpu_training',cycle=cycle+1,total_cycles=9,counts=counts,seconds=time.perf_counter()-start));print('CYCLE',cycle+1,'seconds',time.perf_counter()-start,flush=True)
  assert all(x==9 for c in counts.values() for x in c)
  valid=[]
  for kind in ('initial','ppo','rloo'):
   for k in (0,7,15):
    m=actors[kind][k] if kind!='initial' else t.NewPolicy(cfg).cuda()
    if kind=='initial':m.load_state_dict(t.torch.load(t.P/f'training/direction_{k:02d}.pt',map_location='cuda',weights_only=False)['model'])
    m.eval();m.requires_grad_(False)
    for index,b in enumerate(holdout):
     seed=20269000+index*16+k;t.np.random.seed(seed);t.torch.manual_seed(seed);eps=t.episodes(m,pool,cfg,b,t.directions()[k],seed,end,groups=1);assert len(eps)==1
     valid.append(dict(kind=kind,direction=k,iid=b['iid'],stage=b['stage'],scalar_gain=sum(r['scalar_reward'] for r in eps[0]),steps=5))
    t.write('training/validation.json',valid)
  for k,m in enumerate(actors['rloo']):assert all(t.torch.equal(v,m.state_dict()[name]) for name,v in frozen[k].items())
  t.write('training/COMPLETE.json',dict(passed=True,counts=counts,updates=len(history),transitions=sum(r['transitions'] for r in history),seconds=time.perf_counter()-start,limit_seconds=900,validation_rows=len(valid),rloo_critic_and_encoder_unchanged=True,architecture='exact_v2',reward='scalar_best_gain_only',scope='Matched continued adaptation; no repeated meta-training'))
 finally:
  t.write('training/history.json',history);t.write('training/counts.json',counts)
if __name__=='__main__':
 try:main()
 except BaseException:t.write('FAILURE.json',dict(error=traceback.format_exc()));raise

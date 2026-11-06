import collector as t
import json,time,os,hashlib
from network import load_base
from learning import joint_logprob,returns_and_advantages,ppo_loss
def main():
 t.psutil.Process().cpu_affinity([0,1,2,3,4,5]);t.torch.set_num_threads(1);t.torch.set_num_interop_threads(1);t.torch.manual_seed(20261105)
 start=time.perf_counter();deadline=start+900;cfg=t.load_config(t.F/'evaluation_source/eval_config.json');cfg.project_root=str(t.F/'physical');cfg.cpu_workers=0;cfg.step5_reference_penalty=1.;pool=t.Pool();models=[]
 for k in range(16):
  m=t.NewPolicy(cfg).cuda();load_base(m,t.P/f'training/direction_{k:02d}.pt');models.append(m)
 shared=models[0].stall_gate
 for m in models:m.stall_gate=shared
 opt=t.configure(models[0])
 for m in models[1:]:
  for name,p in m.named_parameters():p.requires_grad_(name.startswith('stall_gate.'))
 frozen=[{n:v.detach().clone() for n,v in m.state_dict().items() if not n.startswith('stall_gate.')} for m in models]
 banks={n:json.loads((t.O/f'collection/train/{n}/training_states.json').read_text()) for n in (50,150,250)};history=[]
 (t.O/'training').mkdir(exist_ok=True)
 for cycle in range(18):
  n=(50,150,250)[cycle%3]
  for k,m in enumerate(models):
   rows=[b for b in banks[n] if b['direction']==k];assert rows
   # Deterministic real-state strata: alternate zero/positive stall when both exist.
   eligible=[b for b in rows if (b['cross_stall']>0)==bool((cycle//3)%2)] or rows
   b=eligible[(cycle//6)%len(eligible)]
   if time.perf_counter()>deadline-30:raise RuntimeError('Training ceiling before declared288 updates')
   eps=t.episodes(m,pool,cfg,b,t.directions()[k],20271100+cycle*16+k,deadline);assert len(eps)==4 and all(len(e)==5 for e in eps)
   batch=sum(eps,[]);old=t.torch.tensor([x['logprob_macro']+x['logprob_micro'] for x in batch],device='cuda');targets=[];advantages=[]
   for ep in eps:
    vals=t.torch.tensor([x['value'] for x in ep],device='cuda');reward=t.torch.tensor([x['scalar_reward'] for x in ep],device='cuda');tar,adv=returns_and_advantages(reward,vals,[x['terminated'] for x in ep],[x['truncated'] for x in ep],vals.new_tensor(0.));targets.append(tar);advantages.append(adv)
   target=t.torch.cat(targets);adv=t.torch.cat(advantages);adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
   for epoch in range(2):
    terms=m.replay_terms(batch);lp=joint_logprob(terms)
    if epoch==0:
     err=float((lp.detach()-old).abs().max());assert err<2e-3,f'density mismatch {err}'
    loss=ppo_loss(lp,old,adv,terms['value'],target,terms['entropy'],value_weight=0.);opt.zero_grad();loss.backward();t.torch.nn.utils.clip_grad_norm_(shared.parameters(),1.);opt.step()
   history.append(dict(cycle=cycle,direction=k,n=n,cross_stall=b['cross_stall'],bank_seconds=b['seconds'],mean_gain=sum(x['scalar_reward'] for x in batch)/4,loss=float(loss.detach()),density_error=err))
  t.torch.save(shared.state_dict(),t.O/'training/gate.pt');t.write('training/history.json',history);t.write('progress.json',dict(phase='train_gate',cycle=cycle+1,total=18,updates=len(history),seconds=time.perf_counter()-start));print('CYCLE',cycle+1,'elapsed',time.perf_counter()-start,flush=True)
 for m,sd in zip(models,frozen):assert all(t.torch.equal(v,m.state_dict()[n]) for n,v in sd.items())
 validation=[]
 for n in (50,150,250):
  states=json.loads((t.O/f'collection/heldout/{n}/training_states.json').read_text())
  chosen=t.np.linspace(0,len(states)-1,min(12,len(states))).astype(int)
  for index in chosen:
   b=states[int(index)];k=b['direction'];m=models[k];learned={key:v.detach().clone() for key,v in shared.state_dict().items()}
   for kind in ('trained','zero'):
    shared.load_state_dict(learned)
    if kind=='zero':
     with t.torch.no_grad():shared.body[-1].weight.zero_();shared.body[-1].bias.zero_()
    eps=t.episodes(m,pool,cfg,b,t.directions()[k],20281000+int(index),deadline,groups=4);assert len(eps)==4
    for rep,e in enumerate(eps):validation.append(dict(n=n,state=int(index),stall=b['cross_stall'],kind=kind,rep=rep,gain=sum(x['scalar_reward'] for x in e),q_ratio=sum(x['action']['q']/n for x in e)/5))
   shared.load_state_dict(learned);t.write('training/validation.json',validation)
 t.write('training/COMPLETE.json',dict(passed=True,updates=len(history),transitions=len(history)*20,seconds=time.perf_counter()-start,all_v2_expert_weights_unchanged=True,real_states=True,validation_trajectories=len(validation),positive_stall_training_updates=sum(h['cross_stall']>0 for h in history),mean_objective=True))
if __name__=='__main__':main()

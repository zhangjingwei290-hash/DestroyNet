"""Recover lost two completed PPO updates from preserved traces; no new trajectories."""
from stage2_common import ROOT,read,write,load_cfg,FitPool,reset
import time,gzip,pickle,math

def main():
 import psutil,numpy as np,torch
 from solver.model import Policy
 from solver.training import optimizer_for,frozen_state,update
 from stage2_schedule import set_lr,schedules
 from solver.environment import Environment,Adaptive
 from stage2_cpu import CpuHead,struct_state,sample
 from stage2_structured import route_context
 psutil.Process().cpu_affinity([0,1]);torch.set_num_threads(1);torch.set_num_interop_threads(1);reset(20261024)
 incident=ROOT/'engineering/replay_guard_direction05_batch003';cfg=load_cfg();model=Policy(cfg).cuda();model.load_state_dict(torch.load(ROOT/'stage2_training/direction_initialization/direction_05.pt',map_location='cuda',weights_only=False)['model']);opt=optimizer_for(model);frozen=frozen_state(model);reconstructed=[]
 for batch in (1,2):
  with gzip.open(ROOT/'stage2_training/trace'/f'adapt05_{batch:05d}.pkl.gz','rb') as f:entry=pickle.load(f)
  set_lr(opt,batch-1,90);metric=update(model,opt,entry['episodes'],cfg,float('inf'),frozen)
  reconstructed.append(dict(batch=batch,old_loss=entry['metric']['loss'],reconstructed_loss=metric['loss'],loss_delta=abs(entry['metric']['loss']-metric['loss']),macro_error=metric['initial_macro_ratio_error'],micro_error=metric['initial_micro_ratio_error']))
  assert abs(entry['metric']['loss']-metric['loss'])<1e-8,reconstructed
 torch.save(dict(model=model.actor_state(),optimizer=opt.state_dict(),updates=2,direction=5,optimizer_steps=4),incident/'recovered_direction05_batch002.pt')
 failure=torch.load(ROOT/'stage2_training/failed_training_state.pt',map_location='cpu',weights_only=False)
 last=torch.load(ROOT/'stage2_training/adaptation_boundaries/direction_15_batch_090.pt',map_location='cpu',weights_only=False)
 write(incident/'RECOVERY_INSPECTION.json',dict(reconstructed=reconstructed,failed_state_keys=list(failure),boundary_keys=list(last),scheduled_third=schedules()['adaptation']['5'][2],latest_advances=failure['progressive_pool']['advances'][-12:]))
 # Current preserved FIT state is used only as a finite diagnostic observation, not optimization.
 pool=FitPool(cfg);cpu=CpuHead(model);rng=np.random.default_rng(20261024);errors=[];examples=[]
 target=schedules()['adaptation']['5'][2]['iid'];slots=[s for s in failure['progressive_pool']['slots'] if s['target']==5 and s['iid']==target]
 slot=slots[0];env=Environment(pool,target,cfg,slot['preference'],slot['routes']);env.archive=slot['archive']
 from stage1_mechanisms import dynamic_coordinates
 env.dynamic_ideal,env.dynamic_span=dynamic_coordinates([p['objectives'] for p in env.archive],env.scales,env.offset+env.scales*np.asarray(env.box['reference']))
 for stag in (0,1,4,8,20):
  env.stagnation=stag;obs=env.observation(torch.tensor(slot['hidden']),'greedy',Adaptive(len(env.q_values)),100.,env.instance.n_customers,2.5,2.5)
  cache=model.refresh(obs,0,True);physical=obs['x'][1:,[18,17,19,12,13]].numpy();raw=obs['context'].numpy();rc=route_context(env.instance,env.routes,physical);z=(env.objectives-env.dynamic_ideal)/env.dynamic_span;position=np.asarray([z.mean(),(env.preference*z).max()],dtype=np.float32)
  for mode in range(3):
   for u in (.00001,.01,.5,.99,.99999):
    action,state=sample(cache,physical,raw,env.instance.n_customers,rng,env.neighbors,cpu,rc,position,0,mode,u);result=model.replay_terms([state]);macro=float(result['logprob_macro'].detach());micro=float(result['logprob_micro'].detach());me=abs(math.exp(macro-state['logprob_macro'])-1);mi=abs(math.exp((micro-state['logprob_micro'])/state['effective_selection_steps'])-1)
    ae=max(abs(float(result[k].detach())-state[k]) for k in ('alpha','beta'));le=float((result['logits'][0].detach().cpu()-torch.tensor(state['logits'])).abs().max());pe=float((result['mode_probabilities'][0].detach().cpu()-torch.tensor(state['mode_probabilities'])).abs().max())
    if mode==2:le=max(le,float((result['route_logits'][0,:len(env.routes)].detach().cpu()-torch.tensor(state['route_logits'])).abs().max()))
    errors.append([me,mi,ae,le,pe]);examples.append(dict(stag=stag,mode=mode,u=u,errors=[me,mi,ae,le,pe],alpha=state['alpha'],beta=state['beta']))
 write(incident/'FINITE_DIAGNOSTIC.json',dict(reconstructed=reconstructed,target_iid=target,max_errors=np.max(errors,axis=0).tolist(),columns=['macro_ratio','micro_ratio','alpha_beta','logits','mode_probs'],worst_examples=sorted(examples,key=lambda r:max(r['errors']),reverse=True)[:15],new_trajectories=0,diagnostic_optimizer_steps=4,counted_formal_steps_unchanged=True))
 print(read(incident/'FINITE_DIAGNOSTIC.json'),flush=True)
if __name__=='__main__':main()

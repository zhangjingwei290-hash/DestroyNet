"""Three bounded FIT-only windows, no optimizer, serialize every replay discrepancy."""
from stage2_common import ROOT,read,write,load_cfg,FitPool,reset
import time

def main():
 import psutil,numpy as np,torch
 from solver.model import Policy
 from solver.config import directions
 from stage2_runtime import Runtime
 from stage2_state_pool import StatePool
 psutil.Process().cpu_affinity([0,1]);torch.set_num_threads(1);torch.set_num_interop_threads(1);reset(20261024)
 folder=ROOT/'engineering/replay_guard_direction05_batch003';cfg=load_cfg();pool=FitPool(cfg);model=Policy(cfg).cuda();model.load_state_dict(torch.load(folder/'recovered_direction05_batch002.pt',map_location='cuda',weights_only=False)['model']);failure=torch.load(ROOT/'stage2_training/failed_training_state.pt',map_location='cpu',weights_only=False);store=StatePool(pool);store.restore(failure['progressive_pool']);runtime=Runtime();rng=np.random.default_rng(20261024);windows=[];maxima=np.zeros(5);bad=[]
 try:
  for window in range(3):
   episodes,p=runtime.collect(model,pool,{'progressive_pool':store},'v5p_train_50_d1_tw4_3',directions(),5,0,int(rng.integers(2**31-1)),rng,time.perf_counter()+90.,f'finite_diagnostic:{window}',require_mature=True);assert p['valid_complete_group'];windows.append(dict(episodes=episodes,provenance=p))
   states=[t for ep in episodes for t in ep]
   for i in range(0,len(states),cfg.microbatch):
    ss=states[i:i+cfg.microbatch];out=model.replay_terms(ss);oldmacro=out['logprob_macro'].new_tensor([t['logprob_macro'] for t in ss]);oldmicro=out['logprob_micro'].new_tensor([t['logprob_micro'] for t in ss]);eff=out['logprob_micro'].new_tensor([t['effective_selection_steps'] for t in ss]);a=(out['logprob_macro']-oldmacro).detach().exp();b=((out['logprob_micro']-oldmicro)/eff).detach().exp()
    for j,t in enumerate(ss):
     ae=max(abs(float(out[k][j].detach())-t[k]) for k in ('alpha','beta'));le=float((out['logits'][j].detach().cpu()-torch.tensor(t['logits'])).abs().max());pe=float((out['mode_probabilities'][j].detach().cpu()-torch.tensor(t['mode_probabilities'])).abs().max())
     if t['mode']==2:le=max(le,float((out['route_logits'][j,:len(t['routes'])].detach().cpu()-torch.tensor(t['route_logits'])).abs().max()))
     errors=np.asarray([abs(float(a[j])-1),abs(float(b[j])-1),ae,le,pe]);maxima=np.maximum(maxima,errors)
     if errors[0]>1e-4 or errors[1]>1e-4 or any(errors[2:]>=5e-5):bad.append(dict(window=window,state_index=i+j,mode=t['mode'],u=t['u'],errors=errors.tolist(),alpha=t['alpha'],beta=t['beta']))
   print('window',window,'states',len(states),'maxima',maxima.tolist(),'bad',len(bad),flush=True)
  torch.save(dict(model=model.actor_state(),windows=windows,maxima=maxima,bad=bad),folder/'live_replay_diagnostic.pt')
  write(folder/'LIVE_REPLAY_DIAGNOSTIC.json',dict(windows=3,trajectories=sum(len(w['episodes']) for w in windows),transitions=sum(len(ep) for w in windows for ep in w['episodes']),maxima=maxima.tolist(),bad=bad,optimizer_steps=0,formal_counts_unchanged=True))
 finally:runtime.close()
if __name__=='__main__':
 import multiprocessing as mp
 mp.freeze_support();main()

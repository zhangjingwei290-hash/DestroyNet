"""Retained actual replay corpus plus nearconstant reduction regression; no update."""
from stage2_common import ROOT,read,write,load_cfg,reset

def main():
 import psutil,numpy as np,torch
 from solver.model import Policy
 from stage2_cpu import CpuHead,struct_state
 psutil.Process().cpu_affinity([0,1]);torch.set_num_threads(1);torch.set_num_interop_threads(1);reset(20261024)
 folder=ROOT/'engineering/replay_guard_direction05_batch003';data=torch.load(folder/'live_replay_diagnostic.pt',map_location='cpu',weights_only=False);model=Policy(load_cfg()).cuda();model.load_state_dict(data['model']);cpu=CpuHead(model);maxima=np.zeros(5);count=0
 for window in data['windows']:
  states=[t for ep in window['episodes'] for t in ep]
  for i in range(0,len(states),32):
   chunk=states[i:i+32];out=model.replay_terms(chunk)
   for j,t in enumerate(chunk):
    a=(out['logprob_macro'][j].detach()-out['logprob_macro'].new_tensor(t['logprob_macro'])).exp();b=((out['logprob_micro'][j].detach()-out['logprob_micro'].new_tensor(t['logprob_micro']))/t['effective_selection_steps']).exp()
    e=[abs(float(a)-1),abs(float(b)-1),max(abs(float(out[k][j].detach())-t[k]) for k in ('alpha','beta')),float((out['logits'][j].detach().cpu()-torch.tensor(t['logits'])).abs().max()),float((out['mode_probabilities'][j].detach().cpu()-torch.tensor(t['mode_probabilities'])).abs().max())]
    if t['mode']==2:e[3]=max(e[3],float((out['route_logits'][j,:len(t['routes'])].detach().cpu()-torch.tensor(t['route_logits'])).abs().max()))
    maxima=np.maximum(maxima,e);count+=1
 assert maxima[0]<=1e-4 and maxima[1]<=1e-4 and all(maxima[2:]<5e-5),maxima
 probes=[]
 for value in (.1,.3,.7,1.1,3.3,7.7):
  x=np.full((50,5),np.float32(value));x[0,0]=np.nextafter(np.float32(value),np.float32(100));p=torch.tensor(x,device='cuda')[None].double();mu=p.mean(1);sd=p.std(1,unbiased=False);cv=(sd/p.abs().mean(1).clamp_min(1e-4)).clamp(0,4)/4;peak=((p.amax(1)-mu)/sd.clamp_min(1e-4)).clamp(0,6)/6;share=(p.topk(5,dim=1).values.abs().sum(1)/p.abs().sum(1).clamp_min(1e-4)).clamp(0,1);state=torch.cat((cv,peak,share),-1).float()[0].cpu().numpy();error=float(abs(state-struct_state(x)).max());assert error<1e-7;probes.append(dict(value=value,error=error))
 write(folder/'PRECISION_REPAIR_VALIDATION.json',dict(passed=True,actual_retained_transitions=count,max_errors=maxima.tolist(),nearconstant_probes=probes,network_fp32=True,changed_only_nonlearned_physical_reduction_precision=True,thresholds_unchanged=True,optimizer_steps=0,causal_limit='Original offending transition was not retained; this reproduces and fixes an independently observed reduction inconsistency, without claiming the original trigger is proven'))
 print(read(folder/'PRECISION_REPAIR_VALIDATION.json'),flush=True)
if __name__=='__main__':main()

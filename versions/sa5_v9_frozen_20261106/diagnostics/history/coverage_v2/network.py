import numpy as np
import torch
from solver.model import Policy
from stage2_cpu import CpuHead
from learning import SearchStateAdapter
from torch.distributions import Beta
from solver.model import route_features_torch,prefix_terms_torch
class NewPolicy(Policy):
 def __init__(self,cfg):
  super().__init__(cfg);self.adapter=SearchStateAdapter(cfg.hidden_dim,3);self.search_state=None;self.set_summary=None;self.customer_delta=torch.nn.Linear(cfg.hidden_dim,5);torch.nn.init.zeros_(self.customer_delta.weight);torch.nn.init.zeros_(self.customer_delta.bias)
 def q_from_physical(self,hidden,physical,raw,ids,dynamic_position):
  mode,a,b,mu=super().q_from_physical(hidden,physical,raw,ids,dynamic_position)
  out=self.adapter(hidden,self.search_state,self.set_summary,mode,a,b)
  return out['mode_logits'],out['alpha'],out['beta'],mu
 def replay_terms(self,states):
  self.search_state=torch.tensor(np.stack([s['search_state'] for s in states]),device=self.device)
  self.set_summary=torch.tensor(np.stack([s['set_summary'] for s in states]),device=self.device)
  out=self.replay_with_customers(states)
  hidden=torch.tensor(np.stack([s['hidden'] for s in states]),device=self.device)
  z=self.adapter.fusion(torch.cat((hidden,self.search_state,self.set_summary),-1))
  out['value']=self.adapter.value(z).squeeze(-1)
  return out
 def replay_with_customers(self,states):
     import numpy as np
     if not states:raise ValueError('Empty replay')
     n=states[0]['n'];ids=[t['task'] for t in states];b=len(states)
     if any(t['n']!=n for t in states):raise ValueError('Replay by N')
     def stack(key,dtype=torch.float32):return torch.as_tensor(np.stack([t[key] for t in states]),dtype=dtype,device=self.device)
     hidden=stack('hidden');nodes=stack('nodes');physical=stack('physical');raw=stack('raw');old_raw=stack('refresh_raw')
     mode_logits,alpha_all,beta_all,mu=self.q_from_physical(hidden,physical,raw,ids,stack('dynamic_position'))
     modes=torch.tensor([t['mode'] for t in states],device=self.device,dtype=torch.long);rows=torch.arange(b,device=self.device)
     alpha=alpha_all[rows,modes];beta=beta_all[rows,modes];u=raw.new_tensor([t['u'] for t in states])
     lo,phys_weights=self.node_logits_parts(nodes,None,old_raw,raw.new_full((b,),.04),ids)
     hi,_=self.node_logits_parts(nodes,None,old_raw,raw.new_full((b,),.25),ids)
     q_condition=raw.new_tensor([t['action']['q_condition'] for t in states]);mix=((q_condition/n-.04)/.21).clamp(0.,1.)[:,None]
     # Nonlearned feature reductions use float64 on both CPU and GPU; network remains FP32.
     pp=physical.double();norm=((pp-pp.mean(1,keepdim=True))/pp.std(1,unbiased=False,keepdim=True).clamp_min(1e-5)).float()
     logits=(1-mix)*lo+mix*hi+(norm*phys_weights[:,None,:]).sum(-1)
     logits=logits+(norm*self.customer_condition(hidden,self.search_state,self.set_summary)[:,None,:]).sum(-1)
     assignments=stack('route_assignment',torch.long);demands=stack('demands',torch.float64)
     route_phi,membership,counts=route_features_torch(physical,assignments,demands,raw.new_tensor([t['capacity'] for t in states]))
     route_means=(membership.double().transpose(1,2).bmm(logits.double()[:,:,None]).squeeze(-1)/counts.clamp_min(1.)).float()
     heads=[self.heads[t] for t in ids]
     route_weights=torch.bmm(raw[:,:5,None].transpose(1,2),torch.stack([h.route_bilinear for h in heads])).squeeze(1)
     scales=torch.stack([h.route_score_scale for h in heads])
     route_scores=scales[:,None]*route_means+(route_phi*route_weights[:,None,:]).sum(-1)
     route_scores=route_scores.masked_fill(counts==0,-torch.inf)
     micro=raw.new_zeros(b);micro_entropy=raw.new_zeros(b)
     for mode in (0,1,2):
         indices=[i for i,t in enumerate(states) if t['mode']==mode]
         if not indices:continue
         index=torch.as_tensor(indices,device=self.device)
         sub=[states[i] for i in indices]
         selected=[[(c-1) for c in t['action']['micro_selected']] if mode!=2 else t['action']['selected_routes'] for t in sub]
         values=route_scores[index] if mode==2 else logits[index]
         if mode==0:
             support=torch.zeros_like(values,dtype=torch.bool)
             for j,t in enumerate(sub):support[j,torch.as_tensor(t['action']['anchor_candidates'],device=self.device)-1]=True
         else:support=torch.isfinite(values)
         lp,entropy_sum=prefix_terms_torch(values,selected,support)
         lengths=raw.new_tensor([t['effective_selection_steps'] for t in sub])
         if mode==0:
             anchors=torch.as_tensor([t['action']['anchor']-1 for t in sub],device=self.device)
             anchor_lp=values.log_softmax(-1);anchor_p=anchor_lp.exp()
             lp=lp+anchor_lp[torch.arange(len(sub),device=self.device),anchors]
             entropy_sum=entropy_sum-(anchor_p*anchor_lp).sum(-1)
         micro[index]=lp;micro_entropy[index]=entropy_sum/lengths
     mode_dist=torch.distributions.Categorical(logits=mode_logits)
     beta_dist=Beta(alpha,beta,validate_args=False)
     macro=mode_dist.log_prob(modes)+beta_dist.log_prob(u)
     entropy=mode_dist.entropy()+2.*(mode_logits.softmax(-1)*Beta(alpha_all,beta_all,validate_args=False).entropy()).sum(-1)+micro_entropy
     return dict(logprob_macro=macro,logprob_micro=micro,entropy=entropy,alpha=alpha,beta=beta,mode_logits=mode_logits,
         mode_probabilities=mode_logits.softmax(-1),logits=logits,route_logits=route_scores,route_features=route_phi)

 def customer_condition(self,hidden,state,summary):
  return self.customer_delta(self.adapter.fusion(torch.cat((hidden,state,summary),-1)))
class NewHead(CpuHead):
 def __init__(self,policy,legacy_feature_reductions=False):
  super().__init__(policy,legacy_feature_reductions);self.policy=policy;self.state=None;self.summary=None;self.value=None
 def outputs(self,hidden,physical_state,raw,n,dynamic_position,task=0):
  mode,a,b,mu=super().outputs(hidden,physical_state,raw,n,dynamic_position,task)
  def tensor(x):return torch.as_tensor(np.asarray(x,dtype=np.float32),device=self.policy.device).reshape(1,-1)
  with torch.no_grad():o=self.policy.adapter(tensor(hidden),tensor(self.state),tensor(self.summary),tensor(mode),tensor(a),tensor(b))
  self.value=float(o['value'][0]);return *(o[k][0].cpu().numpy() for k in ('mode_logits','alpha','beta')),mu

def summary_for(env,archive):
 points=np.asarray([p['objectives'] for p in archive],dtype=float)
 delta=(points-env.objectives)/env.dynamic_span
 distance=np.linalg.norm(delta,axis=1);other=distance>1e-10
 nearest=float(distance[other].min()) if other.any() else 0.
 gap=env.score(env.objectives)-min(env.score(x) for x in points)
 # Presence flag distinguishes a singleton archive from a genuinely dense neighborhood.
 return np.asarray([np.tanh(gap),np.tanh(nearest),float(other.any())],np.float32)

def configure(policy):
 for name,p in policy.named_parameters():p.requires_grad_(name.startswith(('customer_delta.','adapter.','heads.','q_body.','q_film.','film.')))
 return torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=3e-4)

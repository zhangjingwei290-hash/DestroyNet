"""Joint mode and conditional grain PPO likelihood for frozen expert policies."""
import numpy as np,torch
KEYS=('gate_context','expert','batch','local_step','n','u','mode','alpha','beta','mode_logits','base_mode_logits','base_alpha','base_beta','logprob_macro','logprob_micro','global_seconds','last_repair_sec')
def compact(r):return {k:r[k] for k in KEYS}
def terms(gate,rows,device):
 x=torch.as_tensor(np.stack([r['gate_context'] for r in rows]),device=device,dtype=torch.float32)
 a=torch.as_tensor(np.stack([r['base_alpha'] for r in rows]),device=device,dtype=torch.float32);b=torch.as_tensor(np.stack([r['base_beta'] for r in rows]),device=device,dtype=torch.float32)
 y=gate(x);c=a+b;lo=(1+1e-6)/c;center=torch.sigmoid(torch.logit(a/c)+y[:,3:4]).maximum(lo).minimum(1-lo)
 dist=torch.distributions.Beta(center*c,(1-center)*c)
 modes=torch.distributions.Categorical(logits=torch.as_tensor(np.stack([r['base_mode_logits'] for r in rows]),device=device,dtype=torch.float32)+y[:,:3])
 old_modes=torch.distributions.Categorical(logits=torch.as_tensor(np.stack([r['mode_logits'] for r in rows]),device=device,dtype=torch.float32))
 ii=torch.arange(len(rows),device=device);mode=torch.tensor([r['mode'] for r in rows],device=device)
 u=torch.tensor([r['u'] for r in rows],device=device,dtype=torch.float32)
 old=torch.distributions.Beta(torch.tensor([r['alpha'] for r in rows],device=device),torch.tensor([r['beta'] for r in rows],device=device)).log_prob(u)
 lp=dist.log_prob(u[:,None].expand_as(a))[ii,mode]
 return lp-old+modes.log_prob(mode)-old_modes.log_prob(mode),modes.entropy()+(modes.probs*dist.entropy()).sum(-1)

def update(gate,critic,opt,rows,device='cuda'):
 from opportunity import balanced_advantages
 x=torch.tensor(np.stack([r['gate_context'] for r in rows]),device=device)
 target=torch.tensor([r['session_return'] for r in rows],device=device,dtype=torch.float32)
 with torch.no_grad():adv,w=balanced_advantages(target,critic(x).squeeze(-1),[r['n'] for r in rows])
 # Equal mass for every scale/stage/direction, with scale-normalized advantages.
 strata={(r['n'],r['stage'],r['expert']) for r in rows}
 assert len(strata)==144
 for key in strata:
  ids=[i for i,r in enumerate(rows) if (r['n'],r['stage'],r['expert'])==key];w[ids]=1./(144*len(ids))
 stats=[]
 for epoch in range(2):
  opt.zero_grad();loss_sum=0.;logratio=[]
  # Same mathematical joint batch. Group shape is irrelevant for compact replay.
  for start in range(0,len(rows),1024):
   sl=slice(start,start+1024);batch=rows[sl];lr,entropy=terms(gate,batch,device);ratio=lr.exp();aa=adv[sl];value=critic(x[sl]).squeeze(-1)
   loss=((-torch.minimum(ratio*aa,ratio.clamp(.8,1.2)*aa)+.5*(value-target[sl]).square()-.01*entropy)*w[sl]).sum()
   loss.backward();loss_sum+=float(loss.detach());logratio.append(lr.detach())
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in list(gate.parameters())+list(critic.parameters()))
  grad=float(torch.nn.utils.clip_grad_norm_(list(gate.parameters())+list(critic.parameters()),1.));opt.step()
  ll=torch.cat(logratio);stats.append(dict(epoch=epoch,loss=loss_sum,grad_norm=grad,mean_log_ratio=float(ll.mean()),clip_fraction=float((ll.exp().sub(1).abs()>.2).float().mean()),scales=[50,150,250]))
 return dict(actions=len(rows),epochs=stats,scales=[dict(n=n,count=sum(r['n']==n for r in rows),weight=float(w[[i for i,r in enumerate(rows) if r['n']==n]].sum())) for n in (50,150,250)])

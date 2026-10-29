"""20D residual GAT/GRU and preference/stall FiLM.

PPO uses log P(q) + mean selected-node log probabilities, as authorized.
This geometric-mean surrogate is not the original joint action log density.
"""
from types import SimpleNamespace
import torch
from torch import nn
from torch.distributions import Categorical
from .network import Encoder
from .config import CONTEXT,RATIOS

class Head(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.q=nn.Linear(width,len(RATIOS))
        self.node=nn.Sequential(nn.Linear(width,width),nn.ELU(),nn.Linear(width,1))
        self.physical_bilinear=nn.Parameter(torch.diag(torch.tensor([-1.,1.,-1.,1.,1.]))+.02*torch.randn(5,5))

class Policy(nn.Module):
    def __init__(self,cfg,tasks=1,critic=False):
        super().__init__();self.cfg=cfg;h=cfg.hidden_dim
        self.encoder=Encoder(h)
        self.context=nn.Sequential(nn.Linear(len(CONTEXT),h),nn.Tanh())
        # Global graph pooling, context, explicit preference and stage/stall.
        self.q_body=nn.Sequential(nn.Linear(2*h+8,h),nn.ELU())
        self.film=nn.Sequential(nn.Linear(7,h),nn.ELU(),nn.Linear(h,2*h))
        self.heads=nn.ModuleList([Head(h) for _ in range(tasks)])
        self.critics=nn.ModuleList()  # same-start group baseline; no critic error

    @property
    def device(self):return next(self.parameters()).device

    def encode(self,observations):
        n=len(observations[0]['x'])
        if any(o['x'].shape!=(n,20) for o in observations):raise ValueError('Batch equally sized 20D graphs')
        fields=dict(x=torch.cat([o['x'] for o in observations]).to(self.device),
                    state_=torch.cat([o['state_'] for o in observations]).to(self.device),
                    input_=torch.stack([o['input_'].mean(0) for o in observations]).unsqueeze(1).to(self.device))
        for name in ('edge_index_n','edge_index_r0','edge_index_r1'):
            fields[name]=torch.cat([o[name]+i*n for i,o in enumerate(observations)],1).to(self.device)
        hidden,nodes=self.encoder.encode(SimpleNamespace(**fields),n)
        raw=torch.stack([o['context'] for o in observations]).to(self.device)
        return hidden,nodes[:,1:],fields['x'].reshape(-1,n,20)[:,1:],raw

    def q_logits(self,nodes,raw,ids):
        explicit=torch.cat((raw[:,:5],raw[:,[11,13,14]]),-1)
        q_hidden=self.q_body(torch.cat((nodes.mean(1),self.context(raw),explicit),-1))
        return torch.stack([self.heads[t].q(q_hidden[i]) for i,t in enumerate(ids)])

    def node_logits(self,nodes,x,pref,q,n,task,stall):
        # Independent (pref, q/N, stall); no mixing with 40 other fields.
        q_features=torch.stack((pref.new_tensor(q/n),stall))
        gamma,beta=self.film(torch.cat((pref,q_features),-1)).chunk(2,-1)
        modulated=torch.tanh((1+gamma)*nodes+beta)
        head=self.heads[task]
        physical=x[:,[18,17,19,12,13]]
        physical=(physical-physical.mean(0,keepdim=True))/physical.std(0,keepdim=True,unbiased=False).clamp_min(1e-5)
        return head.node(modulated).squeeze(-1)+(physical*(pref@head.physical_bilinear)).sum(-1)

    def terms(self,observations,task_ids=None,actions=None,greedy=False):
        ids=[0]*len(observations) if task_ids is None else list(task_ids)
        if len(ids)!=len(observations) or any(t<0 or t>=len(self.heads) for t in ids):raise ValueError('Invalid task IDs')
        if actions is not None and len(actions)!=len(observations):raise ValueError('Action batch mismatch')
        hidden,nodes,x,raw=self.encode(observations)
        qdist=Categorical(logits=self.q_logits(nodes,raw,ids))
        qi=(qdist.logits.argmax(-1) if greedy else qdist.sample()) if actions is None else torch.tensor([a['q_index'] for a in actions],device=self.device)
        if bool(((qi<0)|(qi>=len(RATIOS))).any()):raise ValueError('Invalid q slot')
        logprob=qdist.log_prob(qi);entropy=5.0*qdist.entropy()
        output=[];lps=[];ents=[];n=nodes.shape[1]
        for i,(o,t) in enumerate(zip(observations,ids)):
            if len(o['q_values'])!=len(RATIOS):raise ValueError('Require all ten q slots')
            q=int(o['q_values'][int(qi[i])]);lp=logprob[i];ent=entropy[i]
            if not 1<=q<=n:raise ValueError('q outside 1..N')
            stored=None if actions is None else actions[i]
            if stored is not None and stored['q']!=q:raise ValueError('q/slot mismatch')
            if q==n:
                # The full set is deterministic. No position head/FiLM/Beta call,
                # no location entropy or likelihood, and no position gradients.
                removed=list(range(1,n+1))
                if stored is not None and stored['removed']!=removed:raise ValueError('Noncanonical full removal')
            else:
                logits=self.node_logits(nodes[i],x[i],raw[i,:5],q,n,t,raw[i,14])
                if stored is None:
                    if greedy:
                        selected=logits.topk(q).indices
                    else:
                        gumbel=-torch.empty_like(logits).exponential_().log()
                        selected=(logits+gumbel).topk(q).indices
                else:
                    selected=torch.tensor([c-1 for c in stored['removed']],dtype=torch.long,device=self.device)
                    if (selected.numel()!=q or bool(((selected<0)|(selected>=n)).any())
                            or torch.unique(selected).numel()!=q):
                        raise ValueError('Invalid/repeated customer')
                mask=torch.zeros((q,n),dtype=torch.bool,device=self.device)
                if q>1:
                    mask[:,selected]=torch.tril(torch.ones((q,q),dtype=torch.bool,device=self.device),diagonal=-1)
                step_logits=logits.unsqueeze(0).masked_fill(mask,-torch.inf)
                log_probs=step_logits.log_softmax(dim=-1)
                lp=lp+log_probs[torch.arange(q,device=self.device),selected].mean()
                ent=ent-(log_probs.exp()*log_probs.masked_fill(mask,0.)).sum(dim=-1).mean()
                removed=(selected+1).tolist()
            output.append(dict(q_index=int(qi[i]),q=q,removed=removed))
            lps.append(lp);ents.append(ent)
        lp=torch.stack(lps)
        return output,lp,torch.zeros_like(lp),torch.stack(ents),hidden.detach().cpu()

    @torch.no_grad()
    def act(self,observations,task_ids=None):
        actions,lp,value,_,hidden=self.terms(observations,task_ids)
        return actions,lp.cpu().tolist(),value.cpu().tolist(),hidden

    @torch.no_grad()
    def greedy(self,observations,task_ids=None):
        actions,lp,value,_,hidden=self.terms(observations,task_ids,greedy=True)
        return actions,lp.cpu().tolist(),value.cpu().tolist(),hidden

    def actor_state(self):return {k:v.detach().clone() for k,v in self.state_dict().items()}

    def start_inner(self,meta):
        source=meta.actor_state();target=self.state_dict()
        for key in target:
            target[key]=source['heads.0.'+key.split('.',2)[2] if key.startswith('heads.') else key]
        self.load_state_dict(target)

    @torch.no_grad()
    def meta_update(self,inner,epsilon):
        if len(self.heads)!=1 or not 0<epsilon<=1:raise ValueError('Invalid meta update')
        source=inner.state_dict();target=self.state_dict()
        for key in target:
            if key.startswith('heads.0.'):
                suffix=key.split('.',2)[2]
                target[key].lerp_(torch.stack([source[f'heads.{t}.{suffix}'] for t in range(len(inner.heads))]).mean(0),epsilon)
            else:target[key].copy_(source[key])
        self.load_state_dict(target)

    def deploy(self,task=0):
        policy=Policy(self.cfg).to(self.device);source=self.actor_state()
        policy.load_state_dict({k:source[f'heads.{task}.'+k.split('.',2)[2] if k.startswith('heads.') else k] for k in policy.state_dict()})
        return policy

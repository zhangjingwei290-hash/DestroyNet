"""Continuous latent-Beta q with one shared captured inference host per shape.

Node log score remains the authorized mean selected PL log probability.
The latent u density is used in PPO, not the marginal mass of rounded integer q.
"""
from types import SimpleNamespace
import math
import torch
from torch import nn
from torch.distributions import Beta
from .network import Encoder
from .config import CONTEXT,RATIOS
from .q_statistics import nearest_q_slot


class Head(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.q_mu=nn.Linear(width,1);self.q_kappa=nn.Linear(width,1)
        self.q_state_mu=nn.Parameter(torch.tensor([-.15,-.10,-.25,.55,-.20,-.15,-.35]))
        self.node=nn.Sequential(nn.Linear(width,width),nn.ELU(),nn.Linear(width,1))
        self.physical_bilinear=nn.Parameter(torch.diag(torch.tensor([-1.,1.,-1.,1.,1.]))+.02*torch.randn(5,5))


def pack(observations):
    n=len(observations[0]['x'])
    if any(o['x'].shape!=(n,20) for o in observations):raise ValueError('Batch equally sized 20D graphs')
    fields=dict(x=torch.cat([o['x'] for o in observations]),state_=torch.cat([o['state_'] for o in observations]),
        input_=torch.stack([o['input_'].mean(0) for o in observations]).unsqueeze(1),
        raw=torch.stack([o['context'] for o in observations]))
    for name in ('edge_index_n','edge_index_r0','edge_index_r1'):
        fields[name]=torch.cat([o[name]+i*n for i,o in enumerate(observations)],1)
    return fields,n


def integer_q(u,n):return ((.01+.59*u)*n).round().long().clamp(1,n)


def selected_logprob(logits,selected,q):
    ordered=logits.gather(1,selected)
    log_z=torch.logcumsumexp(ordered.flip(1),dim=1).flip(1)
    active=torch.arange(logits.shape[1],device=logits.device)[None,:]<q[:,None]
    score=((ordered-log_z)*active).sum(1)/q
    return torch.where(q==logits.shape[1],torch.zeros_like(score),score)


def selected_entropy(logits,selected,q):
    b,n=logits.shape;order=torch.arange(n,device=logits.device)
    ranks=torch.empty_like(selected).scatter_(1,selected,order[None,:].expand(b,-1))
    mask=ranks[:,None,:]<order[None,:,None]
    lp=logits[:,None,:].expand(-1,n,-1).masked_fill(mask,-torch.inf).log_softmax(-1)
    ent=-(lp.exp()*lp.masked_fill(mask,0.)).sum(-1)
    ent=(ent*(order[None,:]<q[:,None])).sum(1)/q
    return torch.where(q==n,torch.zeros_like(ent),ent)


class Policy(nn.Module):
    def __init__(self,cfg,tasks=1,critic=False):
        super().__init__();self.cfg=cfg;h=cfg.hidden_dim
        self.encoder=Encoder(h)
        self.context=nn.Sequential(nn.Linear(len(CONTEXT),h),nn.Tanh())
        self.q_history_norm=nn.LayerNorm(h)
        self.q_body=nn.Sequential(nn.Linear(2*h+27,h),nn.LayerNorm(h),nn.ELU())
        self.q_film=nn.Sequential(nn.Linear(27,h),nn.ELU(),nn.Linear(h,2*h))
        self.film=nn.Sequential(nn.Linear(7,h),nn.ELU(),nn.Linear(h,2*h))
        self.heads=nn.ModuleList([Head(h) for _ in range(tasks)])
        self.critics=nn.ModuleList()
        self.register_buffer('phys_idx',torch.tensor([18,17,19,12,13],dtype=torch.long),persistent=False)

    @property
    def device(self):return next(self.parameters()).device

    def encode_fields(self,fields,n):
        hidden,nodes=self.encoder.encode(SimpleNamespace(**fields),n)
        return hidden,nodes[:,1:],fields['x'].reshape(-1,n,20)[:,1:],fields['raw']

    def encode(self,observations):
        fields,n=pack(observations)
        return self.encode_fields({k:v.to(self.device) for k,v in fields.items()},n)

    def q_beta_params(self,hidden,nodes,x,raw,ids):
        phys=x.index_select(2,self.phys_idx);mu=phys.mean(1);sigma=phys.std(1,unbiased=False)
        scale=phys.abs().mean(1).clamp_min(1e-4)
        cv=(sigma/scale).clamp(0.,4.)/4.
        z_peak=((phys.amax(1)-mu)/sigma.clamp_min(1e-4)).clamp(0.,6.)/6.
        k10=max(1,x.shape[1]//10)
        top10_share=(phys.topk(k10,dim=1).values.abs().sum(1)/phys.abs().sum(1).clamp_min(1e-4)).clamp(0.,1.)
        struct_state=torch.cat((cv,z_peak,top10_share),-1)
        momentum=torch.cat((raw[:,:5],raw[:,11:18]),-1)
        state=torch.cat((struct_state,momentum),-1)
        q_base=self.q_body(torch.cat((self.q_history_norm(hidden),self.context(raw),state),-1))
        gamma,beta=self.q_film(state).chunk(2,-1)
        q_hidden=torch.tanh((1+gamma)*q_base+beta)
        # Exact CONTEXT fields: regret2=22, regret3=23.
        is_regret=raw[:,22]+raw[:,23]
        indicators=torch.stack((top10_share[:,1:].mean(-1)-.38,top10_share[:,0]-.38,
            raw[:,15]-.30,raw[:,14]-.15,(1.-raw[:,13])-.50,raw[:,11]-.50,is_regret-.50),-1)
        heads=[self.heads[t] for t in ids]
        mu_weight=torch.stack([head.q_mu.weight[0] for head in heads]);mu_bias=torch.stack([head.q_mu.bias[0] for head in heads])
        base_raw=(q_hidden*mu_weight).sum(-1)+mu_bias
        # No state-regime anchors or output-logit tanh boxes. The continuous
        # context and structural indicators determine the Beta location directly.
        ind_raw=(indicators*torch.stack([head.q_state_mu for head in heads])).sum(-1)
        mu_prop=.06+.84*torch.sigmoid(base_raw+ind_raw)
        k_weight=torch.stack([head.q_kappa.weight[0] for head in heads]);k_bias=torch.stack([head.q_kappa.bias[0] for head in heads])
        kappa=8.+10.*torch.sigmoid((q_hidden*k_weight).sum(-1)+k_bias)
        return 1.+mu_prop*kappa,1.+(1.-mu_prop)*kappa,mu_prop

    def node_logits_batch(self,nodes,x,raw,q_over_n,ids):
        gamma,beta=self.film(torch.cat((raw[:,:5],q_over_n[:,None],raw[:,14:15]),-1)).chunk(2,-1)
        modulated=torch.tanh((1+gamma[:,None])*nodes+beta[:,None])
        heads=[self.heads[t] for t in ids]
        w=torch.stack([head.node[0].weight for head in heads]);b=torch.stack([head.node[0].bias for head in heads])
        z=torch.nn.functional.elu(torch.bmm(modulated,w.transpose(1,2))+b[:,None,:])
        w=torch.stack([head.node[2].weight[0] for head in heads]);b=torch.stack([head.node[2].bias[0] for head in heads])
        values=(z*w[:,None,:]).sum(-1)+b[:,None]
        physical=x.index_select(2,self.phys_idx)
        physical=(physical-physical.mean(1,keepdim=True))/physical.std(1,keepdim=True,unbiased=False).clamp_min(1e-5)
        physical_weights=torch.bmm(raw[:,:5,None].transpose(1,2),torch.stack([head.physical_bilinear for head in heads])).squeeze(1)
        return values+(physical*physical_weights[:,None,:]).sum(-1)

    def _tensor_sample(self,fields,n,ids,greedy):
        hidden,nodes,x,raw=self.encode_fields(fields,n)
        alpha,beta,mode=self.q_beta_params(hidden,nodes,x,raw,ids)
        distribution=Beta(alpha,beta,validate_args=False)
        u=mode if greedy else distribution.sample()
        q=integer_q(u,n-1)
        logits=self.node_logits_batch(nodes,x,raw,q.to(x.dtype)/(n-1),ids)
        selected=logits.argsort(-1,descending=True) if greedy else (logits-torch.empty_like(logits).exponential_().log()).argsort(-1,descending=True)
        canonical=torch.arange(n-1,device=self.device)[None,:].expand(len(ids),-1)
        selected=torch.where((q==n-1)[:,None],canonical,selected)
        lp=distribution.log_prob(u)+selected_logprob(logits,selected,q)
        return u,q,selected,lp,hidden,alpha,beta,mode,logits

    def _ids(self,observations,task_ids):
        ids=[0]*len(observations) if task_ids is None else list(task_ids)
        if not ids or len(ids)!=len(observations) or any(t<0 or t>=len(self.heads) for t in ids):raise ValueError('Invalid task IDs')
        return ids

    @torch.no_grad()
    def _act(self,observations,task_ids,greedy):
        ids=self._ids(observations,task_ids)
        if self.device.type=='cuda':
            from .cuda_actor import runner_for
            outputs=runner_for(self,len(observations[0]['x'])-1,len(observations),greedy).run(self,observations,ids)
        else:
            fields,n=pack(observations);outputs=self._tensor_sample(fields,n,ids,greedy)
        u,q,selected,lp,hidden=outputs[:5]
        values=torch.cat((u[:,None],q[:,None],selected.to(u.dtype)),1).cpu().tolist()
        actions=[];n=len(observations[0]['x'])-1
        for row in values:
            size=int(row[1]);removed=[int(c)+1 for c in row[2:2+size]]
            actions.append(dict(u=float(row[0]),q_ratio=size/n,q_index=nearest_q_slot(size,n),q=size,removed=removed))
        return actions,lp.cpu().tolist(),[0.]*len(actions),hidden.detach().cpu()

    def act(self,observations,task_ids=None):return self._act(observations,task_ids,False)
    def greedy(self,observations,task_ids=None):return self._act(observations,task_ids,True)

    def terms(self,observations,task_ids=None,actions=None,greedy=False):
        ids=self._ids(observations,task_ids)
        if actions is None:
            actions,_,_,_=self._act(observations,ids,greedy)
        if len(actions)!=len(observations):raise ValueError('Action batch mismatch')
        hidden,nodes,x,raw=self.encode(observations);n=nodes.shape[1]
        alpha,beta,mode=self.q_beta_params(hidden,nodes,x,raw,ids)
        us=[];orders=[]
        for action in actions:
            u=float(action['u']);q=action['q'];removed=action['removed']
            if not math.isfinite(u) or not 0.<u<1.:raise ValueError('Beta action u must be strictly inside (0,1)')
            expected=int(integer_q(torch.tensor(u,dtype=alpha.dtype),n))
            if q!=expected or len(removed)!=q or len(set(removed))!=q or any(type(c)!=int or not 1<=c<=n for c in removed):raise ValueError('Invalid continuous exact-q action')
            if action['q_index']!=nearest_q_slot(q,n):raise ValueError('Wrong reporting bin')
            us.append(u);chosen=[c-1 for c in removed];used=set(chosen)
            orders.append(chosen+[c for c in range(n) if c not in used])
        u=alpha.new_tensor(us);q=integer_q(u,n)
        logits=self.node_logits_batch(nodes,x,raw,q.to(x.dtype)/n,ids)
        selected=torch.tensor(orders,dtype=torch.long,device=self.device)
        distribution=Beta(alpha,beta,validate_args=False)
        lp=distribution.log_prob(u)+selected_logprob(logits,selected,q)
        entropy=5.*distribution.entropy()+selected_entropy(logits,selected,q)
        return actions,lp,torch.zeros_like(lp),entropy,hidden.detach().cpu()

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

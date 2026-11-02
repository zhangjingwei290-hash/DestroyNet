"""Continuous latent-Beta q with one shared captured inference host per shape.

Structured mode/Beta and exact selected-prefix likelihoods; no tail ordering.
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
        self.q_mu=nn.Linear(width,3);self.q_kappa=nn.Linear(width,3)
        self.mode_logits=nn.Linear(width,3)
        self.route_bilinear=nn.Parameter(torch.tensor([[-1.,1.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.],[0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,1.]]))
        self.route_score_scale=nn.Parameter(torch.tensor(1.))
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


class Policy(nn.Module):
    def __init__(self,cfg,tasks=1,critic=False):
        super().__init__();self.cfg=cfg;h=cfg.hidden_dim
        self.encoder=Encoder(h)
        self.context=nn.Sequential(nn.Linear(len(CONTEXT),h),nn.Tanh())
        self.q_history_norm=nn.LayerNorm(h)
        self.q_body=nn.Sequential(nn.Linear(2*h+33,h),nn.LayerNorm(h),nn.ELU())
        self.q_film=nn.Sequential(nn.Linear(33,h),nn.ELU(),nn.Linear(h,2*h))
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

    def q_from_physical(self,hidden,phys,raw,ids,dynamic_position):
        values=phys.double()
        mu=values.mean(1);sigma=values.std(1,unbiased=False)
        scale=values.abs().mean(1).clamp_min(1e-4)
        cv=(sigma/scale).clamp(0.,4.)/4.
        peak=((values.amax(1)-mu)/sigma.clamp_min(1e-4)).clamp(0.,6.)/6.
        k=max(1,values.shape[1]//10)
        share=(values.topk(k,dim=1).values.abs().sum(1)/values.abs().sum(1).clamp_min(1e-4)).clamp(0.,1.)
        n=phys.shape[1]
        extra=torch.stack((raw.new_full((len(ids),),math.log(n/50.)),raw.new_full((len(ids),),10./n),raw[:,19],raw[:,14]),-1)
        state=torch.cat((cv.float(),peak.float(),share.float(),raw[:,:5],raw[:,11:18],extra,dynamic_position),-1)
        base=self.q_body(torch.cat((self.q_history_norm(hidden),self.context(raw),state),-1))
        gamma,bias=self.q_film(state).chunk(2,-1)
        q_hidden=torch.tanh((1+gamma)*base+bias)
        heads=[self.heads[t] for t in ids]
        def predict(name):
            weights=torch.stack([getattr(head,name).weight for head in heads]);biases=torch.stack([getattr(head,name).bias for head in heads])
            return torch.bmm(weights,q_hidden[:,:,None]).squeeze(-1)+biases
        mode_logits=predict('mode_logits')
        mu=.02+.96*torch.sigmoid(predict('q_mu'))
        kappa=2.+18.*torch.sigmoid(predict('q_kappa'))
        return mode_logits,1.+mu*kappa,1.+(1.-mu)*kappa,mu

    def node_logits_parts(self,nodes,x,raw,q_over_n,ids):
        gamma,beta=self.film(torch.cat((raw[:,:5],q_over_n[:,None],raw[:,14:15]),-1)).chunk(2,-1)
        modulated=torch.tanh((1+gamma[:,None])*nodes+beta[:,None])
        heads=[self.heads[t] for t in ids]
        w=torch.stack([head.node[0].weight for head in heads]);b=torch.stack([head.node[0].bias for head in heads])
        z=torch.nn.functional.elu(torch.bmm(modulated,w.transpose(1,2))+b[:,None,:])
        w=torch.stack([head.node[2].weight[0] for head in heads]);b=torch.stack([head.node[2].bias[0] for head in heads])
        values=(z*w[:,None,:]).sum(-1)+b[:,None]
        physical_weights=torch.bmm(raw[:,:5,None].transpose(1,2),torch.stack([head.physical_bilinear for head in heads])).squeeze(1)
        return values,physical_weights

    def node_logits_batch(self,nodes,x,raw,q_over_n,ids):
        values,physical_weights=self.node_logits_parts(nodes,x,raw,q_over_n,ids)
        physical=x.index_select(2,self.phys_idx)
        physical=(physical-physical.mean(1,keepdim=True))/physical.std(1,keepdim=True,unbiased=False).clamp_min(1e-5)
        raw_logits=values+(physical*physical_weights[:,None,:]).sum(-1)
        return raw_logits

    def _tensor_dual_cache(self,fields,n,ids):
        hidden,nodes,x,raw=self.encode_fields(fields,n)
        lo,weights=self.node_logits_parts(nodes,x,raw,raw.new_full((len(ids),),.04),ids)
        hi,_=self.node_logits_parts(nodes,x,raw,raw.new_full((len(ids),),.25),ids)
        return torch.cat((hidden,lo,hi,weights,nodes.reshape(len(ids),-1)),1)

    @torch.no_grad()
    def refresh(self,obs,task=0,retain_nodes=False):
        from .cuda_actor import cache_runner_for
        packed=cache_runner_for(self,len(obs['x'])-1,1).run(self,[obs],[task])
        n=len(obs['x'])-1;h=self.cfg.hidden_dim;prefix=h+2*n+5
        row=(packed if retain_nodes else packed[:,:prefix]).detach().cpu().numpy()[0].copy()
        result=dict(hidden=row[:h].copy(),values_lo=row[h:h+n].copy(),values_hi=row[h+n:h+2*n].copy(),
            physical_weights=row[h+2*n:prefix].copy(),refresh_raw=obs['context'].numpy().copy())
        if retain_nodes:result['nodes']=row[prefix:].reshape(n,h).copy()
        return result

    def replay_terms(self,states):
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

    def _ids(self,observations,task_ids):
        ids=[0]*len(observations) if task_ids is None else list(task_ids)
        if not ids or len(ids)!=len(observations) or any(t<0 or t>=len(self.heads) for t in ids):raise ValueError('Invalid task IDs')
        return ids

    def act(self,*args,**kwargs):
        raise RuntimeError('Stage2 must use cached CPU sampler and replay_terms; legacy selected-PL score is disabled')
    def terms(self,states,*args,**kwargs):return self.replay_terms(states)

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
            else:target[key].lerp_(source[key],epsilon)
        self.load_state_dict(target)

    def deploy(self,task=0):
        policy=Policy(self.cfg).to(self.device);source=self.actor_state()
        policy.load_state_dict({k:source[f'heads.{task}.'+k.split('.',2)[2] if k.startswith('heads.') else k] for k in policy.state_dict()})
        return policy


def route_features_torch(physical,assignment,demands,capacities):
    b,n,_=physical.shape;v=int(assignment.max().detach().cpu())+1
    membership=(assignment[:,:,None]==torch.arange(v,device=physical.device)[None,None,:]).float()
    counts=membership.double().sum(1);valid=counts>0;v_count=valid.sum(1,keepdim=True)
    values=torch.cat((demands[:,:,None],physical.double()),-1)
    totals=membership.double().transpose(1,2).bmm(values)
    means=totals/counts.clamp_min(1.)[:,:,None]
    phi=torch.stack((counts/(n/v_count),1.-totals[:,:,0]/capacities.double()[:,None],means[:,:,2],1.-means[:,:,3],means[:,:,4],means[:,:,5]),-1)
    mask=valid[:,:,None];average=(phi*mask).sum(1,keepdim=True)/v_count[:,:,None]
    variance=((phi-average).square()*mask).sum(1,keepdim=True)/v_count[:,:,None]
    normalized=((phi-average)/variance.sqrt().clamp_min(1e-5)).masked_fill(~mask,0.)
    return normalized.float(),membership,counts


def prefix_terms_torch(logits,selected,support):
    """Sum exact prefix PL terms; unselected tail appears only as unordered mass."""
    b,n=logits.shape;l=max(map(len,selected));device=logits.device
    chosen=torch.full((b,l),n,device=device,dtype=torch.long);active=torch.zeros((b,l),device=device,dtype=torch.bool)
    unselected=support.clone()
    for j,order in enumerate(selected):
        ids=torch.as_tensor(order,device=device,dtype=torch.long);chosen[j,:len(order)]=ids;active[j,:len(order)]=True;unselected[j,ids]=False
    extended=torch.cat((logits,logits.new_zeros((b,1))),-1)
    ordered=extended.gather(1,chosen).masked_fill(~active,-1e30)
    real_tail=unselected.any(-1)
    tail_mask=torch.cat((unselected,(~real_tail)[:,None]),-1)
    safe_tail=extended.masked_fill(~tail_mask,-torch.inf).logsumexp(-1)
    tail=torch.where(real_tail,safe_tail,logits.new_full((b,),-torch.inf))
    suffix=torch.logcumsumexp(ordered.flip(1),1).flip(1)
    logz=torch.logaddexp(tail[:,None],suffix)
    lp=torch.where(active,ordered-logz,torch.zeros_like(ordered)).sum(-1)
    rank=torch.full((b,n+1),l+1,device=device,dtype=torch.long)
    rank.scatter_(1,chosen,torch.arange(l,device=device)[None,:].expand(b,-1))
    step=torch.arange(l,device=device)[None,:,None]
    available=(rank[:,:n][:,None,:]>=step)&support[:,None,:]
    available=torch.cat((available,(~active)[:,:,None]),-1)
    masked=extended[:,None,:].expand(-1,l,-1).masked_fill(~available,-torch.inf)
    logp=masked-masked.logsumexp(-1,keepdim=True)
    entropy=-(logp.exp()*logp.masked_fill(~available,0.)).sum(-1)
    return lp,(entropy*active).sum(-1)

"""Float32 CPU current-state q head and exact auxiliary-action density."""
import math,time
import numpy as np
from stage1_mechanisms import normalize_physical

def struct_state(phys):
    # Stable nonlearned reductions; neural matrices and returned features remainFP32.
    phys=np.asarray(phys,dtype=np.float64)
    mean=phys.mean(0);sigma=phys.std(0)
    cv=np.clip(sigma/np.maximum(np.abs(phys).mean(0),1e-4),0.,4.)/4.
    peak=np.clip((phys.max(0)-mean)/np.maximum(sigma,1e-4),0.,6.)/6.
    k=max(1,len(phys)//10)
    top=np.partition(phys,len(phys)-k,axis=0)[-k:]
    share=np.clip(np.abs(top).sum(0)/np.maximum(np.abs(phys).sum(0),1e-4),0.,1.)
    return np.concatenate((cv,peak,share)).astype(np.float32)
def legacy_struct_state(phys):
    phys=np.asarray(phys,dtype=np.float32)
    mean=phys.mean(0);sigma=phys.std(0)
    cv=np.clip(sigma/np.maximum(np.abs(phys).mean(0),1e-4),0.,4.)/4.
    peak=np.clip((phys.max(0)-mean)/np.maximum(sigma,1e-4),0.,6.)/6.
    k=max(1,len(phys)//10)
    top=np.partition(phys,len(phys)-k,axis=0)[-k:]
    share=np.clip(np.abs(top).sum(0)/np.maximum(np.abs(phys).sum(0),1e-4),0.,1.)
    return np.concatenate((cv,peak,share)).astype(np.float32)

def linear(x,params,name):return x @ params[name+'.weight'].T+params[name+'.bias']
def elu(x):return np.where(x>=0,x,np.expm1(np.minimum(x,0))).astype(np.float32)
def layernorm(x,params,name):
    return ((x-x.mean())/np.sqrt(x.var()+np.float32(1e-5))*params[name+'.weight']+params[name+'.bias']).astype(np.float32)
def sigmoid(x):return np.float32(1.)/(np.float32(1.)+np.exp(-np.clip(x,np.float32(-80.),np.float32(80.))))

class CpuHead:
    def __init__(self,policy,legacy_feature_reductions=False):
        self.legacy_feature_reductions=legacy_feature_reductions
        self.params={k:v.detach().cpu().numpy().copy() for k,v in policy.state_dict().items() if not k.startswith('encoder.')}
    def outputs(self,hidden,physical_state,raw,n,dynamic_position,task=0):
        p=self.params;raw=np.asarray(raw,dtype=np.float32)
        scale=np.asarray([math.log(n/50.),10./n,raw[19],raw[14]],dtype=np.float32)
        state=np.concatenate((physical_state,raw[:5],raw[11:18],scale,np.asarray(dynamic_position,dtype=np.float32))).astype(np.float32)
        context=np.tanh(linear(raw,p,'context.0'));h=layernorm(np.asarray(hidden,dtype=np.float32).reshape(-1),p,'q_history_norm')
        base=elu(layernorm(linear(np.concatenate((h,context,state)),p,'q_body.0'),p,'q_body.1'))
        gamma,bias=np.split(linear(elu(linear(state,p,'q_film.0')),p,'q_film.2'),2)
        qh=np.tanh((1+gamma)*base+bias)
        mode_logits=linear(qh,p,f'heads.{task}.mode_logits')
        mu=np.float32(.02)+np.float32(.96)*sigmoid(linear(qh,p,f'heads.{task}.q_mu'))
        kappa=np.float32(2.)+np.float32(18.)*sigmoid(linear(qh,p,f'heads.{task}.q_kappa'))
        return mode_logits.astype(np.float32),(1+mu*kappa).astype(np.float32),(1+(1-mu)*kappa).astype(np.float32),mu.astype(np.float32)

def sample(cache,physical,raw,n,rng,neighbors,head,route_state,dynamic_position,task=0,forced_mode=None,forced_u=None):
    from stage2_structured import softmax,categorical_logprob,beta_logprob,q_for_mode,interpolate,prefix_logprob,top_prefix,route_logits,route_removal,MODE_NAMES
    q_start=time.perf_counter()
    mode_logits,alpha,beta,mus=head.outputs(cache['hidden'],(legacy_struct_state(physical) if head.legacy_feature_reductions else struct_state(physical)),raw,n,dynamic_position,task)
    q_seconds=time.perf_counter()-q_start
    mode=int(rng.choice(3,p=softmax(mode_logits))) if forced_mode is None else int(forced_mode)
    u=float(np.clip(np.float32(rng.beta(float(alpha[mode]),float(beta[mode]))) if forced_u is None else np.float32(forced_u),np.nextafter(np.float32(0),np.float32(1)),np.nextafter(np.float32(1),np.float32(0))))
    size,q_condition=q_for_mode(mode,u,n,len(route_state['routes']))
    logits=(interpolate(cache['values_lo'],cache['values_hi'],q_condition,n)+normalize_physical(physical)@cache['physical_weights']).astype(np.float32)
    anchor=None;candidates=None;selected_routes=[];over_cap=False;cap=None
    route_scores=None
    if mode==0:
        anchor=int(rng.choice(n,p=softmax(logits)))+1
        M=min(n-1,max(2*(size-1),16))
        candidates=[int(c) for c in neighbors[anchor] if c not in (0,anchor)][:M]
        selected=top_prefix(logits,size-1,rng,np.asarray(candidates)-1)
        removed=[anchor]+(selected+1).tolist();effective=size
        micro=categorical_logprob(logits,anchor-1)+prefix_logprob(logits,selected,np.asarray(candidates)-1)
        micro_selected=(selected+1).tolist()
    elif mode==1:
        selected=top_prefix(logits,size,rng);removed=(selected+1).tolist();effective=size
        micro=prefix_logprob(logits,selected);micro_selected=removed
    else:
        p=head.params;route_scores=route_logits(logits,route_state,raw[:5],p[f'heads.{task}.route_bilinear'],p[f'heads.{task}.route_score_scale'])
        selected_routes=top_prefix(route_scores,size,rng).tolist()
        removed,cap,over_cap=route_removal(route_state['routes'],selected_routes,logits,n)
        effective=size;micro=prefix_logprob(route_scores,selected_routes);micro_selected=selected_routes
    q=len(removed)
    macro=categorical_logprob(mode_logits,mode)+beta_logprob(float(alpha[mode]),float(beta[mode]),u)
    from solver.q_statistics import nearest_q_slot
    action=dict(mode=mode,mode_name=MODE_NAMES[mode],u=u,q=q,q_ratio=q/n,q_index=nearest_q_slot(q,n),removed=removed,
        q_condition=q_condition,effective_selection_steps=effective,anchor=anchor,anchor_candidates=candidates,
        selected_routes=selected_routes,micro_selected=micro_selected,q_cap=cap,first_route_exceeds_cap=over_cap)
    replay=dict(hidden=cache['hidden'],nodes=cache.get('nodes'),refresh_raw=cache['refresh_raw'],physical=physical,raw=raw,
        route_assignment=route_state['assignment'],demands=route_state['demands'],capacity=route_state['capacity'],routes=route_state['routes'],
        dynamic_position=np.asarray(dynamic_position,dtype=np.float32),task=task,n=n,u=u,mode=mode,action=action,
        alpha=float(alpha[mode]),beta=float(beta[mode]),mode_logits=mode_logits,mode_probabilities=softmax(mode_logits),
        logits=logits,route_logits=route_scores,logprob_macro=macro,logprob_micro=micro,effective_selection_steps=effective,q_head_seconds=q_seconds)
    return action,replay

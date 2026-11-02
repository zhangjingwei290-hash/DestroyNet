"""Structured modes and exact selected-prefix densities; no unselected tail ordering."""
import math
import numpy as np
MODE_NAMES=('spatial_cluster','bottleneck_global','route_coherent')

def logsumexp(x):
    x=np.asarray(x,dtype=np.float64)
    if not len(x):return -math.inf
    maximum=float(x.max());return maximum+math.log(float(np.exp(x-maximum).sum()))
def categorical_logprob(logits,index):return float(logits[index])-logsumexp(logits)
def softmax(logits):
    values=np.asarray(logits,dtype=np.float64);v=np.exp(values-values.max());return v/v.sum()
def beta_logprob(a,b,u):return (a-1)*math.log(u)+(b-1)*math.log1p(-u)+math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)

def prefix_logprob(logits,selected,support=None):
    logits=np.asarray(logits,dtype=np.float64);chosen=np.asarray(selected,dtype=int)
    support=np.arange(len(logits)) if support is None else np.asarray(support,dtype=int)
    assert len(chosen)>0 and len(set(chosen.tolist()))==len(chosen) and set(chosen)<=set(support)
    remaining=support[~np.isin(support,chosen)]
    unselected_log_z=logsumexp(logits[remaining])
    chosen_logits=logits[chosen]
    suffix=np.logaddexp.accumulate(chosen_logits[::-1])[::-1]
    return float((chosen_logits-np.logaddexp(unselected_log_z,suffix)).sum())

def q_for_mode(mode,u,n,v):
    if mode==0:
        upper=min(max(12,math.floor(.22*n)),45)
        q=int(np.clip(np.rint(2.*(upper/2.)**float(u)),2,n));return q,q
    if mode==1:
        upper=min(max(10,math.floor(.35*n)),75)
        q=int(np.clip(np.rint(3.*(upper/3.)**float(u)),3,n));return q,q
    if mode!=2:raise ValueError('Unknown mode')
    upper=min(v,max(1,min(4,math.floor(.25*v))))
    k=int(np.clip(1+math.floor(float(u)*upper),1,upper))
    estimated=int(np.clip(np.rint(k*n/v),2,n));return k,estimated

def interpolate(values_lo,values_hi,q_condition,n):
    coefficient=np.float32(np.clip((q_condition/n-.04)/.21,0.,1.))
    return ((np.float32(1.)-coefficient)*values_lo+coefficient*values_hi).astype(np.float32)

def route_context(instance,routes,physical):
    from itertools import chain
    n=instance.n_customers;lengths=np.asarray([len(r) for r in routes],dtype=int)
    if not len(lengths) or np.any(lengths<=0):raise ValueError('Nonempty routes only')
    clients=np.fromiter(chain.from_iterable(routes),dtype=int,count=n)
    assignment=np.empty(n,dtype=np.int64);assignment[clients-1]=np.repeat(np.arange(len(routes)),lengths)
    values=np.column_stack((np.ones(n),np.asarray(instance.customer_demands,dtype=np.float64),np.asarray(physical,dtype=np.float64)))
    totals=np.zeros((len(routes),7),dtype=np.float64);np.add.at(totals,assignment,values)
    count=totals[:,0];v=len(routes)
    features=np.column_stack((count/(n/v),1-totals[:,1]/instance.vehicle_capacity,
        totals[:,3]/count,1-totals[:,4]/count,totals[:,5]/count,totals[:,6]/count))
    normalized=(features-features.mean(0))/np.maximum(features.std(0),1e-5)
    return dict(assignment=assignment,counts=count.astype(np.float32),normalized=normalized.astype(np.float32),
        features=features,demands=np.asarray(instance.customer_demands,dtype=np.float64),capacity=float(instance.vehicle_capacity),routes=routes)

def route_logits(node_logits,context,preference,bilinear,score_scale):
    means=np.bincount(context['assignment'],weights=np.asarray(node_logits,dtype=np.float64),minlength=len(context['counts']))/context['counts']
    return (float(score_scale)*means+context['normalized']@(np.asarray(preference,dtype=np.float32)@bilinear)).astype(np.float32)

def top_prefix(logits,k,rng,support=None):
    support=np.arange(len(logits)) if support is None else np.asarray(support,dtype=int)
    exp=np.maximum(rng.exponential(size=len(support)).astype(np.float32),np.finfo(np.float32).tiny)
    scores=np.asarray(logits,dtype=np.float32)[support]-np.log(exp)
    return support[np.argsort(-scores,kind='stable')[:k]]

def route_removal(routes,selected,node_logits,n):
    cap=min(int(np.rint(.45*n)),100);first=list(routes[int(selected[0])]);removed=list(first)
    exceeded=len(first)>cap
    if exceeded:return removed,cap,True
    for rid in selected[1:]:
        route=list(routes[int(rid)]);remaining=cap-len(removed)
        if remaining<=0:break
        if len(route)<=remaining:removed.extend(route)
        else:
            order=np.argsort(-np.asarray(node_logits)[np.asarray(route)-1],kind='stable')[:remaining]
            removed.extend(np.asarray(route)[order].tolist());break
    assert len(removed)==len(set(removed))
    return removed,cap,False

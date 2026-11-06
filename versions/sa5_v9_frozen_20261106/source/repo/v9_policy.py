"""Scale/preference controller and NumPy inference for five-step destroy sessions."""
import math
import numpy as np
import torch
from torch import nn

CONTEXT_FIELDS=('log_n50','n250','lambda0','lambda1','lambda2','lambda3','lambda4',
                'step_fraction','stagnation','temperature_fraction','mean_load_fraction',
                'route_count_fraction','time_remaining','mean_route_length','mean_detour',
                'mean_makespan_slack','mean_waiting','mean_tardiness','archive_gap',
                'archive_nearest','archive_other','last_repair_sec')

def q_bounds(mode,n,cap):
    if mode not in (0,1,2) or n<1 or cap<1:raise ValueError('Invalid mode/N/cap')
    lo=max(3,round(.04*n));hi=min(n,int(cap),max(12,min(round(.35*n),60)))
    return min(lo,hi),hi

def q_for_mode(u,mode,n,cap):
    lo,hi=q_bounds(mode,n,cap)
    return int(np.rint(lo+np.clip(u,0.,1.)*(hi-lo)))

def context(n,preference,state,physical,route_state,stall,summary,last_repair_sec):
    v=len(route_state['routes']);load=float(np.sum(route_state['demands']))/(v*route_state['capacity'])
    x=np.concatenate(([math.log(n/50.),n/250.],preference,
       [min(1.,5*(1-state[0])/4),min(stall,30)/30.,state[1],load,v/n,state[-1]],
       np.asarray(physical,dtype=np.float64).mean(0),summary,[last_repair_sec])).astype(np.float32)
    assert x.shape==(len(CONTEXT_FIELDS),) and np.isfinite(x).all()
    return x

class Controller(nn.Module):
    def __init__(self):
        super().__init__();self.body=nn.Sequential(nn.Linear(len(CONTEXT_FIELDS),32),nn.Tanh(),nn.Linear(32,4))
        nn.init.orthogonal_(self.body[0].weight);nn.init.zeros_(self.body[0].bias)
        nn.init.normal_(self.body[2].weight,std=.01);nn.init.zeros_(self.body[2].bias)
    def forward(self,x):
        return self.body(x)

class NumpyController:
    def __init__(self,controller):
        self.p={k:v.detach().cpu().numpy().astype(np.float32,copy=True) for k,v in controller.state_dict().items()}
    def __call__(self,x):
        x=np.asarray(x,dtype=np.float32);p=self.p
        h=np.tanh(x@p['body.0.weight'].T+p['body.0.bias']);y=h@p['body.2.weight'].T+p['body.2.bias']
        return y

class NumpyMLP:
    """Inference for the Linear/Tanh/ELU/LayerNorm blocks used by light heads."""
    def __init__(self,module):
        self.layers=[]
        for layer in module:
            if isinstance(layer,nn.Linear):self.layers.append(('linear',layer.weight.detach().cpu().numpy().copy(),layer.bias.detach().cpu().numpy().copy()))
            elif isinstance(layer,nn.Tanh):self.layers.append(('tanh',))
            elif isinstance(layer,nn.ELU):self.layers.append(('elu',float(layer.alpha)))
            elif isinstance(layer,nn.LayerNorm):self.layers.append(('norm',layer.weight.detach().cpu().numpy().copy(),layer.bias.detach().cpu().numpy().copy(),layer.eps))
            else:raise TypeError(type(layer).__name__)
    def __call__(self,x):
        x=np.asarray(x,dtype=np.float32)
        for layer in self.layers:
            if layer[0]=='linear':x=x@layer[1].T+layer[2]
            elif layer[0]=='tanh':x=np.tanh(x)
            elif layer[0]=='elu':x=np.where(x>=0,x,layer[1]*np.expm1(np.minimum(x,0)))
            else:x=(x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+layer[3])*layer[1]+layer[2]
        return x.astype(np.float32)

def select_route_prefix(routes,ordering,node_logits,target,cap):
    selected=[];removed=[];first_oversize=False
    for rid in ordering:
        route=list(routes[int(rid)])
        selected.append(int(rid));removed.extend(route)
        if len(selected)==1:first_oversize=len(route)>cap
        if len(removed)>=min(target,cap):break
    assert removed and len(removed)==len(set(removed))
    return selected,removed,first_oversize

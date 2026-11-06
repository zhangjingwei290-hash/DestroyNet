"""Explicit preference/physical/set conditioning; legacy actor retained at init."""
import importlib.util
from pathlib import Path
import numpy as np,torch
from torch import nn
from aligned_cpu import struct_state
from solver.config import directions
P=Path('D:/workspace/fuxian/DPR/development/sa5_coverage_v2_20261104')
spec=importlib.util.spec_from_file_location('legacy_v2_network',P/'network.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
def summary_for(env,archive):
 old=legacy.summary_for(env,archive)
 pts=np.asarray([p['objectives'] for p in archive],float);z=(pts-env.dynamic_ideal)/env.dynamic_span;cur=(env.objectives-env.dynamic_ideal)/env.dynamic_span
 best=int(np.argmin([env.score(p) for p in pts]));gap=np.tanh((env.objectives-pts[best])/env.dynamic_span)
 interaction=np.tanh(env.preference*cur)
 # Preferred-region membership: among all fixed decomposition directions,
 # find the one with minimum weighted Tchebycheff value for each archive point.
 dirs=np.asarray(directions());cost=(z[:,None,:]*dirs[None,:,:]).max(-1)
 assigned=cost.argmin(1);near=np.argsort(np.linalg.norm(dirs-env.preference,axis=1))[:5];inside=np.isin(assigned,near)
 fraction=float(inside.mean());dist=np.linalg.norm(z-cur,axis=1);different=dist>1e-10;mask=inside&different
 nearest=float(np.tanh(dist[mask].min())) if mask.any() else 1.
 return np.r_[old,gap,interaction,fraction,nearest].astype(np.float32)
class NewPolicy(legacy.NewPolicy):
 def __init__(self,cfg):
  super().__init__(cfg);h=cfg.hidden_dim;condition_dim=15+5+2+7+12
  self.rich_condition=nn.Sequential(nn.Linear(condition_dim,h),nn.LayerNorm(h),nn.ELU())
  self.rich_film=nn.Linear(h,2*h);self.rich_hidden=nn.Linear(h,h)
  self.rich_mode=nn.Linear(h,3);self.rich_beta=nn.Linear(h,6)
  self.rich_value=nn.Sequential(nn.Linear(h+condition_dim,h),nn.ELU(),nn.Linear(h,1))
  for layer in (self.rich_mode,self.rich_beta):nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
 def rich(self,hidden,physical,raw,position,state,summary,mode,a,b):
  # Nonlearned reductions match the frozen CPU q path exactly.
  ps=torch.as_tensor(np.stack([struct_state(x.detach().cpu().numpy()) for x in physical]),device=hidden.device)
  cond=torch.cat((ps,raw[:,:5],position,state,summary[:,3:]),-1)
  c=self.rich_condition(cond);gamma,bias=self.rich_film(c).chunk(2,-1)
  z=torch.tanh((1+.25*torch.tanh(gamma))*self.rich_hidden(hidden)+bias)
  da,db=self.rich_beta(z).chunk(2,-1)
  value=self.rich_value(torch.cat((hidden,cond),-1)).squeeze(-1)
  return dict(mode_logits=mode+self.rich_mode(z),alpha=a*torch.exp(da.clamp(-3,3)),beta=b*torch.exp(db.clamp(-3,3)),value=value)
 def q_from_physical(self,hidden,physical,raw,ids,position):
  full=self.set_summary;self.set_summary=full[:,:3]
  try:mode,a,b,mu=super().q_from_physical(hidden,physical,raw,ids,position)
  finally:self.set_summary=full
  out=self.rich(hidden,physical,raw,position,self.search_state,full,mode,a,b);self.latest_value=out['value']
  return out['mode_logits'],out['alpha'],out['beta'],mu
 def customer_condition(self,hidden,state,summary):return super().customer_condition(hidden,state,summary[:,:3])
 def replay_terms(self,states):
  self.search_state=torch.tensor(np.stack([s['search_state'] for s in states]),device=self.device)
  self.set_summary=torch.tensor(np.stack([s['set_summary'] for s in states]),device=self.device)
  out=self.replay_with_customers(states);out['value']=self.latest_value;return out
class NewHead(legacy.NewHead):
 def outputs(self,hidden,physical_state,raw,n,dynamic_position,task=0):
  full=self.summary;self.summary=full[:3]
  try:mode,a,b,mu=super().outputs(hidden,physical_state,raw,n,dynamic_position,task)
  finally:self.summary=full
  def t(x):return torch.as_tensor(np.asarray(x,np.float32),device=self.policy.device).reshape(1,-1)
  # rich() requires raw physical rows for the identical struct_state reduction.
  with torch.no_grad():
   physical=t(physical_state);cond=torch.cat((physical,t(raw)[:,:5],t(dynamic_position),t(self.state),t(full)[...,3:]),-1)
   c=self.policy.rich_condition(cond);ga,bi=self.policy.rich_film(c).chunk(2,-1)
   z=torch.tanh((1+.25*torch.tanh(ga))*self.policy.rich_hidden(t(hidden))+bi)
   da,db=self.policy.rich_beta(z).chunk(2,-1)
   mo=t(mode)+self.policy.rich_mode(z);aa=t(a)*torch.exp(da.clamp(-3,3));bb=t(b)*torch.exp(db.clamp(-3,3))
   self.value=float(self.policy.rich_value(torch.cat((t(hidden),cond),-1))[0,0])
  return mo[0].cpu().numpy(),aa[0].cpu().numpy(),bb[0].cpu().numpy(),mu
def configure(policy):
 for name,p in policy.named_parameters():p.requires_grad_(name.startswith(('rich_','customer_delta.','adapter.','heads.','q_body.','q_film.','film.')))
 return torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=1e-4)

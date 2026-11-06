from pathlib import Path
import importlib.util,numpy as np,torch
P=Path('D:/workspace/fuxian/DPR/development/sa5_coverage_v2_20261104')
spec=importlib.util.spec_from_file_location('v2_exact_network',P/'network.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
summary_for=legacy.summary_for
class Gate(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.body=torch.nn.Sequential(torch.nn.Linear(34,32),torch.nn.Tanh(),torch.nn.Linear(32,1))
  self.film=torch.nn.Sequential(torch.nn.Linear(34,32),torch.nn.Tanh(),torch.nn.Linear(32,64))
  torch.nn.init.zeros_(self.body[-1].weight);torch.nn.init.zeros_(self.body[-1].bias)
  torch.nn.init.zeros_(self.film[-1].weight);torch.nn.init.zeros_(self.film[-1].bias)
 def forward(self,context):
  h=self.body[:2](context);gamma,beta=self.film(context).chunk(2,-1)
  return .15*torch.tanh(self.body[2]((1+gamma)*h+beta).squeeze(-1))
 def load_v7(self,sd):
  # Preserve the old mapping exactly at initialization; new features start neutral.
  with torch.no_grad():
   self.body[0].weight.zero_();self.body[0].weight[:,:32].copy_(sd['body.0.weight'])
   self.body[0].bias.copy_(sd['body.0.bias'])
   self.body[2].weight.copy_(sd['body.2.weight']);self.body[2].bias.copy_(sd['body.2.bias'])
def gate_features(physical,raw,n,stall,state,summary,last_repair_sec,budget,route_count):
 if budget<=0 or last_repair_sec<0 or route_count<1:raise ValueError('Bad timing/route state')
 # Exact training/replay schema; mean customers/vehicle normalized by fixed 50.
 x=np.concatenate((np.asarray(raw[:5]),[n/250.,min(stall,30)/30.],np.asarray(physical),np.asarray(state),np.asarray(summary),[last_repair_sec/budget,(n/route_count)/50.])).astype(np.float32)
 assert x.shape==(34,) and np.isfinite(x).all()
 return x
class NewPolicy(legacy.NewPolicy):
 def __init__(self,cfg):super().__init__(cfg);self.stall_gate=Gate();self.cross_stall=None
 def shifted(self,a,b,w,n,stall):
  shift=self.stall_gate(self.gate_context);c=a+b;lo=(1.+1e-6)/c;center=(a/c+shift[:,None]).maximum(lo).minimum(1-lo)
  return center*c,(1-center)*c,shift
 def q_from_physical(self,hidden,physical,raw,ids,dynamic_position):
  mode,a,b,mu=super().q_from_physical(hidden,physical,raw,ids,dynamic_position)
  stall=raw.new_zeros(len(raw)) if self.cross_stall is None else self.cross_stall
  # raw19 is not N: replay supplies exact n separately.
  a,b,_=self.shifted(a,b,raw[:,:5],self.cross_n,stall)
  return mode,a,b,mu
 def replay_terms(self,states):
  self.gate_context=torch.as_tensor(np.stack([s['gate_context'] for s in states]),device=self.device);self.cross_stall=torch.tensor([s['cross_stall'] for s in states],device=self.device,dtype=torch.float32);self.cross_n=torch.tensor([s['n'] for s in states],device=self.device,dtype=torch.float32)
  return super().replay_terms(states)
class NewHead(legacy.NewHead):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.q_bias=0.;self.cross_stall=0.;self.gate_enabled=False;self.gate_shift=0.
 def outputs(self,hidden,physical,raw,n,dynamic_position,task=0):
  mode,a,b,mu=super().outputs(hidden,physical,raw,n,dynamic_position,task)
  self.gate_shift=0.
  if self.gate_enabled:
   self.gate_context=gate_features(physical,raw,n,self.cross_stall,self.state,self.summary,self.last_repair_sec,self.run_budget,self.route_count)
   d=self.policy.device
   with torch.no_grad():self.gate_shift=float(self.policy.stall_gate(torch.as_tensor(self.gate_context,device=d).reshape(1,-1)))
  shift=self.q_bias+self.gate_shift
  if shift:
   c=a+b;lo=(1.+1e-6)/c;center=np.clip(a/c+np.float32(shift),lo,1-lo);a=(center*c).astype(np.float32);b=((1-center)*c).astype(np.float32)
  return mode,a,b,mu
def configure(m):
 for name,p in m.named_parameters():p.requires_grad_(name.startswith('stall_gate.'))
 return torch.optim.Adam(m.stall_gate.parameters(),lr=3e-4)
def load_base(m,path):
 x=torch.load(path,map_location=m.device,weights_only=False);missing,unexpected=m.load_state_dict(x['model'],strict=False)
 assert not unexpected and all(n.startswith('stall_gate.') for n in missing)

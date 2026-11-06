from pathlib import Path
import importlib.util,numpy as np,torch
P=Path('D:/workspace/fuxian/DPR/development/sa5_coverage_v2_20261104')
spec=importlib.util.spec_from_file_location('v2_exact_network',P/'network.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
summary_for=legacy.summary_for
class Gate(torch.nn.Module):
 def __init__(self):
  super().__init__();self.body=torch.nn.Sequential(torch.nn.Linear(7,16),torch.nn.Tanh(),torch.nn.Linear(16,1));torch.nn.init.zeros_(self.body[-1].weight);torch.nn.init.zeros_(self.body[-1].bias)
 def forward(self,w,n,stall):
  s=stall.clamp(0,30)/30.;x=torch.cat((w,n.reshape(-1,1)/250.,s.reshape(-1,1)),-1)
  return .15*s*torch.tanh(self.body(x).squeeze(-1))
class NewPolicy(legacy.NewPolicy):
 def __init__(self,cfg):super().__init__(cfg);self.stall_gate=Gate();self.cross_stall=None
 def shifted(self,a,b,w,n,stall):
  shift=self.stall_gate(w,n,stall);c=a+b;lo=(1.+1e-6)/c;center=(a/c+shift[:,None]).maximum(lo).minimum(1-lo)
  return center*c,(1-center)*c,shift
 def q_from_physical(self,hidden,physical,raw,ids,dynamic_position):
  mode,a,b,mu=super().q_from_physical(hidden,physical,raw,ids,dynamic_position)
  stall=raw.new_zeros(len(raw)) if self.cross_stall is None else self.cross_stall
  # raw19 is not N: replay supplies exact n separately.
  a,b,_=self.shifted(a,b,raw[:,:5],self.cross_n,stall)
  return mode,a,b,mu
 def replay_terms(self,states):
  self.cross_stall=torch.tensor([s['cross_stall'] for s in states],device=self.device,dtype=torch.float32);self.cross_n=torch.tensor([s['n'] for s in states],device=self.device,dtype=torch.float32)
  return super().replay_terms(states)
class NewHead(legacy.NewHead):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.q_bias=0.;self.cross_stall=0.;self.gate_enabled=False;self.gate_shift=0.
 def outputs(self,hidden,physical,raw,n,dynamic_position,task=0):
  mode,a,b,mu=super().outputs(hidden,physical,raw,n,dynamic_position,task)
  self.gate_shift=0.
  if self.gate_enabled:
   d=self.policy.device
   with torch.no_grad():self.gate_shift=float(self.policy.stall_gate(torch.as_tensor(np.asarray(raw[:5],np.float32),device=d).reshape(1,5),torch.tensor([n],device=d),torch.tensor([self.cross_stall],device=d)))
  shift=self.q_bias+self.gate_shift
  if shift:
   c=a+b;lo=(1.+1e-6)/c;center=np.clip(a/c+np.float32(shift),lo,1-lo);a=(center*c).astype(np.float32);b=((1-center)*c).astype(np.float32)
  return mode,a,b,mu
 def response(self,raw,n,stall):
  with torch.no_grad():return float(self.policy.stall_gate(torch.as_tensor(np.asarray(raw[:5],np.float32),device=self.policy.device).reshape(1,5),torch.tensor([n],device=self.policy.device),torch.tensor([stall],device=self.policy.device)))
def configure(m):
 for name,p in m.named_parameters():p.requires_grad_(name.startswith('stall_gate.'))
 return torch.optim.Adam(m.stall_gate.parameters(),lr=3e-4)
def load_base(m,path):
 x=torch.load(path,map_location=m.device,weights_only=False);missing,unexpected=m.load_state_dict(x['model'],strict=False)
 assert not unexpected and all(n.startswith('stall_gate.') for n in missing)

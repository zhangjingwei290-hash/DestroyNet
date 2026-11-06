from pathlib import Path
import numpy as np,torch
from solver.model import Policy
from aligned_cpu import CpuHead
from v9_policy import Controller,NumpyController
F=Path('D:/workspace/destroy-repair-V0.01-publication/versions/v5_step5_r2_frozen_20261102')
class NewPolicy(Policy):
 def __init__(self,cfg):
  super().__init__(cfg);self.stall_gate=Controller()
class NewHead(CpuHead):
 def __init__(self,policy,legacy_feature_reductions=False):
  super().__init__(policy,legacy_feature_reductions);self.policy=policy;self.numpy_gate=NumpyController(policy.stall_gate);self.state=None;self.summary=None;self.cross_stall=0;self.last_repair_sec=0.;self.gate_enabled=True;self.q_bias=0.;self.gate_shift=0.
 def outputs(self,hidden,physical,raw,n,dynamic_position,task=0):
  mode,a,b,mu=super().outputs(hidden,physical,raw,n,dynamic_position,task)
  self.base_mode_logits=mode.copy();self.base_alpha=a.copy();self.base_beta=b.copy()
  y=self.numpy_gate(self.gate_context);self.gate_shift=float(y[3]);mode=mode+y[:3]
  c=a+b;lo=(1+1e-6)/c;base_center=a/c;logit_center=np.log(base_center)-np.log1p(-base_center);center=np.clip(1/(1+np.exp(-np.clip(logit_center+y[3],-80,80))),lo,1-lo)
  return mode.astype(np.float32),(center*c).astype(np.float32),((1-center)*c).astype(np.float32),mu

def load_base(model,path):
 # Preserve each requested direction's own original expert; never use V2's shared initialization.
 p=F/'checkpoints'/Path(path).name;sd=torch.load(p,map_location=model.device,weights_only=False)['model']
 missing,unexpected=model.load_state_dict(sd,strict=False)
 assert not unexpected and all(k.startswith('stall_gate.') for k in missing)
 model.expert_source=str(p)

def summary_for(env,archive):
 points=np.asarray([p['objectives'] for p in archive],dtype=float);delta=(points-env.objectives)/env.dynamic_span
 dist=np.linalg.norm(delta,axis=1);other=dist>1e-10
 return np.asarray([np.tanh(env.score(env.objectives)-min(env.score(x) for x in points)),np.tanh(float(dist[other].min()) if other.any() else 0.),float(other.any())],np.float32)

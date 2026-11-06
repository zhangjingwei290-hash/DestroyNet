"""V2 parameter/feature schema; optional q-mean intervention only."""
from pathlib import Path
import importlib.util,numpy as np
P=Path('D:/workspace/fuxian/DPR/development/sa5_coverage_v2_20261104')
spec=importlib.util.spec_from_file_location('v2_exact_network',P/'network.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
NewPolicy=legacy.NewPolicy
summary_for=legacy.summary_for
configure=legacy.configure
class NewHead(legacy.NewHead):
 def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.q_bias=0.
 def outputs(self,*args,**kwargs):
  mode,a,b,mu=super().outputs(*args,**kwargs)
  if self.q_bias:
   concentration=a+b;center=a/concentration
   lo=(1.+1e-6)/concentration;hi=1.-lo
   center=np.clip(center+np.float32(self.q_bias),lo,hi)
   a=(center*concentration).astype(np.float32);b=((1.-center)*concentration).astype(np.float32)
  return mode,a,b,mu

from pathlib import Path
import sys,ast,json
import numpy as np,torch
O=Path(__file__).resolve().parent
F=Path('D:/workspace/destroy-repair-V0.01-publication/versions/v5_step5_r2_frozen_20261102')
sys.path[:0]=[str(O),str(F/'evaluation_source'),str(F/'evaluation_source/stage2_repo')]
from opportunity import OpportunityEMA,balanced_advantages
from network import Gate,gate_features

def main():
 torch.manual_seed(11)
 # Causal reward: with constant historical rate, longer sessions pay more.
 score=lambda x,w:float(np.max(np.asarray(x)*w))
 halfscore=lambda x,w:score(x,w)/2
 e=OpportunityEMA(window=3);r=dict(before=[2]*5,after=[1]*5,weight=[1]*5)
 assert e.value(score)==0
 e.observe([r],2);assert e.value(score)==.25
 assert e.value(halfscore)==.125,'History must rebase with current scalar coordinates'
 rate=e.value(score);assert 1-rate*4<1-rate*2
 for _ in range(30):e.observe([dict(r,after=[2]*5)],2)
 assert e.value(score)==0
 # Unequal per-scale counts still contribute equal weight; zero variance finite.
 ns=[50]*3+[150]*4+[250]*5;y=torch.arange(12,dtype=torch.float32)
 adv,w=balanced_advantages(y,torch.zeros_like(y),ns)
 for n in (50,150,250):
  ii=[i for i,k in enumerate(ns) if k==n];assert abs(float(w[ii].sum())-1/3)<1e-6;assert abs(float(adv[ii].mean()))<1e-6
 try:balanced_advantages(y[:3],y[:3],[50]*3);raise AssertionError('Missing scales accepted')
 except ValueError:pass
 # Forward compatibility, neutral new features/FiLM, then trainable conditions.
 old=torch.nn.Sequential(torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,1));sd={'body.'+k:v for k,v in old.state_dict().items()};g=Gate();g.load_v7(sd)
 x=torch.randn(12,34);expected=.15*torch.tanh(old(x[:,:32]).squeeze(-1));assert torch.allclose(g(x),expected,atol=2e-8)
 out=g(x);loss=(out-y/12).square().mean();loss.backward();assert g.film[-1].weight.grad.abs().sum()>0 and g.body[0].weight.grad[:,-2:].abs().sum()>0
 z=gate_features(np.zeros(15),np.zeros(20),150,6,np.zeros(7),np.zeros(3),.3,30,10);assert len(z)==34 and abs(z[-2]-.01)<1e-8 and abs(z[-1]-.3)<1e-7
 # AST guard: optimizer steps outside scale collection; all inside PPO epoch.
 tree=ast.parse((O/'train_joint.py').read_text());steps=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step']
 assert len(steps)==1
 for n in ast.walk(tree):
  if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='n':assert not any(x is steps[0] for x in ast.walk(n))
 result=dict(passed=True,causal_ema=True,dynamic_coordinate_rebase=True,three_scale_equal_weights=True,incomplete_scales_rejected=True,identity_migration=True,film_and_new_features_receive_gradients=True,no_per_scale_optimizer_step=True)
 (O/'CONTRACT_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()

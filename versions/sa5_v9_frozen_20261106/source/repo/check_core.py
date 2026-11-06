from pathlib import Path
import os,json,sys
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np,torch,psutil
psutil.Process().cpu_affinity([6,7]);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261105)
F=Path('D:/workspace/destroy-repair-V0.01-publication/versions/v5_step5_r2_frozen_20261102')
sys.path[:0]=[str(F/'evaluation_source/stage2_repo'),str(F/'evaluation_source')]
from v9_policy import Controller,NumpyController,NumpyMLP,q_bounds,q_for_mode,select_route_prefix,CONTEXT_FIELDS
R=Path(__file__).resolve().parent.parent
bounds=[dict(n=n,mode=m,bounds=q_bounds(m,n,min(round(.45*n),100)),midpoint=q_for_mode(.5,m,n,min(round(.45*n),100))) for n in (50,150,250) for m in (0,1,2)]
for a in bounds:
 lo,hi=a['bounds'];assert 1<=lo<=a['midpoint']<=hi<=a['n']
 values=[q_for_mode(u,a['mode'],a['n'],min(round(.45*a['n']),100)) for u in np.linspace(0,1,101)];assert values==sorted(values)
c=Controller().cuda();nc=NumpyController(c);rng=np.random.default_rng(20261105);x=rng.uniform(-2,2,(4096,len(CONTEXT_FIELDS))).astype(np.float32)
x[:,1]=rng.choice([.2,.6,1.],len(x));x[:,2:7]=rng.dirichlet(np.ones(5),len(x)).astype(np.float32)
with torch.no_grad():a=c(torch.tensor(x,device='cuda')).cpu().numpy()
b=np.stack([nc(row) for row in x]);err=float(abs(a-b).max());assert err<1e-6,err
from gate_ppo import terms
rows=[]
for i in range(96):
 xx=x[i];base_a=np.array([3.,4.,5.],np.float32);base_b=np.array([5.,4.,3.],np.float32);base_m=np.array([.1,-.2,.3],np.float32)
 yy=nc(xx);cc=base_a+base_b;lo=(1+1e-6)/cc;center=np.clip(1/(1+np.exp(-(np.log(base_a/base_b)+yy[3]))),lo,1-lo);m=i%3
 rows.append(dict(gate_context=xx,base_alpha=base_a,base_beta=base_b,base_mode_logits=base_m,mode_logits=base_m+yy[:3],mode=m,u=float(rng.uniform(.1,.9)),alpha=float(center[m]*cc[m]),beta=float((1-center[m])*cc[m])))
lr,entropy=terms(c,rows,'cuda');assert float(lr.detach().abs().max())<1e-5
adv=torch.linspace(-1,1,len(rows),device='cuda');ratio=lr.exp();loss=(-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)-.01*entropy).mean();loss.backward()
grad=c.body[-1].weight.grad.norm(dim=1).detach().cpu().tolist();assert all(g>0 for g in grad),grad
assert all(p.grad is not None and p.grad.norm()>0 for p in c.parameters())
# Verify all actual original expert tensors are loaded from their own direction checkpoint.
experts=[];first=None;encoder_equal=True
for k in range(16):
 p=F/f'checkpoints/direction_{k:02d}.pt';sd=torch.load(p,map_location='cpu',weights_only=False)['model']
 if first is None:first=sd
 encoder_equal &= all(torch.equal(v,first[n]) for n,v in sd.items() if n.startswith('encoder.'))
 experts.append(dict(direction=k,path=str(p),physical=sd['heads.0.physical_bilinear'].tolist(),route=sd['heads.0.route_bilinear'].tolist()))
distinct_physical=len({json.dumps(t['physical']) for t in experts});distinct_route=len({json.dumps(t['route']) for t in experts})
# Replay generic lightweight blocks with a fixed CPU input to catch transpose/activation errors.
seq=torch.nn.Sequential(torch.nn.Linear(len(CONTEXT_FIELDS),32),torch.nn.Tanh(),torch.nn.Linear(32,8),torch.nn.LayerNorm(8),torch.nn.ELU()).cuda();ns=NumpyMLP(seq)
with torch.no_grad():a=seq(torch.tensor(x[:100],device='cuda')).cpu().numpy()
e=float(np.max(abs(a-ns(x[:100]))));assert e<2e-6,e
sel,removed,over=select_route_prefix([[1,2,3],[4,5,6],[7,8]],[0,1,2],np.arange(8),5,6);assert removed==[1,2,3,4,5,6] and sel==[0,1] and not over
receipt=dict(passed=True,searches=0,controller_features=list(CONTEXT_FIELDS),controller_params=sum(p.numel() for p in c.parameters()),max_numpy_gpu_error=err,mlp_numpy_gpu_error=e,all_controller_layers_nonzero_gradient=True,ppo_output_gradient_norms=grad,expert_count=len(experts),distinct_physical_matrices=distinct_physical,distinct_route_matrices=distinct_route,original_encoder_equal=encoder_equal,q_bounds=bounds,notes=['All modes share physical bounds; controller selects mode and grain without scale priors.','First whole route can exceed target/cap; matches original protection behavior and must be logged.','These are implementation checks, not policy behavior or HV results.'])
(R/'CORE_CHECK.json').write_text(json.dumps(receipt,indent=2));(R/'EXPERT_SOURCES.json').write_text(json.dumps(experts,indent=2));print(json.dumps(receipt),flush=True)

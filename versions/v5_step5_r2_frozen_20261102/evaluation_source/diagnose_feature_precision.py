"""Finite comparison of CPU/GPU physical state reductions on preserved FIT archive states."""
from stage2_common import ROOT,read,write,load_cfg,FitPool,reset
import math

def main():
 import psutil,numpy as np,torch
 from solver.model import Policy
 from solver.environment import Environment,Adaptive
 from stage2_cpu import CpuHead,struct_state
 from stage1_mechanisms import physical_features,dynamic_coordinates
 psutil.Process().cpu_affinity([0,1]);torch.set_num_threads(1);torch.set_num_interop_threads(1);reset(20261024)
 folder=ROOT/'engineering/replay_guard_direction05_batch003';cfg=load_cfg();pool=FitPool(cfg);p=Policy(cfg).cuda();p.load_state_dict(torch.load(folder/'recovered_direction05_batch002.pt',map_location='cuda',weights_only=False)['model']);cpu=CpuHead(p)
 failure=torch.load(ROOT/'stage2_training/failed_training_state.pt',map_location='cpu',weights_only=False);iid='v5p_train_50_d1_tw4_3';slot=next(s for s in failure['progressive_pool']['slots'] if s['iid']==iid and s['target']==5);archive=failure['progressive_pool']['archives'][iid];print('archive_points',len(archive),flush=True)
 physical=[];raws=[];positions=[];hidden=[];cpu_values=[];indices=[]
 for index,point in enumerate(archive[:128]):
  env=Environment(pool,iid,cfg,slot['preference'],point['routes']);env.archive=archive;env.stagnation=9
  env.dynamic_ideal,env.dynamic_span=dynamic_coordinates([x['objectives'] for x in archive],env.scales,env.offset+env.scales*np.asarray(env.box['reference']))
  obs=env.observation(torch.tensor(slot['hidden']),'greedy',Adaptive(len(env.q_values)),100.,139.,2.5,2.5);phys=obs['x'][1:,[18,17,19,12,13]].numpy();raw=obs['context'].numpy();z=(env.objectives-env.dynamic_ideal)/env.dynamic_span;pos=np.asarray([z.mean(),(env.preference*z).max()],dtype=np.float32)
  physical.append(phys);raws.append(raw);positions.append(pos);hidden.append(np.asarray(slot['hidden']).reshape(-1));cpu_values.append(cpu.outputs(hidden[-1],struct_state(phys),raw,50,pos,0));indices.append(index)
 pp=torch.tensor(np.stack(physical),device='cuda');h=torch.tensor(np.stack(hidden),device='cuda');rr=torch.tensor(np.stack(raws),device='cuda');zz=torch.tensor(np.stack(positions),device='cuda')
 with torch.no_grad():gpu=p.q_from_physical(h,pp,rr,[0]*len(indices),zz)
 errors=[]
 for j in range(len(indices)):
  e=[float(np.max(abs(gpu[k][j].cpu().numpy()-cpu_values[j][k]))) for k in range(4)];errors.append(dict(archive_index=j,mode_logits=e[0],alpha=e[1],beta=e[2],mu=e[3]))
 # Explicit constant/near-constant actual-scale feature probes isolate numerical variance cancellation.
 fixtures=[]
 for value in (.1,.3,.7,1.1,3.3,7.7):
  x=np.full((50,5),np.float32(value));x[0,0]=np.nextafter(np.float32(value),np.float32(100.))
  t=torch.tensor(x,device='cuda')[None];mean=t.mean(1);sigma=t.std(1,unbiased=False);cv=(sigma/t.abs().mean(1).clamp_min(1e-4)).clamp(0,4)/4;peak=((t.amax(1)-mean)/sigma.clamp_min(1e-4)).clamp(0,6)/6;k=5;share=(t.topk(k,dim=1).values.abs().sum(1)/t.abs().sum(1).clamp_min(1e-4)).clamp(0,1)
  state=torch.cat((cv,peak,share),-1)[0].cpu().numpy();c=struct_state(x);fixtures.append(dict(value=value,max_state_difference=float(abs(state-c).max()),cpu_state=c.tolist(),gpu_state=state.tolist()))
 write(folder/'FEATURE_PRECISION_DIAGNOSTIC.json',dict(actual_archive_points=len(errors),worst_actual=sorted(errors,key=lambda x:max(x['alpha'],x['beta']),reverse=True)[:10],max_actual_alpha_beta_error=max(max(x['alpha'],x['beta']) for x in errors),constant_probes=fixtures,new_trajectories=0))
 print(read(folder/'FEATURE_PRECISION_DIAGNOSTIC.json'),flush=True)
if __name__=='__main__':main()

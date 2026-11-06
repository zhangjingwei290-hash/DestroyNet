from pathlib import Path
import os,sys,time,json,hashlib,subprocess,traceback
O=Path(__file__).resolve().parent;F=Path('D:/workspace/destroy-repair-V0.01-publication/versions/v5_step5_r2_frozen_20261102');P=O.parent/'sa5_coverage_v2_20261104';V=O.parent/'sa5_state_action_v6_stage2_20261105'
sys.path[:0]=[str(O),str(F/'evaluation_source'),str(F/'evaluation_source/stage2_repo')]
def write(name,x):
 p=O/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2),encoding='utf-8');os.replace(q,p)
def case(iid,index,gate,rollout):
 import psutil
 root=O/f'cases/{index}';root.mkdir(parents=True,exist_ok=False);env=dict(os.environ,CASE_ROOT=str(root),CASE_IID=iid,CASE_SEED='20261105',CASE_SECONDS='30',METHODS='["gate_trained"]',METHOD_WEIGHTS='{}',STAGE_MANIFEST=str(O/'SOURCE_MANIFEST.json'),CHECKPOINT_ROOT=str(P/'training'),GATE_PATH=str(gate),TRAIN_CAPTURE='1')
 if rollout:env['GLOBAL_ROLLOUT']='1'
 with (root/'stdout.log').open('wb') as out,(root/'stderr.log').open('wb') as err:
  c=subprocess.Popen([sys.executable,'-B','-X','utf8',str(O/'evaluate_case.py')],env=env,stdout=out,stderr=err);p=psutil.Process(c.pid);write('active_child.json',dict(pid=p.pid,created=p.create_time(),cmd=p.cmdline(),case=index));code=c.wait()
 assert code==0,f'case{index} failed';assert json.loads((root/'analysis/AUDIT.json').read_text())['passed'];return root

def main():
 import torch,numpy as np,psutil
 from run import load_config
 from network import NewPolicy,load_base,Gate
 from learning import joint_logprob
 p=psutil.Process();p.cpu_affinity([0,1,2,3]);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261105);start=time.perf_counter();fd=os.open(O/'claim.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd);write('process.json',dict(pid=p.pid,created=p.create_time(),cmd=p.cmdline()))
 fs=list(O.glob('*.py'))+[O/'PROTOCOL.md',V/'training/gate.pt',P/'network.py']+list((P/'training').glob('direction_*.pt'));write('SOURCE_MANIFEST.json',dict(files=[dict(windows_path=str(x),sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in fs]))
 cfg=load_config(F/'evaluation_source/eval_config.json');gate=Gate().cuda();gate.load_state_dict(torch.load(V/'training/gate.pt',map_location='cuda',weights_only=False));critic=torch.nn.Sequential(torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,1)).cuda();torch.nn.init.zeros_(critic[-1].weight);torch.nn.init.zeros_(critic[-1].bias);opt=torch.optim.Adam(list(gate.parameters())+list(critic.parameters()),lr=3e-4);models=[]
 for k in range(16):
  m=NewPolicy(cfg).cuda();load_base(m,P/f'training/direction_{k:02d}.pt');m.requires_grad_(False);m.stall_gate=gate;models.append(m)
 torch.save(gate.state_dict(),O/'initial_gate.pt');torch.save(gate.state_dict(),O/'gate.pt');history=[]
 for cycle in range(2):
  for n in (50,150,250):
   assert time.perf_counter()-start<1500;index=f'train_{cycle}_{n}';write('progress.json',dict(phase='full_host_training',case=index,completed_updates=len(history)));root=case(f'v5p_train_{n}_d0_tw0_0',index,O/'gate.pt',True);d=torch.load(root/'global_rollout.pt',weights_only=False);rows=d['rows'];assert d['batches']>1 and len(rows)>80
   x=torch.as_tensor(np.stack([r['gate_context'] for r in rows]),device='cuda');target=torch.tensor([d['final_quality']-r['global_quality'] for r in rows],device='cuda',dtype=torch.float32)
   with torch.no_grad():adv=target-critic(x).squeeze(-1);adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
   maxerr=0.
   for epoch in range(2):
    opt.zero_grad();loss_sum=0.
    for k,m in enumerate(models):
     ids=[i for i,r in enumerate(rows) if r['expert']==k]
     for st in range(0,len(ids),16):
      ii=ids[st:st+16];batch=[rows[i] for i in ii];terms=m.replay_terms(batch);lp=joint_logprob(terms);old=torch.tensor([r['logprob_macro']+r['logprob_micro'] for r in batch],device='cuda');err=float((lp.detach()-old).abs().max())
      if epoch==0:maxerr=max(maxerr,err);assert err<2e-3,f'replay error{err}'
      ratio=(lp-old).exp();a=adv[ii];value=critic(x[ii]).squeeze(-1);loss=(-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a)+.5*(value-target[ii]).square()-.01*terms['entropy']).sum()/len(rows);loss.backward();loss_sum+=float(loss.detach())
    torch.nn.utils.clip_grad_norm_(list(gate.parameters())+list(critic.parameters()),1.);opt.step()
   history.append(dict(case=index,actions=len(rows),outer_batches=d['batches'],budget=d['budget'],actual_seconds=d['actual'],final_quality=d['final_quality'],mean_remaining_return=float(target.mean()),max_density_error=maxerr,loss=loss_sum));torch.save(gate.state_dict(),O/'gate.pt');torch.save(critic.state_dict(),O/'critic.pt');write('training_history.json',history)
 validation=[]
 for n in (50,150,250):
  for kind,path in [('initial',O/'initial_gate.pt'),('trained',O/'gate.pt')]:
   index=f'heldout_{n}_{kind}';write('progress.json',dict(phase='heldout_full_host',case=index));root=case(f'v5p_train_{n}_d1_tw2_0',index,path,True);d=torch.load(root/'global_rollout.pt',weights_only=False);validation.append(dict(n=n,kind=kind,quality=d['final_quality'],batches=d['batches'],actions=len(d['rows']),seconds=d['actual']));write('validation.json',validation)
 write('COMPLETE.json',dict(passed=True,training_runs=6,heldout_runs=6,seconds=time.perf_counter()-start,full_host_return=True,no_local_terminal_bootstrap=True,validation=validation))
if __name__=='__main__':
 try:main()
 except BaseException:write('FAILURE.json',dict(error=traceback.format_exc()));raise

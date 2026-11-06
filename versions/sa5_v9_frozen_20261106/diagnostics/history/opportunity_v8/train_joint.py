"""Collect all scales with one frozen behavior checkpoint, then one joint PPO update.
Default is a bounded integration check (3 x 6 seconds, one cycle), not an HV study.
"""
from pathlib import Path
import os,sys,time,json,hashlib,subprocess,traceback,argparse
O=Path(__file__).resolve().parent
F=Path('D:/workspace/destroy-repair-V0.01-publication/versions/v5_step5_r2_frozen_20261102')
P=O.parent/'sa5_coverage_v2_20261104';V=O.parent/'sa5_state_action_v6_stage2_20261105'
sys.path[:0]=[str(O),str(F/'evaluation_source'),str(F/'evaluation_source/stage2_repo')]
def write(root,name,x):
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2),encoding='utf-8');os.replace(q,p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 import torch,numpy as np,psutil
 from run import load_config
 from network import Gate,NewPolicy,load_base
 from learning import joint_logprob
 from opportunity import balanced_advantages
 a=argparse.ArgumentParser();a.add_argument('--seconds',type=float,default=6);a.add_argument('--cycles',type=int,default=1);a.add_argument('--output',default='technical_check');args=a.parse_args()
 if args.seconds<=0 or args.cycles<1:raise ValueError('Positive budgets required')
 root=O/args.output;root.mkdir(parents=True,exist_ok=False);fd=os.open(root/'claim.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
 proc=psutil.Process();proc.cpu_affinity([0,1,2,3]);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261105)
 write(root,'process.json',dict(pid=proc.pid,created=proc.create_time(),cmd=proc.cmdline()));start=time.perf_counter()
 fs=list(O.glob('*.py'))+[O/'PROTOCOL.md',V/'training/gate.pt',P/'network.py']+list((P/'training').glob('direction_*.pt'))
 write(root,'SOURCE_MANIFEST.json',dict(files=[dict(windows_path=str(x),sha256=sha(x)) for x in fs]))
 gate=Gate().cuda();gate.load_v7(torch.load(V/'training/gate.pt',map_location='cuda',weights_only=False))
 critic=torch.nn.Sequential(torch.nn.Linear(34,32),torch.nn.Tanh(),torch.nn.Linear(32,1)).cuda();torch.nn.init.zeros_(critic[-1].weight);torch.nn.init.zeros_(critic[-1].bias)
 params=list(gate.parameters())+list(critic.parameters());opt=torch.optim.Adam(params,lr=3e-4);models=[];cfg=load_config(F/'evaluation_source/eval_config.json')
 for k in range(16):
  m=NewPolicy(cfg).cuda();load_base(m,P/f'training/direction_{k:02d}.pt');m.requires_grad_(False);m.stall_gate=gate;models.append(m)
 torch.save(gate.state_dict(),root/'initial_gate.pt');history=[];steps=0
 for cycle in range(args.cycles):
  behavior=root/f'behavior_{cycle}.pt';torch.save(gate.state_dict(),behavior);behavior_sha=sha(behavior);rows=[];scale_counts={};collected=[]
  # Collection can be serial; no gradient update is allowed inside this loop.
  for n in (50,150,250):
   case=root/f'cases/cycle{cycle}_{n}';case.mkdir(parents=True)
   env=dict(os.environ,CASE_ROOT=str(case),CASE_IID=f'v5p_train_{n}_d0_tw0_0',CASE_SEED=str(20261105+cycle),CASE_SECONDS=str(args.seconds),METHODS='["gate_trained"]',METHOD_WEIGHTS='{}',STAGE_MANIFEST=str(root/'SOURCE_MANIFEST.json'),CHECKPOINT_ROOT=str(P/'training'),GATE_PATH=str(behavior),TRAIN_CAPTURE='1',SESSION_ROLLOUT='1')
   write(root,'progress.json',dict(phase='collect',cycle=cycle,n=n,optimizer_steps=steps,behavior_sha=behavior_sha))
   with (case/'stdout.log').open('wb') as out,(case/'stderr.log').open('wb') as err:
    c=subprocess.Popen([sys.executable,'-B','-X','utf8',str(O/'evaluate_case.py')],env=env,stdout=out,stderr=err);cp=psutil.Process(c.pid);write(root,'active_child.json',dict(pid=cp.pid,created=cp.create_time(),cmd=cp.cmdline(),n=n));code=c.wait()
   assert code==0,f'Collection failed {case} exit {code}'
   assert json.loads((case/'analysis/AUDIT.json').read_text())['passed']
   assert sha(behavior)==behavior_sha,'Behavior changed during collection'
   d=torch.load(case/'session_rollout.pt',weights_only=False);assert d['rows'],f'No complete sessions N{n}'
   for r in d['rows']:assert r['n']==n and len(r['gate_context'])==34
   for s in d['sessions']:assert abs(s['net_return']-(s['gain']-s['v_rate']*s['seconds']))<1e-12
   scale_counts[n]=len(d['rows']);rows.extend(d['rows']);collected.append(dict(n=n,actions=len(d['rows']),sessions=sum(s['complete'] for s in d['sessions']),partial_actions=d['excluded_partial_actions'],behavior_sha=behavior_sha))
  assert set(scale_counts)=={50,150,250}
  x=torch.as_tensor(np.stack([r['gate_context'] for r in rows]),device='cuda');target=torch.tensor([r['session_return'] for r in rows],device='cuda',dtype=torch.float32)
  with torch.no_grad():adv,weights=balanced_advantages(target,critic(x).squeeze(-1),[r['n'] for r in rows])
  density_errors=[];epoch_stats=[]
  for epoch in range(2):
   opt.zero_grad();loss_sum=0.
   for n in (50,150,250):
    for k,m in enumerate(models):
     ids=[i for i,r in enumerate(rows) if r['expert']==k and r['n']==n]
     for st in range(0,len(ids),16):
      ii=ids[st:st+16];batch=[rows[i] for i in ii];terms=m.replay_terms(batch);lp=joint_logprob(terms);old=torch.tensor([r['logprob_macro']+r['logprob_micro'] for r in batch],device='cuda')
      if epoch==0:
       error=float((lp.detach()-old).abs().max());density_errors.append(error);assert error<2e-3,f'Replay density mismatch {error}'
      ratio=(lp-old).exp();aa=adv[ii];value=critic(x[ii]).squeeze(-1)
      loss=((-torch.minimum(ratio*aa,ratio.clamp(.8,1.2)*aa)+.5*(value-target[ii]).square()-.01*terms['entropy'])*weights[ii]).sum()
      loss.backward();loss_sum+=float(loss.detach())
   assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
   grad=float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step();steps+=1;epoch_stats.append(dict(epoch=epoch,loss=loss_sum,grad_norm=grad,scales=[50,150,250]))
  diagnostics=[]
  for n in (50,150,250):
   ids=[i for i,r in enumerate(rows) if r['n']==n];diagnostics.append(dict(n=n,count=len(ids),weight=float(weights[ids].sum()),adv_mean=float(adv[ids].mean()),adv_sd=float(adv[ids].std(unbiased=False)),mean_return=float(target[ids].mean())))
  history.append(dict(cycle=cycle,collected=collected,behavior_sha=behavior_sha,epochs=epoch_stats,max_density_error=max(density_errors),scale_diagnostics=diagnostics));write(root,'training_history.json',history);torch.save(gate.state_dict(),root/'gate.pt');torch.save(critic.state_dict(),root/'critic.pt')
 write(root,'COMPLETE.json',dict(passed=True,technical_check=True,cycles=args.cycles,optimizer_steps=steps,collection_runs=3*args.cycles,seconds=time.perf_counter()-start,hv_performance_claim=False,history=history))
if __name__=='__main__':
 try:main()
 except BaseException:
  # Preserve all case evidence; never dispatch retries automatically.
  (O/'LAST_FAILURE.json').write_text(json.dumps(dict(error=traceback.format_exc()),indent=2),encoding='utf-8');raise

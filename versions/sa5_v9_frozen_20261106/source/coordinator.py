from pathlib import Path
import os,sys,json,time,traceback,shutil
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'repo'))
from runner import F,write,run

def train():
 import torch,numpy as np
 from v9_policy import Controller,CONTEXT_FIELDS
 from gate_ppo import update,terms
 torch.set_num_threads(1);torch.manual_seed(20261105)
 gate=Controller().cuda();gate.load_state_dict(torch.load(R/'checkpoints/initial.pt',weights_only=False));critic=torch.nn.Sequential(torch.nn.Linear(len(CONTEXT_FIELDS),32),torch.nn.Tanh(),torch.nn.Linear(32,1)).cuda()
 opt=torch.optim.Adam(list(gate.parameters())+list(critic.parameters()),lr=3e-4)
 rng=np.random.default_rng(20261105);fit=sorted((F/'data/fit').glob('*.json'));assert len(fit)==180
 pools={n:[p.stem for p in fit if json.loads(p.read_text())['n_customers']==n] for n in (50,150,250)}
 chosen={n:rng.choice(ids,6,replace=False).tolist() for n,ids in pools.items()};write(R/'TRAIN_SELECTION.json',dict(pool=180,sampled=chosen,seed=20261105))
 current=R/'checkpoints/initial.pt';history=[]
 for cycle in range(6):
  rows=[]
  for n in (50,150,250):
   iid=chosen[n][cycle];root=R/f'training/cycle{cycle+1}/N{n}';write(R/'progress.json',dict(phase='training',cycle=cycle+1,total_cycles=6,n=n,iid=iid))
   run(iid,20261105+cycle,180,current,root,'fit',True)
   data=torch.load(root/'session_rollout.pt',weights_only=False);rows.extend(data['rows'])
  coverage={(n,s,k):sum(r['n']==n and r['stage']==s and r['expert']==k for r in rows) for n in (50,150,250) for s in range(3) for k in range(16)}
  assert all(coverage.values()),f'Missing joint batch strata: {[k for k,v in coverage.items() if not v]}'
  with torch.no_grad():lr,_=terms(gate,rows,'cuda');err=float(lr.abs().max());assert err<1e-4,err
  stat=update(gate,critic,opt,rows);stat.update(cycle=cycle+1,maximum_behavior_logratio_error=err,minimum_stratum_count=min(coverage.values()),strata=144)
  current=R/f'checkpoints/cycle_{cycle+1}.pt';torch.save(gate.state_dict(),current);history.append(stat);write(R/'training_history.json',history)
 write(R/'TRAINING_COMPLETE.json',dict(passed=True,effective_sampling_seconds=3240,joint_cycles=6,ppo_epochs_per_cycle=2,all_144_strata_per_update=True))

def evaluate():
 from reports import paired_metrics,make_report
 val=[];ids=sorted(p.stem for p in (F/'data/validation').glob('*.json'));assert len(ids)==45
 for cycle in (3,6):
  for idx,iid in enumerate(ids):
   write(R/'progress.json',dict(phase='validation',cycle=cycle,completed=len(val),total=90,iid=iid))
   rec=run(iid,20261105,180,R/f'checkpoints/cycle_{cycle}.pt',R/f'validation/cycle{cycle}/{iid}','validation');rec['method']=f'cycle{cycle}';val.append(rec)
   write(R/'validation_completed.json',val)
 metrics=paired_metrics(val);scores={c:sum(r['hv'] for r in metrics if r['method']==f'cycle{c}')/45 for c in (3,6)};best=max((3,6),key=lambda c:(scores[c],-c))
 selected=R/'checkpoints/v9_best_gate.pt';shutil.copyfile(R/f'checkpoints/cycle_{best}.pt',selected);write(R/'VALIDATION_SELECTION.json',dict(cycle=best,macro_hv=scores,selection='45 validation IID common reference; macro HV; earlier checkpoint tie'))
 old=R.parent/'sa5_v8_numpy_gate_9iid3seed_20261105';baseline=json.loads((old/'BASELINE_REUSE.json').read_text())['roulette'];v8=json.loads((old/'v8_completed.json').read_text());assert len(v8)==27
 for r in v8:r['method']='v8'
 plan=json.loads((old/'plan.json').read_text());new=[]
 for iid in plan['test_ids']:
  for seed in plan['test_seeds']:
   write(R/'progress.json',dict(phase='test',completed=len(new),total=27,iid=iid,seed=seed))
   new.append(run(iid,seed,plan['budgets'][iid],selected,R/f'test/{iid}/{seed}','test'));write(R/'v9_completed.json',new)
 make_report(baseline+v8+new,R/'analysis','V9 vs V8 vs roulette: 9 IID x 3 seeds')
 write(R/'COMPLETE.json',dict(passed=True,new_tests=27,report='analysis/REPORT.md'))

def main():
 import psutil
 fd=os.open(R/'claim.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd);p=psutil.Process();p.cpu_affinity([6,7]);write(R/'process.json',dict(pid=p.pid,created=p.create_time(),cmd=p.cmdline()))
 train();evaluate()
if __name__=='__main__':
 try:main()
 except BaseException:write(R/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise

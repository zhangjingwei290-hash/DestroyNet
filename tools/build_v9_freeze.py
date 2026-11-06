"""Package completed V9 files and compact records without solver execution."""
from pathlib import Path
import os,json,hashlib,shutil,gzip,csv,collections
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import psutil
psutil.Process().cpu_affinity([14,15])
PUB=Path('D:/workspace/destroy-repair-V0.01-publication');BASE=Path('D:/workspace/fuxian/DPR/runs/experiments');R=BASE/'sa5_v9_scale_preference_20261105';F=PUB/'versions/v5_step5_r2_frozen_20261102';V=PUB/'versions/sa5_v9_frozen_20261106';OUT=Path('D:/workspace/outputs');DEV=Path('D:/workspace/fuxian/DPR/development')
assert not (V/'FREEZE_DECLARATION.json').exists()
V.mkdir(exist_ok=True);COPIES=[];TRANS=[]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cv(x):
 if hasattr(x,'tolist'):return x.tolist()
 if hasattr(x,'item'):return x.item()
 raise TypeError(type(x).__name__)
def copy(p,rel):
 p=Path(p);t=V/rel;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t);assert sha(p)==sha(t)
 COPIES.append(dict(path=t.relative_to(PUB).as_posix(),source=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 return t
for p in sorted((R/'repo').glob('*.py')):copy(p,'source/repo/'+p.name)
copy(R/'coordinator.py','source/coordinator.py')
for name in ['plan.json','PROTOCOL.md','SOURCE_MANIFEST.json','BASELINE_REUSE.json','TRAIN_SELECTION.json','training_history.json','TRAINING_COMPLETE.json','VALIDATION_SELECTION.json','COMPLETE.json','DELIVERY.json','CORE_CHECK.json','PPO_SMOKE.json','SMOKE.json','EXPERT_SOURCES.json']:
 copy(R/name,'evidence/'+name)
for p in sorted((R/'checkpoints').glob('*.pt')):copy(p,'checkpoints/controller/'+p.name)
for p in sorted((F/'checkpoints').glob('direction_*.pt')):copy(p,'checkpoints/experts/'+p.name)
# The exact shared dependencies are already immutable in the previous version.
shared=[]
for x in read(R/'SOURCE_MANIFEST.json')['files']:
 p=Path(x['windows_path']);assert sha(p)==x['sha256'],str(p)
 try:relative=p.relative_to(F)
 except ValueError:continue
 shared.append(dict(source=str(p),repository_path='../v5_step5_r2_frozen_20261102/'+relative.as_posix(),sha256=x['sha256']))
dump(V/'evidence/SHARED_DEPENDENCIES.json',shared)
# Input roles are references to committed original bytes, with the actual selection retained above.
dump(V/'data/INPUT_ROLES.json',dict(fit_pool='../v5_step5_r2_frozen_20261102/data/fit',fit_pool_count=180,sampled_training_iids=read(R/'TRAIN_SELECTION.json')['sampled'],sampled_count=18,validation='../v5_step5_r2_frozen_20261102/data/validation',validation_iids=sorted(p.stem for p in (F/'data/validation').glob('*.json')),test_iids=read(R/'plan.json')['test_ids'],test_source='../v5_step5_r2_frozen_20261102/data/test'))
# Compact archives preserve final routes/objectives and initial populations; drop long event streams.
DROP={'records','batches','point_events','archive_snapshots'}
def compact(src,dest,method,stage):
 d=read(src);assert d.get('physical_audit_passed',d.get('independent_physical_audit',{}).get('passed',False))
 selected={k:v for k,v in d.items() if k not in DROP};selected['published_method']=method
 t=V/dest;t.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(t,'wt',encoding='utf-8') as f:json.dump(selected,f,separators=(',',':'),ensure_ascii=False,default=cv)
 TRANS.append(dict(path=t.relative_to(V).as_posix(),source=str(src),source_sha256=sha(src),published_sha256=sha(t),method=method,stage=stage,iid=d['iid'],seed=d['seed'],omitted_fields=sorted(DROP.intersection(d)),records=len(d.get('records',[])),batches=len(d.get('batches',[]))))
 return d
base=read(R/'BASELINE_REUSE.json');tests=base['roulette']+base['v8']+read(R/'v9_completed.json')
for x in tests:compact(Path(x['path']),f"results/test/{x['method']}/{x['iid']}_seed{x['seed']}.json.gz",x['method'],'test')
for x in read(R/'validation_completed.json'):compact(Path(x['path']),f"results/validation/{x['method']}/{x['iid']}_seed{x['seed']}.json.gz",x['method'],'validation')
# Real compact PPO rows contain all changing factors; no model forward pass is needed to serialize them.
import torch
torch.set_num_threads(1)
train=[]
for cycle in range(1,7):
 for n in (50,150,250):
  root=R/f'training/cycle{cycle}/N{n}';src=root/'session_rollout.pt';roll=torch.load(src,map_location='cpu',weights_only=False)
  dst=V/f'training/rollouts/cycle{cycle}_N{n}.json.gz';dst.parent.mkdir(parents=True,exist_ok=True)
  payload={k:v for k,v in roll.items() if k!='full_replay_proof'}
  with gzip.open(dst,'wt',encoding='utf-8') as f:json.dump(payload,f,default=cv,separators=(',',':'))
  TRANS.append(dict(path=dst.relative_to(V).as_posix(),source=str(src),source_sha256=sha(src),published_sha256=sha(dst),stage='training_replay',cycle=cycle,n=n,rows=len(roll['rows']),omitted_fields=['full_replay_proof']))
  result=compact(root/'results/gate_trained.json',f'results/training/cycle{cycle}_N{n}.json.gz','v9','training');train.append(dict(cycle=cycle,n=n,iid=result['iid'],seed=result['seed'],budget=result['budget_seconds'],online_seconds=result['online_seconds'],sampled_rows=len(roll['rows']),excluded_partial_actions=roll['excluded_partial_actions']))
  copy(root/'analysis/AUDIT.json',f'evidence/audits/training/cycle{cycle}_N{n}.json')
for x in read(R/'validation_completed.json'):copy(Path(x['path']).parent.parent/'analysis/AUDIT.json',f"evidence/audits/validation/{x['method']}_{x['iid']}.json")
for x in tests:copy(Path(x['path']).parent.parent/'analysis/AUDIT.json',f"evidence/audits/test/{x['method']}_{x['iid']}_seed{x['seed']}.json")
for n in (50,150,250):
 root=R/f'smoke/N{n}';compact(root/'results/gate_trained.json',f'results/smoke/N{n}.json.gz','v9_untrained','smoke');copy(root/'analysis/AUDIT.json',f'evidence/audits/smoke/N{n}.json')
dump(V/'training/COLLECTION_COUNTS.json',train)
for p in sorted((R/'analysis').iterdir()):
 if p.suffix in ('.csv','.json','.md'):copy(p,'analysis/main/'+p.name)
# Historical offline comparisons are separate reference families.
for name,folder in [('v9_vs_r2_longsession_9iid_20261106','r2_long'),('v9_vs_dbmt_native_9iid_20261106','dbmt_native')]:
 for p in sorted((OUT/name).iterdir()):
  if p.suffix in ('.csv','.json','.md','.py'):copy(p,'analysis/historical/'+folder+'/'+p.name)
if (BASE/'dbmt_ls9_3seed_fullr2_20261103/results/dbmt_original_ls').exists():
 for p in sorted((BASE/'dbmt_ls9_3seed_fullr2_20261103/results/dbmt_original_ls').glob('*.json')):compact(p,'results/historical_dbmt/'+p.stem+'.json.gz','dbmt_original','historical')
# Historical diagnostic JSON/CSV and source scripts, separated from V9's own checks.
hist=[(OUT/'sa5_frozen_response_20261104','frozen_response'),(OUT/'sa5_action_value_diagnostic_20261104','action_value'),(OUT/'sa5_coverage_v2_response_20261104','coverage_v2_response'),(OUT/'sa5_time_objective_audit_20261105','time_objective'),(DEV/'sa5_richstate_v3_20261104','richstate_v3'),(DEV/'sa5_rloo_stallq_v4_20261104','rloo_stallq_v4'),(DEV/'sa5_cross_session_v5_20261105','cross_session_v5'),(DEV/'sa5_state_action_v6_20261105','state_action_v6'),(DEV/'sa5_state_action_v6_stage2_20261105','state_action_v6_stage2'),(DEV/'sa5_global_return_v7_20261105','global_return_v7'),(OUT/'sa5_global_return_v7_results_20261105','global_return_v7_results'),(DEV/'sa5_opportunity_v8_20261105','opportunity_v8')]
WHITELIST={'summary.json','AUDIT.json','ANALYSIS_AUDIT.json','COMPLETE.json','COMPLETE.json','range_summary.json','comparisons.csv','old_new_comparison.json','candidate_panel.csv','continuations.csv','continuation_summary.json','crossfit.json','fixed_preference.csv','fixed_preference_summary.json','FIXED_PREFERENCE_AUDIT.json','heldout_summary.json','stall_bins.json','response.json','CONTRACT_CHECK.json','IMPLEMENTATION_AUDIT.json'}
for root,label in hist:
 for p in sorted(root.iterdir()):
  if p.is_file() and (p.name in WHITELIST or p.suffix=='.py'):copy(p,'diagnostics/history/'+label+'/'+p.name)
 for name in ['training/COMPLETE.json','response/AUDIT.json','response/summary.json','technical_check_corrected/COMPLETE.json','technical_check_corrected/IMPLEMENTATION_AUDIT.json']:
  if (root/name).exists():copy(root/name,'diagnostics/history/'+label+'/'+name)
for parent,label in [(BASE/'sa5_v8_joint18_20261105','v8_joint18'),(BASE/'sa5_v8_numpy_gate_9iid3seed_20261105','v8_numpy')]:
 for name in ['NUMERICAL_EQUIVALENCE.json','TRAINING_COMPLETE.json','VALIDATION_SELECTION.json','training_history.json','COMPLETE.json','V2_SUPPLEMENT_CANCELLED_BY_USER.json']:
  if (parent/name).exists():copy(parent/name,'diagnostics/history/'+label+'/'+name)
# Bind selected deployment assets and preceding GitHub commit.
gate=V/'checkpoints/controller/v9_best_gate.pt';assert sha(gate)==sha(V/'checkpoints/controller/cycle_3.pt')
dump(V/'FREEZE_DECLARATION.json',dict(version=V.name,date='2026-11-06',previous_github_commit='b16ca2318add176e8f9a81101f141f4f8d90f65e',previous_version=F.name,source_run=str(R),selected_controller='checkpoints/controller/v9_best_gate.pt',selected_cycle=3,selected_sha256=sha(gate),expert_files=[dict(path=p.relative_to(V).as_posix(),sha256=sha(p),unchanged_from_previous=True) for p in sorted((V/'checkpoints/experts').glob('*.pt'))],controller_parameters=868,critic_parameters=769,optimizer_steps=12,joint_cycles=6,training_rows=sum(x['sampled_rows'] for x in train),training_iids=18,fit_pool=180,validation_iids=45,validation_runs=90,new_tests=27,reused_test_controls=54,live_processes_included=False,new_training_during_packaging=False,new_searches_during_packaging=False,critic_checkpoint_saved=False,optimizer_checkpoint_saved=False))
dump(V/'results/REPORT_PROVENANCE.json',TRANS);dump(V/'evidence/COPY_PROVENANCE.json',COPIES)
print(json.dumps(dict(version=str(V),byte_copies=len(COPIES),transformed=len(TRANS),training_rows=sum(x['sampled_rows'] for x in train),size_MB=sum(p.stat().st_size for p in V.rglob('*') if p.is_file())/1e6)),flush=True)

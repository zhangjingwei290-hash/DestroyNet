"""Verified r2 archive and isolated r3 inputs, no torch or experiment launch."""
from pathlib import Path
import shutil,json,time,hashlib,os
ROOT=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 import psutil
 psutil.Process().cpu_affinity([14,15]);assert json.loads((ROOT/'stage2_analysis/AUDIT.json').read_text())['passed']
 assert json.loads((ROOT/'stage2_coordinator/COMPLETE.json').read_text())['complete']
 archive=ROOT/'stage2_archive_r2';archive.mkdir(exist_ok=False)
 names=['stage2_training','stage2_evaluation','stage2_analysis','stage2_coordinator','stage2_repo','STAGE2_REPORT.md','STAGE2_CHANGES.md','STAGE2_PLAN.json','STAGE2_PROTOCOL.md','STAGE2_FINISHING_NOTES.md','PREFLIGHT_STAGE2.json','stage2_summary.json','STAGE2_R2_STARTED.json','STAGE2_R2_PREPARATION.json','STAGE2_MATH_UNITS.json']
 names+= [p.name for p in ROOT.glob('*stage2*.py')]
 names+= ['stage2_common.py','stage2_cpu.py','stage2_session.py','stage2_structured.py','stage2_state_pool.py','stage2_runtime.py','stage2_initialize.py']
 receipts=[]
 for name in dict.fromkeys(names):
  source=ROOT/name;target=archive/name
  if source.is_dir():shutil.copytree(source,target,copy_function=shutil.copy2)
  else:shutil.copy2(source,target)
  files=list(source.rglob('*')) if source.is_dir() else [source]
  for p in files:
   if not p.is_file():continue
   rel=p.relative_to(ROOT);dest=archive/rel;digest=sha(p);assert sha(dest)==digest
   receipts.append(dict(path=rel.as_posix(),sha256=digest,size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,archive_mtime_ns=dest.stat().st_mtime_ns))
  print('archived',name,flush=True)
 write(archive/'ARCHIVE_RECEIPT.json',dict(passed=True,source_root=str(ROOT),files=receipts,created_unix=time.time()))
 r3=ROOT/'stage2_r3_full_convergence_45cell';r3.mkdir(exist_ok=False)
 for name in ['stage2_repo','pretrained','fit_inputs','common.py','stage2_common.py','stage2_cpu.py','stage2_session.py','stage2_structured.py','stage2_state_pool.py','stage2_runtime.py','stage2_initialize.py','preflight_stage2.py','stage2_math_units.py','stage2_gate.py','stage2_eval.py','stage2_train.py','stage2_report.py','verify_stage2.py','stats.py','FIT_MANIFEST.json','STAGE2_PLAN.json','eval_config.json','DECISIONS_PENDING.json','coordinator_stage2.py','launch_stage2.py','watch_step5.py']:
  source=ROOT/name;target=r3/name
  if source.is_dir():shutil.copytree(source,target,copy_function=shutil.copy2)
  else:shutil.copy2(source,target)
 shutil.copy2(ROOT/'provenance/USER_STAGE2_R2_STRUCTURED_REQUEST.txt',r3/'USER_R2_MECHANISMS.txt')
 request=Path('C:/Users/awe/.codex/attachments/2331d7e3-72d3-4411-9f20-9f26ca223de4/已粘贴的文本.txt');shutil.copy2(request,r3/'USER_R3_REQUEST.txt')
 baseline=ROOT.parent/'v5_desktop_alnstuned45_20seed_20261028';old45=ROOT.parent/'v5_step4_k4_adapted45_20seed_20261030'
 plan=json.loads((ROOT/'plan.json').read_text());full=json.loads((old45/'plan.json').read_text());plan.update(test_ids=full['test_ids'],core_ids=plan['test_ids'],search_seeds=list(range(20261024,20261044)),method='step5_stage2_r3',new_runs=900,stage2_r2_root=str(archive),stage1_root=str(ROOT),stage4_root=str(old45),baseline_root=str(baseline),revision='stage2_r3_full_convergence_45cell',stage2_training_hard_limit_seconds=10800)
 write(r3/'plan.json',plan)
 (r3/'inputs').mkdir();(r3/'inputs_core').mkdir()
 for iid in full['test_ids']:
  src=old45/'inputs'/f'{iid}.json';shutil.copy2(src,r3/'inputs'/src.name)
  if iid in plan['core_ids']:shutil.copy2(src,r3/'inputs_core'/src.name)
 shutil.copy2(old45/'boxes.json',r3/'boxes.json')
 cfg=json.loads((ROOT/'eval_config.json').read_text());cfg['study_root']=str(r3);cfg['project_root']=str(ROOT/'physical');write(r3/'eval_config.json',cfg)
 write(r3/'PARENT_R2.json',dict(root=str(ROOT),archive=str(archive),archive_receipt_sha256=sha(archive/'ARCHIVE_RECEIPT.json'),r2_audit_sha256=sha(ROOT/'stage2_analysis/AUDIT.json'),r2_meta_sha256=sha(archive/'stage2_training/meta_retrained.pt'),r2_pool_sha256=sha(archive/'stage2_training/progressive_state_pool.pt')))
 print('prepared',r3,flush=True)
if __name__=='__main__':main()

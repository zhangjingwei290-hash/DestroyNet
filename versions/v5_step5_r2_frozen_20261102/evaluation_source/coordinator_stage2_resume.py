"""Resume training at a verified boundary, then retain the original evaluation schedule."""
from stage2_common import ROOT,read,write,sha,readonly_check
from pathlib import Path
import sys,os,time,subprocess,traceback

def main():
 import psutil
 psutil.Process().cpu_affinity([14,15]);out=ROOT/'stage2_coordinator';assert read(ROOT/'engineering/replay_guard_direction05_batch003/REPAIR.json')['ready_for_controlled_resume']
 write(out/'process.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),argv=sys.argv,revision='stage2_r3_full12pass_co_adapted',engineering_resume=True))
 manifest=read(out/'EFFECTIVE_SOURCE_MANIFEST.json')
 for name,digest in manifest.items():assert sha(ROOT/name)==digest,name
 plan=read(ROOT/'plan.json');core=plan['core_ids'];remaining=[i for i in plan['test_ids'] if i not in core];first=[20261024,20261025,20261026]
 def run(phase,args):
  write(out/'progress.json',dict(phase=phase,engineering_resume=True,updated_unix=time.time()))
  with (out/f'{phase}_stdout.log').open('ab') as stdout,(out/f'{phase}_stderr.log').open('ab') as stderr:
   child=subprocess.Popen([sys.executable,'-B','-X','utf8',str(ROOT/args[0]),*map(str,args[1:])],cwd=ROOT,stdout=stdout,stderr=stderr)
   write(out/'active_child.json',dict(pid=child.pid,created=psutil.Process(child.pid).create_time(),phase=phase,argv=args));code=child.wait()
  if code:raise RuntimeError(f'{phase} exited {code}; retain new exact mismatch dump/boundary, no automatic retry')
 try:
  run('training_resume',['stage2_resume_train.py']);run('q_gate',['stage2_gate.py'])
  run('technical_smoke',['stage2_eval.py','--smoke','--ids','150-0-0','--seeds',20261024,'--output-dir',str(ROOT/'stage2_technical_smoke')])
  run('core36_r3',['stage2_eval.py','--ids',*core,'--seeds',*first,'--output-dir',str(ROOT/'evaluation_core_r3')]);run('core36_audit',['verify_stage2_batch.py','--ids',*core,'--seeds',*first,'--gate-r3']);run('core_model_selection',['stage2_reporting.py','--gate']);selection=read(ROOT/'MODEL_SELECTION.json')
  checkpoint=ROOT/'stage2_training/checkpoints' if selection['selected']=='r3' else Path(read(ROOT/'PARENT_R2.json')['archive'])/'stage2_training/checkpoints';common=['--checkpoint-dir',str(checkpoint),'--model-source',selection['selected'],'--method',selection['method']]
  if selection['selected']=='r2':run('core36_selected_r2_audit',['verify_stage2_batch.py','--ids',*core,'--seeds',*first])
  run('remaining99',['stage2_eval.py',*common,'--ids',*remaining,'--seeds',*first]);run('remaining99_audit',['verify_stage2_batch.py','--ids',*remaining,'--seeds',*first]);run('full45_3seed_report',['stage2_reporting.py','--seeds',*first]);completed=list(first)
  for seed in range(20261027,20261044):
   run(f'seed_{seed}',['stage2_eval.py',*common,'--ids',*plan['test_ids'],'--seeds',seed]);run(f'audit_{seed}',['verify_stage2_batch.py','--ids',*plan['test_ids'],'--seeds',seed]);completed.append(seed);run(f'report_{seed}',['stage2_reporting.py','--seeds',*completed])
  run('final_audit',['verify_stage2_final.py']);assert read(ROOT/'AUDIT.json')['passed'];verification=readonly_check()
  for name,digest in manifest.items():assert sha(ROOT/name)==digest,name
  write(out/'COMPLETE.json',dict(complete=True,selected_model=selection['selected'],selected_runs=900,r3_gate_runs=36,total_new_runs=900,primary_reused_core_runs=36 if selection['selected']=='r2' else 0,readonly=verification,audit=True,engineering_resume=True,completed_unix=time.time()));write(ROOT/'COMPLETE.json',dict(complete=True,selected_runs=900,selected_model=selection['selected'],training_144_meta_1440_adaptation=True,audit_passed=True,engineering_resume=True))
 except BaseException:
  (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()

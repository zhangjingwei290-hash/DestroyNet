"""Preserve original failure/source before any engineering change."""
from stage2_common import ROOT,read,write,sha
from pathlib import Path
import time,shutil

def main():
 import psutil
 psutil.Process().cpu_affinity([14,15]);out=ROOT/'engineering'/'replay_guard_direction05_batch003';out.mkdir(exist_ok=False)
 for name in ['stage2_train.py','coordinator_stage2.py','launch_stage2.py','stage2_runtime.py','stage2_cpu.py','stage2_state_pool.py','stage2_schedule.py','stage2_blend.py','stage2_common.py','PROTOCOL.md','STAGE2_PLAN.json','stage2_coordinator/SOURCE_MANIFEST.json','stage2_coordinator/failure.txt','stage2_coordinator/training_stderr.log','stage2_training/failure.txt','stage2_training/failure_budget.json','stage2_training/failed_training_state.pt','stage2_training/progress.json','stage2_training/direction_initialization/direction_05.pt','stage2_training/trace/adapt05_00001.pkl.gz','stage2_training/trace/adapt05_00002.pkl.gz']:
  source=ROOT/name;target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 shutil.copytree(ROOT/'stage2_repo',out/'stage2_repo')
 files={p.relative_to(out).as_posix():dict(sha256=sha(p),size=p.stat().st_size) for p in out.rglob('*') if p.is_file()}
 write(out/'PRESERVATION.json',dict(preserved=True,files=files,actual_failure_budget=read(ROOT/'stage2_training/failure_budget.json'),observed_error='Structured macro/micro replay mismatch; detailed offending metrics were not serialized by the original guard',processes_alive=False,created_unix=time.time()))
 print(dict(preserved=True,files=len(files)),flush=True)
if __name__=='__main__':main()

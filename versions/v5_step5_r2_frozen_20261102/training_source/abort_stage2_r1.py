"""Identity-bound user-requested Stage2 r1 stop, preserving completed boundaries."""
from pathlib import Path
import json,time,os
import psutil
ROOT=Path(__file__).resolve().parent
def read(p):
    with p.open(encoding='utf-8-sig') as f:return json.load(f)
def main():
    psutil.Process().cpu_affinity([14,15]);out=ROOT/'stage2_aborted_r1';out.mkdir(exist_ok=False)
    info=[];owned={}
    for name in ['stage2_coordinator/process.json','stage2_coordinator/active_child.json','stage2_training/process.json']:
        p=ROOT/name
        if not p.exists():continue
        d=read(p)
        try:
            proc=psutil.Process(d['pid']);cmd=proc.cmdline()
            assert abs(proc.create_time()-d['created'])<.01 and any(ROOT.name in a for a in cmd),d
            for c in [proc]+proc.children(recursive=True):
                owned[c.pid]=c
                info.append(dict(pid=c.pid,created=c.create_time(),ppid=c.ppid(),cmd=c.cmdline()))
        except psutil.NoSuchProcess:pass
    before=read(ROOT/'stage2_training/progress.json')
    # There is no preinstalled cooperative stop channel in r1/pythonw.
    # Preserve its atomic last-completed checkpoint, then freeze the process tree before termination.
    for p in owned.values():
        try:p.suspend()
        except psutil.NoSuchProcess:pass
    for p in owned.values():
        try:p.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(list(owned.values()),timeout=3.)
    for p in alive:
        try:p.kill()
        except psutil.NoSuchProcess:pass
    psutil.wait_procs(alive,timeout=3.)
    remaining=[]
    for d in info:
        try:
            p=psutil.Process(d['pid'])
            if abs(p.create_time()-d['created'])<.01 and p.is_running():remaining.append(d)
        except psutil.NoSuchProcess:pass
    assert not remaining
    stopped=time.time();start=read(ROOT/'stage2_training/process.json')['created']
    receipt=dict(user_requested=True,revision_aborted='stage2_r1_clone_before_dom_span002',reason_user_specified=['full_permutation_tail_PPO_ratio_noise','repair_scalar_not_unified','zero_step_non_target_source_early_state_pollution','missing_whole_route_mode'],stopped_unix=stopped,elapsed_process_seconds=stopped-start,last_complete_progress=before,process_identities=info,remaining_processes=remaining,new_revision='stage2_r2_structured_modes_unified_scalar',stop_method='suspend_identity_verified_tree_then_terminate; latest_atomic_completed_boundary_retained; no cooperative channel existed',new_training_not_started=True)
    (out/'ABORT_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
if __name__=='__main__':main()

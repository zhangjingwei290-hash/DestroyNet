"""Whole-Step5 lightweight monitor: stdlib+psutil only, CPU14/15."""
from pathlib import Path
import os,time,json
import psutil
ROOT=Path(__file__).resolve().parent

def read(p):
    for attempt in range(4):
        try:
            with p.open(encoding='utf-8-sig') as f:return json.load(f)
        except (PermissionError,json.JSONDecodeError):
            if attempt==3:raise
            time.sleep(.1)

def main():
    psutil.Process().cpu_affinity([14,15]);owned={};out=dict(checked_unix=time.time(),available_ram_gb=psutil.virtual_memory().available/2**30)
    for name in ['stage2_launch.json','stage2_coordinator/process.json','stage2_coordinator/active_child.json','stage2_coordinator/progress.json','stage2_training/process.json','stage2_training/progress.json','stage2_training/complete.json','MODEL_SELECTION.json','stage2_analysis_45iid_20seed/progress_summary.json','evaluation_core_r3/process.json','evaluation_core_r3/progress.json','evaluation_core_r3/complete.json','stage2_evaluation/process.json','stage2_evaluation/progress.json','stage2_evaluation/complete.json','stage2_coordinator/COMPLETE.json','COMPLETE.json']:
        path=ROOT/name
        if not path.exists():continue
        d=read(path);out[name]=d
        if not ('pid' in d and 'created' in d):continue
        try:
            p=psutil.Process(d['pid']);cmd=p.cmdline();match=abs(p.create_time()-d['created'])<.01 and any(ROOT.name in x for x in cmd)
            out[name+'_identity']=dict(pid=p.pid,created=p.create_time(),ppid=p.ppid(),cmd=cmd,identity_matches=match)
            if match:
                for c in [p]+p.children(recursive=True):
                    owned[c.pid]=c;c.cpu_percent(None)
        except psutil.NoSuchProcess:out[name+'_identity']=dict(alive=False)
    time.sleep(1);out['resources']=[]
    for p in owned.values():
        try:out['resources'].append(dict(pid=p.pid,created=p.create_time(),ppid=p.ppid(),cmd=p.cmdline(),affinity=p.cpu_affinity(),cpu_percent=p.cpu_percent(None),rss_gb=p.memory_info().rss/2**30))
        except psutil.NoSuchProcess:pass
    logs=list((ROOT/'stage2_coordinator').glob('*stderr.log'))+[ROOT/'stage2_coordinator/failure.txt',ROOT/'stage2_training/failure.txt',ROOT/'stage2_training/failure_budget.json',ROOT/'stage2_training/resume_failure.txt',ROOT/'stage2_training/resume_failure_budget.json']
    for p in logs:
        if p.is_file():
            with p.open('rb') as f:f.seek(max(0,p.stat().st_size-5000));out[p.relative_to(ROOT).as_posix()]=f.read().decode('utf-8','replace')
    statepath=ROOT/'monitor_state_step5.json';state=read(statepath) if statepath.exists() else {}
    state['last_check']=out;temp=statepath.with_suffix('.tmp');temp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,statepath)
    print(json.dumps(out,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

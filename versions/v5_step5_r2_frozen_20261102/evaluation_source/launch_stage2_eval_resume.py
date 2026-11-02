"""Launch the inspected evaluation continuation once, without training dispatch."""
from stage2_common import ROOT,read,write
import subprocess,time,json

def main():
    import psutil
    psutil.Process().cpu_affinity([14,15])
    assert read(ROOT/'engineering/smoke_missing_diagnostics/REPAIR.json')['ready_for_evaluation_resume']
    assert read(ROOT/'stage2_training/complete.json')['all_direction_targets_met']
    assert read(ROOT/'stage2_training/Q_GATE.json')['passed']
    if (ROOT/'stage2_eval_resume_launch.json').exists():raise FileExistsError('Continuation already attempted; inspect evidence')
    for name in ['stage2_launch.json','stage2_training/process.json','stage2_coordinator/active_child.json']:
        d=read(ROOT/name)
        try:p=psutil.Process(d['pid'])
        except psutil.NoSuchProcess:continue
        if abs(p.create_time()-d['created'])<.01 and any(ROOT.name in x for x in p.cmdline()):
            raise RuntimeError('Recorded owned process still alive; refuse duplicate')
    command='"D:/anaconda3/envs/Pytorch/pythonw.exe" -B -X utf8 "'+str(ROOT/'coordinator_stage2_eval_resume.py')+'"'
    script='$r=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=\''+command.replace("'","''")+'\';CurrentDirectory=\''+str(ROOT).replace("'","''")+'\'}; $r|ConvertTo-Json -Compress'
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',script],capture_output=True,text=True,check=True)
    d=json.loads(result.stdout.strip());assert d['ReturnValue']==0,d
    p=psutil.Process(d['ProcessId'])
    record=dict(pid=p.pid,created=p.create_time(),cmd=p.cmdline(),launched_unix=time.time(),engineering_resume=True,resume_phase='technical_smoke',training_dispatched=False)
    write(ROOT/'stage2_eval_resume_launch.json',record);write(ROOT/'stage2_launch.json',record)
    print(record,flush=True)

if __name__=='__main__':main()

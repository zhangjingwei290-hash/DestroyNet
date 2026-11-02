"""WMI detached launch after Stage1 audit and current Stage2 preflight."""
from stage2_common import ROOT,read,write,require_stage1
import subprocess,json,sys,time

def main():
    import psutil
    psutil.Process().cpu_affinity([14,15]);require_stage1()
    assert read(ROOT/'PREFLIGHT_STAGE2.json')['passed'] and read(ROOT/'PREFLIGHT_STAGE2.json')['revision']=='stage2_r2_structured_modes_unified_scalar'
    if (ROOT/'stage2_coordinator').exists() or (ROOT/'stage2_training').exists():raise FileExistsError('Stage2 already attempted; inspect records, no duplicate launch')
    command='"D:/anaconda3/envs/Pytorch/pythonw.exe" -B -X utf8 "'+str(ROOT/'coordinator_stage2.py')+'"'
    script='$r=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=\''+command.replace("'","''")+'\';CurrentDirectory=\''+str(ROOT).replace("'","''")+'\'}; $r|ConvertTo-Json -Compress'
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',script],capture_output=True,text=True,check=True)
    d=json.loads(result.stdout.strip());assert d['ReturnValue']==0,d;p=psutil.Process(d['ProcessId'])
    write(ROOT/'stage2_launch.json',dict(pid=p.pid,created=p.create_time(),cmd=p.cmdline(),launched_unix=time.time()))

if __name__=='__main__':main()

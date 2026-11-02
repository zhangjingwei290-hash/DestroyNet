"""One WMI-launched inference-only coordinator with no console window."""
import os,sys,json,time,subprocess
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parent.parent
psutil.Process().cpu_affinity([14,15]);assert not (ROOT/'step4_analysis/coordinator').exists()
for p in psutil.process_iter(['pid','cmdline']):
    if p.pid==os.getpid():continue
    try:
        cmd=' '.join(p.info['cmdline'] or []).lower()
        if any(s in cmd for s in ('coordinator_step4.py','eval_step4_both.py')):raise RuntimeError('Existing Step4 process '+str(p.pid))
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
command='"'+str(Path(sys.executable).with_name('pythonw.exe'))+'" -X utf8 "'+str(ROOT/'step4_repo/coordinator_step4.py')+'"'
script="$ErrorActionPreference='Stop'; $r=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"+command.replace("'","''")+"';CurrentDirectory='"+str(ROOT/'step4_repo').replace("'","''")+"'}; $r | Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress"
p=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',script],capture_output=True,text=True,check=True)
value=json.loads(p.stdout);assert value['ReturnValue']==0,value;p=psutil.Process(value['ProcessId'])
receipt=dict(pid=p.pid,created=p.create_time(),stage=4,script='coordinator_step4.py',launcher='WMI Win32_Process.Create',launched_unix=time.time())
(ROOT/'step4_analysis/coordinator_launch.json').write_text(json.dumps(receipt),encoding='utf-8');print(json.dumps(receipt),flush=True)

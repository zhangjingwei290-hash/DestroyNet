"""One explicitly verified WMI recovery launch; refuses duplicate live source processes."""
from stage2_common import ROOT,read,write
import subprocess,time,json

def main():
 import psutil
 psutil.Process().cpu_affinity([14,15]);assert read(ROOT/'engineering/replay_guard_direction05_batch003/REPAIR.json')['ready_for_controlled_resume']
 for name in ['stage2_launch.json','stage2_training/process.json']:
  d=read(ROOT/name)
  try:p=psutil.Process(d['pid'])
  except psutil.NoSuchProcess:continue
  if abs(p.create_time()-d['created'])<.01:raise RuntimeError('Original process still alive; refuse recovery duplicate')
 if (ROOT/'stage2_resume_launch.json').exists():raise FileExistsError('Already attempted recovery; inspect before another launch')
 command='"D:/anaconda3/envs/Pytorch/pythonw.exe" -B -X utf8 "'+str(ROOT/'coordinator_stage2_resume.py')+'"'
 script='$r=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=\''+command.replace("'","''")+'\';CurrentDirectory=\''+str(ROOT).replace("'","''")+'\'}; $r|ConvertTo-Json -Compress'
 result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',script],capture_output=True,text=True,check=True);d=json.loads(result.stdout.strip());assert d['ReturnValue']==0,d;p=psutil.Process(d['ProcessId']);record=dict(pid=p.pid,created=p.create_time(),cmd=p.cmdline(),launched_unix=time.time(),engineering_resume=True)
 write(ROOT/'stage2_resume_launch.json',record);write(ROOT/'stage2_launch.json',record)
if __name__=='__main__':main()

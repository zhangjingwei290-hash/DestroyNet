"""Read-only lightweight monitor: stdlib+psutil only, cores14/15."""
from pathlib import Path
import json,time,psutil
ROOT=Path(__file__).resolve().parent.parent
psutil.Process().cpu_affinity([14,15])
def read(p):
    for attempt in range(4):
        try:
            with p.open(encoding='utf-8-sig') as f:return json.load(f)
        except (OSError,json.JSONDecodeError):
            if attempt==3:raise
            time.sleep(.1)
r=dict(checked_unix=time.time(),available_ram_gb=psutil.virtual_memory().available/2**30)
for folder in ('step4_analysis/coordinator','step4_analysis/technical_smoke','step4_evaluation','step4_analysis'):
    for name in ('process.json','progress.json','complete.json','COMPLETE.json','VERIFICATION.json','READONLY_VERIFICATION.json','failure.txt'):
        p=ROOT/folder/name
        if p.is_file():
            if name.endswith('.json'):r[folder+'/'+name]=read(p)
            else:
                with p.open() as f:r[folder+'/'+name]=f.read()[-5000:]
for p in (ROOT/'step4_analysis/coordinator').glob('*stderr.log'):
    with p.open('rb') as f:f.seek(max(0,p.stat().st_size-3000));r[p.name]=f.read().decode('utf-8',errors='replace')
for name in ('coordinator_launch.json','coordinator/active_child.json'):
    p=ROOT/'step4_analysis'/name
    if p.is_file():
        receipt=read(p)
        try:
            proc=psutil.Process(receipt['pid']);r[name]=dict(**receipt,identity_matches=abs(proc.create_time()-receipt['created'])<.01,
                cmd=proc.cmdline(),children=[dict(pid=c.pid,created=c.create_time(),affinity=c.cpu_affinity()) for c in proc.children(recursive=True)])
        except psutil.NoSuchProcess:r[name]=dict(**receipt,alive=False)
print(json.dumps(r,ensure_ascii=False),flush=True)

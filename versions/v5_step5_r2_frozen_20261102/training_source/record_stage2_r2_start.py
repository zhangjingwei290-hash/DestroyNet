"""Record verified launch without touching sealed source or old artifacts."""
from pathlib import Path
import json,time,os
import psutil
ROOT=Path(__file__).resolve().parent
psutil.Process().cpu_affinity([14,15])
def read(p):
    with (ROOT/p).open(encoding='utf-8') as f:return json.load(f)
s=read('monitor_state_step5.json');check=s['last_check'];pre=read('PREFLIGHT_STAGE2.json')
with (ROOT/'stage2_training/train.log').open(encoding='utf-8') as f:first=json.loads(next(f))
result=dict(revision='stage2_r2_structured_modes_unified_scalar',launched=read('stage2_launch.json'),verified_unix=time.time(),
    training=read('stage2_training/process.json'),workers=[p for p in check['resources'] if p['ppid']==check['stage2_training/process.json']['pid']],
    first_completed_meta_batch={k:first[k] for k in ('batch','fit_coverage','initial_macro_ratio_error','initial_micro_ratio_error','optimizer_steps','transitions')},
    preflight_passed=pre['passed'],preflight_updates_discarded=pre['preflight_updates_discarded'],hourly_automation='dpr-step5',automation_active=True)
(ROOT/'STAGE2_R2_STARTED.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
s['last_notification']=dict(unix=time.time(),kind='stage2_r2_started',message='r2预检通过且正式训练首批完成；每小时监控，正常静默')
temp=ROOT/'monitor_state_step5.json.tmp';temp.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,ROOT/'monitor_state_step5.json')
print(result['first_completed_meta_batch'],flush=True)

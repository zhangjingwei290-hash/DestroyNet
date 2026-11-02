"""Inference-only Step4. Never dispatch training or previous baselines."""
import os,sys
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from pilot_common import ROOT,read,write
import time,subprocess,traceback,psutil
OUT=ROOT/'step4_analysis/coordinator'
def status(phase,**extra):write(OUT/'progress.json',dict(phase=phase,updated_unix=time.time(),**extra))
def run(name,script,args):
    if psutil.virtual_memory().available/2**30<3.:raise RuntimeError('Below3GB RAM; no new stage dispatched')
    with (OUT/f'{name}_stdout.log').open('w',encoding='utf-8') as stdout,(OUT/f'{name}_stderr.log').open('w',encoding='utf-8') as stderr:
        p=subprocess.Popen([str(Path(sys.executable).with_name('pythonw.exe')),'-X','utf8',str(ROOT/'step4_repo'/script),*args],cwd=ROOT/'step4_repo',
            stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        write(OUT/'active_child.json',dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),script=script,name=name,args=args));status(name,pid=p.pid);code=p.wait()
    write(OUT/f'{name}_exit.json',dict(exit_code=code,finished_unix=time.time()))
    if code:raise RuntimeError(f'{name} exited {code}; inspect retained stderr')
def main():
    OUT.mkdir(exist_ok=False);psutil.Process().cpu_affinity([14,15]);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    write(OUT/'process.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),argv=sys.argv,stage=4,training_dispatched=False))
    try:
        assert read(ROOT/'step4_analysis/PREFLIGHT.json')['passed']
        run('step4_technical_smoke','eval_step4_both.py',['--smoke','--output-dir',str(ROOT/'step4_analysis/technical_smoke')])
        assert read(ROOT/'step4_analysis/technical_smoke/complete.json')['runs']==6
        run('step4_evaluation','eval_step4_both.py',[])
        run('step4_analysis','analyze_step4.py',['--through-stage','4'])
        changed=[]
        for entry in read(ROOT/'step4_analysis/READONLY_RECEIPT.json')['files']:
            p=ROOT/entry['path']
            if not p.is_file():changed.append(entry['path']);continue
            st=p.stat()
            if (st.st_size,st.st_mtime_ns)!=(entry['size'],entry['mtime_ns']):changed.append(entry['path'])
        original=read(ROOT/'ORIGINAL_READONLY_RECEIPT.json');base=Path(original['root'])
        for entry in original['files']:
            p=base/entry['path']
            if not p.is_file():changed.append('original/'+entry['path']);continue
            st=p.stat()
            if (st.st_size,st.st_mtime_ns)!=(entry['size'],entry['mtime_ns']):changed.append('original/'+entry['path'])
        write(ROOT/'step4_analysis/READONLY_VERIFICATION.json',dict(passed=not changed,changed=changed,baseline_files=len(original['files'])))
        assert not changed
        assert read(ROOT/'step4_analysis/summary.json')['raw_runs']==243
        write(ROOT/'step4_analysis/COMPLETE.json',dict(completed=True,stage=4,new_runs=54,reused_runs=189,total_runs=243,methods=9,
            instances=9,seeds=3,training_seconds=0,source_checkpoints='step3_training/meta_pilot.pt and16 direction files',
            original_results_unchanged=True,stop_after_pilot=True,completed_unix=time.time()))
        status('complete_stop_after_step4',report=str(ROOT/'step4_analysis/REPORT.md'))
    except BaseException:
        (OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');status('failed');raise
if __name__=='__main__':
    sys.stdout=(ROOT/'step4_analysis/coordinator_stdout.log').open('a',encoding='utf-8',buffering=1)
    sys.stderr=(ROOT/'step4_analysis/coordinator_stderr.log').open('a',encoding='utf-8',buffering=1)
    main()

"""Ordered Stage2 training, frozen evaluation and five-method audit; singleton."""
from stage2_common import ROOT,read,write,sha,readonly_check,require_stage1
import os,sys,time,subprocess,traceback

def main():
    import psutil
    psutil.Process().cpu_affinity([14,15]);gate=require_stage1()
    assert read(ROOT/'PREFLIGHT_STAGE2.json')['passed']
    assert read(ROOT/'PREFLIGHT_STAGE2.json')['revision']=='stage2_r2_structured_modes_unified_scalar'
    out=ROOT/'stage2_coordinator';out.mkdir(exist_ok=False)
    write(out/'process.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),argv=sys.argv,stage1_gate=gate))
    frozen={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'stage2_repo').rglob('*.py')}
    for p in ROOT.glob('*stage2*.py'):frozen[p.name]=sha(p)
    for name in ('stage2_structured.py','stage2_state_pool.py','stage2_math_units.py','stage2_common.py','stage2_cpu.py','stage2_session.py','stage2_runtime.py','stage2_initialize.py'):frozen[name]=sha(ROOT/name)
    for name in ('STAGE2_PLAN.json','STAGE2_PROTOCOL.md','STAGE2_FINISHING_NOTES.md','PREFLIGHT_STAGE2.json','STAGE2_LOCAL_READONLY_RECEIPT.json'):
        frozen[name]=sha(ROOT/name)
    write(out/'SOURCE_MANIFEST.json',frozen)
    try:
        steps=[('training',['stage2_train.py']),('q_gate',['stage2_gate.py']),
            ('technical_smoke',['stage2_eval.py','--smoke','--output-dir',str(ROOT/'stage2_technical_smoke')]),
            ('formal36',['stage2_eval.py']),('report',['stage2_report.py']),('audit',['verify_stage2.py'])]
        for phase,args in steps:
            write(out/'progress.json',dict(phase=phase,updated_unix=time.time()))
            with (out/f'{phase}_stdout.log').open('wb') as stdout,(out/f'{phase}_stderr.log').open('wb') as stderr:
                child=subprocess.Popen([sys.executable,'-B','-X','utf8',str(ROOT/args[0]),*args[1:]],cwd=ROOT,stdout=stdout,stderr=stderr)
                write(out/'active_child.json',dict(pid=child.pid,created=psutil.Process(child.pid).create_time(),phase=phase,argv=args))
                code=child.wait()
            if code:raise RuntimeError(f'{phase} exited {code}; retain trace, no automatic restart')
        assert read(ROOT/'stage2_analysis/AUDIT.json')['passed']
        for name,digest in frozen.items():assert sha(ROOT/name)==digest,name
        verification=readonly_check()
        write(out/'COMPLETE.json',dict(complete=True,total_runs=180,new_stage2_runs=36,reused_runs=144,methods=5,stage1_r2_runs=36,aborted_partial_excluded=24,readonly=verification))
        write(ROOT/'COMPLETE.json',dict(complete=True,stage1=True,stage2=True,total_runs=180,methods=5,stage1_audit=True,stage2_audit=True))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()

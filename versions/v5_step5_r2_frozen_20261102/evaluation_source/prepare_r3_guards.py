"""Seal r3 sources and protect r2/all old experiments before formal execution."""
from stage2_common import ROOT,read,write,sha,readonly_check
from pathlib import Path
import time,ast,shutil

def main():
    import psutil
    psutil.Process().cpu_affinity([14,15]);parent=Path(read(ROOT/'PARENT_R2.json')['root']);archive=Path(read(ROOT/'PARENT_R2.json')['archive'])
    assert read(parent/'stage2_analysis/AUDIT.json')['passed'] and read(ROOT/'PREFLIGHT_STAGE2.json')['passed']
    original=read(parent/'READONLY_RECEIPT.json');roots=original['roots'];files=original['files']
    roots.update(parent_r2=str(parent),archive_r2=str(archive))
    for name,base in [('parent_r2',parent),('archive_r2',archive)]:
        for p in base.rglob('*'):
            if not p.is_file():continue
            relative=p.relative_to(base)
            if name=='parent_r2' and relative.parts[0] in (ROOT.name,'stage2_archive_r2'):continue
            st=p.stat();files.append(dict(root=name,path=relative.as_posix(),size=st.st_size,mtime_ns=st.st_mtime_ns))
    write(ROOT/'READONLY_RECEIPT.json',dict(roots=roots,files=files,created_unix=time.time()));verification=readonly_check()
    for filename in ['stage2_structured.py','stage2_cpu.py','stage2_session.py','stage2_repo/solver/model.py','stage2_repo/solver/outer.py','stage2_repo/stage1_mechanisms.py','stage2_repo/solver/repair.py','stage2_repo/solver/base_environment.py','stage2_repo/solver/environment.py']:
        assert sha(ROOT/filename)==sha(archive/filename),filename
    for p in list(ROOT.glob('*.py'))+list((ROOT/'stage2_repo').rglob('*.py')):ast.parse(p.read_text(encoding='utf-8'),str(p))
    for module in ['stage2_train','stage2_eval','stage2_gate','stage2_reporting','stage2_schedule','verify_stage2_batch','verify_stage2_final','coordinator_stage2']:__import__(module)
    pre=read(ROOT/'PREFLIGHT_STAGE2.json');assert pre['revision']=='stage2_r3_full12pass_co_adapted' and pre['projection_updates_verified'] and pre['preflight_updates_discarded'] and pre['deployment_fit_smoke']
    assert read(ROOT/'REPORTING_UNITS.json')['passed']
    failed=ROOT/'PREFLIGHT_STAGE2_FAILURE.json'
    if failed.exists():
        d=ROOT/'engineering/preflight_scope_initial';d.mkdir(exist_ok=True);shutil.move(str(failed),str(d/failed.name))
    write(ROOT/'PREPARATION.json',dict(prepared=True,revision=pre['revision'],inherited_mechanisms_byte_identical=True,readonly=verification,preflight_sha256=sha(ROOT/'PREFLIGHT_STAGE2.json'),prepared_unix=time.time()))
    print(dict(prepared=True,protected_counts=verification['counts']),flush=True)
if __name__=='__main__':main()

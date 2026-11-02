"""Final read-only guards and sealed-source receipt; no model import."""
from stage2_common import ROOT,read,write,sha,readonly_check,require_stage1
import time,ast

def main():
    import psutil
    psutil.Process().cpu_affinity([14,15]);gate=require_stage1();old=readonly_check()
    for name in ('ABORTED_READONLY_RECEIPT.json','STAGE2_R1_READONLY_RECEIPT.json'):
        for e in read(ROOT/name)['files']:
            st=(ROOT/e['path']).stat();assert (st.st_size,st.st_mtime_ns)==(e['size'],e['mtime_ns'])
    for name,digest in read(ROOT/'stage1_coordinator/SOURCE_MANIFEST.json').items():assert sha(ROOT/name)==digest,name
    paths=[]
    for name in ('repo','stage1_evaluation','stage1_analysis','stage1_coordinator'):
        paths.extend(p for p in (ROOT/name).rglob('*') if p.is_file())
    paths.extend(ROOT/name for name in ('STAGE1_REPORT.md','stage1_summary.json','stage1_mechanisms.py'))
    receipt=dict(files=[dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(set(paths))],created_unix=time.time())
    write(ROOT/'STAGE2_LOCAL_READONLY_RECEIPT.json',receipt)
    source=list((ROOT/'stage2_repo').rglob('*.py'))+list(ROOT.glob('*stage2*.py'))
    for p in source:ast.parse(p.read_text(encoding='utf-8'),str(p))
    for name in ('stage2_report','stage2_eval','stage2_gate','stage2_train','verify_stage2'):
        __import__(name)  # import-only; none may run a search or load Torch at top level
    pre=read(ROOT/'PREFLIGHT_STAGE2.json');assert pre['passed'] and pre['revision']==read(ROOT/'STAGE2_PLAN.json')['revision']
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in source}
    write(ROOT/'STAGE2_R2_PREPARATION.json',dict(prepared=True,revision=pre['revision'],stage1_gate=gate,old_readonly=old,local_protected_files=len(receipt['files']),source=manifest,preflight_sha256=sha(ROOT/'PREFLIGHT_STAGE2.json'),created_unix=time.time()))
    print(dict(prepared=True,revision=pre['revision'],old_protected=sum(old['counts'].values()),local_protected=len(receipt['files']),source_files=len(manifest)),flush=True)
if __name__=='__main__':main()

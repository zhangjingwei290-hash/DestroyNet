"""Bounded Windows-spawn import check; no model loads, optimization or search."""
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import importlib

def worker():
    import psutil
    psutil.Process().cpu_affinity([14,15])
    names=['step3_diagnostics','stage2_eval','stage2_reporting','verify_stage2_batch','verify_stage2_final']
    for name in names:importlib.import_module(name)
    import diagnostics,step3_diagnostics
    assert diagnostics.BINS==(.01,.02,.05,.10,.15,.20,.30,.40,.50,.60)
    assert len(step3_diagnostics.EXACT_LABELS)==11
    assert Path(diagnostics.__file__).resolve().parent==Path(__file__).resolve().parent
    return dict(passed=True,spawn_imports=names,diagnostics_path=diagnostics.__file__,searches=0,optimizer_steps=0,weights_loaded=0)

def main():
    import psutil
    psutil.Process().cpu_affinity([14,15])
    with ProcessPoolExecutor(1,mp_context=mp.get_context('spawn')) as ex:
        result=ex.submit(worker).result(timeout=45)
    from common import ROOT,write
    write(ROOT/'engineering/smoke_missing_diagnostics/IMPORT_CHECK.json',result)
    print(result,flush=True)

if __name__=='__main__':
    mp.freeze_support();main()

"""Resume only missing searches; retain every complete pre-interruption report."""
from pilot_common import ROOT,read,write,sha,log
import eval_step2_both as original
import os,sys,time,multiprocessing as mp,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import psutil

METHODS=('step2_meta_only','step2_adapted')
ONE=original.one

def row(report,reused=False):
    return dict(iid=report['iid'],seed=report['seed'],method=report['method'],
        steps=report['q_diagnostics']['blocks']['overall']['count'],
        search_seconds=report['search_seconds'],archive=len(report['archive']),reused=reused)

def resume_one(iid,seed,method,smoke):
    path=original._OUT/f'{iid}_seed{seed}_{method}.json'
    if not path.exists():return ONE(iid,seed,method,smoke)
    report=read(original._OUT/'metric_inputs'/path.name)
    assert report['evaluated_checkpoint_sha256']==original._DIGESTS[method]
    assert report['raw_bytes']==path.stat().st_size
    return row(report,True)

def initialize(*args):
    original.initialize(*args)
    original.one=resume_one

def main():
    psutil.Process().cpu_affinity([14,15]);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    out=ROOT/'step2_evaluation';plan=read(ROOT/'plan.json')
    assert not (out/'complete.json').exists()
    digests={'step2_meta_only':[sha(ROOT/'step2_training/meta_pilot.pt')],
        'step2_adapted':[sha(ROOT/f'step2_training/checkpoints/direction_{k:02d}.pt') for k in range(16)]}
    cases=[(i,s) for s in plan['search_seeds'] for i in plan['test_ids']]
    done={};inventory=[];tasks=[];split=[]
    for index,(iid,seed) in enumerate(cases):
        found=[]
        for method in METHODS:
            path=out/f'{iid}_seed{seed}_{method}.json';compact=out/'metric_inputs'/path.name
            assert path.exists()==compact.exists(),f'Incomplete report transaction: {path}'
            if not path.exists():continue
            report=read(compact)
            assert (report['iid'],report['seed'],report['method'])==(iid,seed,method)
            assert not report['technical_smoke'] and report['evaluated_checkpoint_sha256']==digests[method]
            assert report['raw_bytes']==path.stat().st_size
            baseline=read(ROOT/'baseline_cache'/path.name.replace(method,'v5_meta_frozen'))
            assert report['initial_population']==baseline['initial_population']
            assert report['budget_seconds']==baseline['budget_seconds']
            done[(iid,seed,method)]=row(report,True);found.append(method)
            inventory.append(dict(name=path.name,size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns))
        if len(found)<2:tasks.append((iid,seed,index,False))
        if len(found)==1:split.append(dict(iid=iid,seed=seed,previously_completed_method=found[0]))
    assert len(done)==34 and sum(2-sum((i,s,m) in done for m in METHODS) for i,s,_,_ in tasks)==20
    if '--check-only' in sys.argv:
        print(dict(passed=True,preserved_runs=len(done),missing_runs=54-len(done),pending_blocks=len(tasks),cross_interruption_pairs=split),flush=True)
        return
    write(out/'resume_receipt.json',dict(user_authorized=True,started_unix=time.time(),preserved_runs=len(done),
        missing_runs=54-len(done),preserved_raw_inventory=inventory,cross_interruption_pairs=split,
        checkpoint_sha256=digests,retrained=False,same_worker_pair_exception='Two partial pairs span pre/post interruption workers'))
    write(out/'process.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),stage=2,argv=sys.argv,
        expected_runs=54,expected_blocks=27,resume=True,methods=list(METHODS)))
    def progress(last):
        blocks=sum(all((i,s,m) in done for m in METHODS) for i,s in cases)
        write(out/'progress.json',dict(completed=len(done),total=54,completed_blocks=blocks,total_blocks=27,
            updated_unix=time.time(),resume=True,last=last))
    progress([]);start=time.time()
    try:
        with ProcessPoolExecutor(3,mp_context=mp.get_context('spawn'),initializer=initialize,
                initargs=(str(ROOT/'step2_training/checkpoints'),str(ROOT/'inputs'),str(out))) as executor:
            futures=[executor.submit(original.pair,t) for t in tasks]
            for future in as_completed(futures):
                results=future.result()
                for result in results:done[(result['iid'],result['seed'],result['method'])]=result
                progress(results);log(out/'evaluation.log',dict(pair=results,completed=len(done),resumed=True))
        for entry in inventory:
            st=(out/entry['name']).stat();assert (st.st_size,st.st_mtime_ns)==(entry['size'],entry['mtime_ns'])
        assert len(done)==54
        write(out/'complete.json',dict(completed=True,runs=54,blocks=27,smoke=False,resumed=True,
            resumed_wall_seconds=time.time()-start,methods=list(METHODS),preserved_runs=34,new_resumed_runs=20,
            preserved_reports_unchanged=True,same_worker_per_pair=False,same_worker_pairs=25,
            cross_interruption_pairs=split,alternating_order=True))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':mp.freeze_support();main()

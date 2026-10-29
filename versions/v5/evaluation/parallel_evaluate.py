"""Independent search processes with dynamic paired-job queue. Explicit --run only."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse,concurrent.futures as cf,multiprocessing as mp,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STATE=None

def initialize(affinity,core_queue,startup_barrier):
    global STATE
    import psutil,torch
    assigned=core_queue.get(timeout=30) if core_queue is not None else affinity
    if assigned:psutil.Process().cpu_affinity(assigned)
    torch.set_num_interop_threads(1)
    import evaluate as e
    cfg,pool,model,Search,plan=e.setup()
    STATE=(e,cfg,pool,model,Search,plan,{k:v.clone() for k,v in model.state_dict().items()})
    startup_barrier.wait(180)

def paired_job(job):
    import torch,psutil,numpy as np
    e,cfg,pool,model,Search,plan,frozen=STATE
    seed,iid,si,ii,resume=job;out=ROOT/'results';pair={};methods=plan['methods'] if (si+ii)%2==0 else plan['methods'][::-1]
    for method in methods:
        e.write(out/f'worker_{os.getpid()}.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),iid=iid,seed=seed,method=method,phase='search',unix=time.time()))
        path=out/f'{iid}_seed{seed}_{method}.json'
        if path.exists():
            if not resume:raise RuntimeError('Existing output requires resume')
            report=e.read(path)
            if (report['iid'],report['seed'],report['method'],report['checkpoint_sha256'])!=(iid,seed,method,plan['checkpoint_sha256']):raise ValueError('Resume identity mismatch')
        else:
            e.reset(seed);s=Search(pool,iid,cfg,seed,None if method=='heuristic' else [model]*16)
            start=e.initial(s);s.run();report=s.report()
            report.update(method=method,scope=plan['scope'],initial_population=start,checkpoint_sha256=plan['checkpoint_sha256'],execution='parallel_independent_search',worker_pid=os.getpid())
            vals=(np.array(start['objectives'])-s.base.offset)/s.base.scales
            report['initial_hv']=float(e.moocore.hypervolume(vals,ref=s.base.box['reference']))
            report['objective_minima']=np.array([p['objectives'] for p in report['archive']]).min(0).tolist()
            if any(not torch.equal(v,model.state_dict()[k]) for k,v in frozen.items()):raise RuntimeError('Frozen weights changed')
            e.write(path,report)
        pair[method]=report
    e.assert_pair(pair['heuristic'],pair['v5_meta_frozen'])
    e.write(out/f'worker_{os.getpid()}.json',dict(pid=os.getpid(),iid=iid,seed=seed,phase='pair_complete',unix=time.time()))
    return iid,seed

def main(workers,resume,affinity,pin_cores):
    import evaluate as e,psutil,msvcrt
    out=ROOT/'results'
    # Cross-entry duplicate guard, without touching unrelated jobs.
    for p in psutil.process_iter(['pid','cmdline']):
        try:
            cmd=p.info['cmdline'] or []
            if p.pid!=os.getpid() and '--run' in cmd and any(Path(x).name in ('evaluate.py','parallel_evaluate.py') and Path(x).parent==ROOT for x in cmd):raise RuntimeError('Evaluation already active')
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    lock=(ROOT/'evaluation.lock').open('a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if out.exists() and not resume:raise RuntimeError('Results exist; explicit --resume required')
    out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Campaign already complete')
    if resume and (out/'failure.txt').exists():(out/'failure.txt').rename(out/('failure_'+str(time.time_ns())+'.txt'))
    plan=e.read(ROOT/'plan.json');cfg=e.read(ROOT/'eval_config.json')
    buckets={n:[] for n in (50,150,250)}
    for si,seed in enumerate(plan['search_seeds']):
        for ii,iid in enumerate(plan['ids']):buckets[int(iid.split('-')[0])].append((seed,iid,si,ii,resume))
    if any(len(v)!=300 for v in buckets.values()):raise ValueError('Require 300 paired jobs per N')
    # Initial task wave balanced over scales, followed by dynamic queue reuse.
    jobs=[buckets[n][k] for k in range(300) for n in (50,150,250)]
    if len({(j[0],j[1]) for j in jobs})!=900:raise ValueError('Duplicate or missing paired jobs')
    e.write(out/'process.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),command=sys.argv,start_unix=time.time(),workers=workers,scope=plan['scope'],cpu_threads_per_search=1,affinity=([[2*i,2*i+1] for i in range(workers)] if pin_cores else affinity or 'OS scheduled'),nominal_search_seconds=plan['nominal_search_seconds']))
    ctx=mp.get_context('spawn');core_queue=None
    if pin_cores:
        if workers>8 or psutil.cpu_count()!=16 or psutil.cpu_count(logical=False)!=8:raise ValueError('Pinned topology requires this verified 8-core/16-thread host and at most 8 workers')
        core_queue=ctx.Queue()
        for i in range(workers):core_queue.put([2*i,2*i+1])
    startup_barrier=ctx.Barrier(workers)
    executor=cf.ProcessPoolExecutor(workers,mp_context=ctx,initializer=initialize,initargs=(affinity,core_queue,startup_barrier))
    completed=0;pending={}
    try:
        for _ in range(workers):
            job=jobs.pop(0);pending[executor.submit(paired_job,job)]=job
        while pending:
            done,_=cf.wait(pending,timeout=30,return_when=cf.FIRST_COMPLETED)
            for future in done:
                future.result();pending.pop(future);completed+=1
                if jobs:
                    job=jobs.pop(0);pending[executor.submit(paired_job,job)]=job
            e.write(out/'progress.json',dict(phase='parallel_evaluation',completed_pairs=completed,total_pairs=900,active_pairs=len(pending),queued_pairs=len(jobs),workers=workers,updated_unix=time.time(),available_ram_gb=psutil.virtual_memory().available/2**30))
            if psutil.virtual_memory().available<2*2**30:raise RuntimeError('RAM guard: available memory below 2 GiB; stop experiment workers')
        executor.shutdown(wait=True)
        from summarize import main as summarize
        summarize();e.write(out/'complete.json',dict(completed=True,runs=1800,pairs=900,frozen_weights_verified=True,workers=workers,completed_unix=time.time()))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        for f in pending:f.cancel()
        # Do not allow unmonitored child searches to continue after parent failure.
        for p in list(getattr(executor,'_processes',{}).values()):
            if p.is_alive():p.terminate()
        executor.shutdown(wait=True,cancel_futures=True)
        raise
    finally:
        lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--workers',type=int,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--affinity',type=int,nargs='+');p.add_argument('--pin-cores',action='store_true');a=p.parse_args()
    if not 1<=a.workers<=16:p.error('workers must be 1..16')
    if a.run:main(a.workers,a.resume,a.affinity,a.pin_cores)
    else:print('Prepared only: explicit --run required; no evaluation started',flush=True)

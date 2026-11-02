"""Seven CPU repair processes; eight independent search coordinators; one GPU.

A lane acquires an active slot before starting its local trajectory. Unstarted
scheduling wait is excluded, like preserved searches between training batches.
Active GPU/IPC/repair waits count in LocalSession and its parent search budget.
"""
import os,time,threading,queue,multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,Future
import numpy as np
import torch
from .parallel_context import use_client
_WORKER_POOL=None

def _worker_init(cfg_dict):
    global _WORKER_POOL
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from .config import Config
    from .data import TrainPool
    _WORKER_POOL=TrainPool(Config(**cfg_dict))
    import psutil
    allowed=psutil.Process().cpu_affinity();index=(mp.current_process()._identity[0]-1)%7
    targets=allowed[2::2]
    if targets:psutil.Process().cpu_affinity([targets[index%len(targets)]])

def _warm_worker():return os.getpid()

def _repair_worker(iid,routes,removed,name,pref,scale,seed,deadline):
    if iid not in _WORKER_POOL.split['fit_ids']:raise ValueError('Repair training outside FIT')
    instance,_,_,_=_WORKER_POOL.load(iid)
    from .repair import repair
    return repair(instance,routes,removed,name,pref,scale,seed,deadline)

class ParallelRuntime:
    critics=[]
    def __init__(self,cfg):
        self.cfg=cfg;self.requests=queue.Queue();self.closed=False;self.actor_lock=threading.Lock();self.inflight=[]
        self.repairs=ProcessPoolExecutor(cfg.cpu_workers,mp_context=mp.get_context('spawn'),
            initializer=_worker_init,initargs=(cfg.to_dict(),))
        self.active=threading.Semaphore(cfg.cpu_workers)
        self.trajectories=ThreadPoolExecutor(max_workers=cfg.independent_iid_batch*cfg.replicas)
        self.coordinators=ThreadPoolExecutor(max_workers=cfg.independent_iid_batch)
        self.calls=0;self.gpu_seconds=0.;self.batch_hist={};self.inactive_wait=0.
        began=time.perf_counter();self.worker_pids=sorted({f.result() for f in [self.repairs.submit(_warm_worker) for _ in range(cfg.cpu_workers*3)]})
        self.worker_pids=sorted(self.repairs._processes)
        self.startup_seconds=time.perf_counter()-began

    def bound_repair(self,iid):
        def dispatch(instance,routes,removed,name,pref,scale,seed,deadline):
            if instance.instance_id!=iid:raise ValueError('Repair dispatch IID mismatch')
            return self.repairs.submit(_repair_worker,iid,routes,removed,name,pref,scale,seed,deadline).result()
        return dispatch

    def act(self,observations,task_ids=None):
        future=Future()
        with self.actor_lock:
            if self.closed:raise RuntimeError('Actor service closed')
            self.requests.put((observations,task_ids or [0]*len(observations),future))
        return future.result()

    def _session(self,session,policy,task=0,episode=None,setup_pause=0.):
        from .outer import run_session
        began=time.perf_counter();self.active.acquire();wait=time.perf_counter()-began
        try:
            if session.solver_task_deadline is not None:session.solver_task_deadline+=wait+setup_pause
            run_session(session,policy,task,episode)
            end=time.perf_counter()
            for record in session.records:record['inactive_pause_before_candidate']=wait+setup_pause
            return wait,end
        finally:self.active.release()

    def run_primary(self,session,policy):
        wait,end=self._session(session,policy)
        self.inactive_wait+=wait
        return wait,end

    def run_group(self,sessions,policy,task,setup_pause):
        episodes=[[] for _ in sessions]
        futures=[self.trajectories.submit(self._session,s,policy,task,ep,setup_pause if j==0 else 0.)
                 for j,(s,ep) in enumerate(zip(sessions,episodes))]
        results=[f.result() for f in futures];ended=time.perf_counter()
        paused=setup_pause+results[0][0]+max(0.,ended-results[0][1])
        self.inactive_wait+=sum(x[0] for x in results)
        return episodes,paused

    def _collect_one(self,pool,split,cfg,weights,iid,state,rng_state,deadline):
        from .training import collect_closed
        rng=np.random.default_rng();rng.bit_generator.state=rng_state
        state['active_iid']=iid
        began=time.perf_counter()
        with use_client(self):
            episodes,complete,provenance=collect_closed(pool,split,cfg,self,None,weights,state,rng,deadline,runtime=self)
        return dict(iid=iid,episodes=episodes,complete=complete,provenance=provenance,
            state=state,rng=rng.bit_generator.state,duration=time.perf_counter()-began,
            continuation=state.get('last_continuation'))

    def collect(self,policy,pool,split,weights,ids,states,rngs,deadline,notify=lambda _:None):
        if any(i not in split['fit_ids'] for i in ids) or len(ids)>self.cfg.independent_iid_batch:raise ValueError('Invalid IID batch')
        futures=[self.coordinators.submit(self._collect_one,pool,split,self.cfg,weights,i,states[i],rngs[i],deadline) for i in ids]
        pending=[];last_notice=0.;began=time.perf_counter()
        try:
            while not all(f.done() for f in futures) or not self.requests.empty() or pending:
                try:pending.append(self.requests.get(timeout=.002))
                except queue.Empty:pass
                while True:
                    try:pending.append(self.requests.get_nowait())
                    except queue.Empty:break
                while pending:
                    key=len(pending[0][0][0]['x'])
                    candidates=[j for j,r in enumerate(pending) if len(r[0][0]['x'])==key]
                    take=min(8,len(candidates));take=2**(take.bit_length()-1)
                    indices=set(candidates[:take]);selected=[r for j,r in enumerate(pending) if j in indices]
                    pending=[r for j,r in enumerate(pending) if j not in indices]
                    obs=[r[0][0] for r in selected];tasks=[r[1][0] for r in selected]
                    self.inflight=selected
                    sampled=time.perf_counter();actions,lp,value,hidden=policy.act(obs,tasks)
                    self.gpu_seconds+=time.perf_counter()-sampled;self.calls+=take
                    self.batch_hist[str(take)]=self.batch_hist.get(str(take),0)+1
                    for j,request in enumerate(selected):request[2].set_result(([actions[j]],[lp[j]],[value[j]],hidden[j:j+1]))
                    self.inflight=[]
                if time.perf_counter()-last_notice>=10:
                    notify(dict(phase='collecting',active_iids=ids,finished_iids=sum(f.done() for f in futures),
                        collection_wall=time.perf_counter()-began,actor_calls=self.calls))
                    last_notice=time.perf_counter()
            return [f.result() for f in futures]
        except BaseException as exc:
            with self.actor_lock:self.closed=True
            for r in pending+self.inflight:
                if not r[2].done():r[2].set_exception(exc)
            while not self.requests.empty():
                r=self.requests.get_nowait()
                if not r[2].done():r[2].set_exception(exc)
            raise

    def close(self):
        self.closed=True
        self.coordinators.shutdown(wait=True,cancel_futures=True)
        self.trajectories.shutdown(wait=True,cancel_futures=True)
        self.repairs.shutdown(wait=True,cancel_futures=True)

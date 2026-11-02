"""One GPU actor service, four independent CPU repair lanes, no neural CPU training."""
from stage2_common import ROOT,read,write,FitPool,load_cfg
import os,time,queue,copy,threading,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,Future
import numpy as np
_POOL=None

def worker_init():
    global _POOL
    import psutil
    index=(mp.current_process()._identity[0]-1)%4
    psutil.Process().cpu_affinity([[2],[4],[6],[8]][index])
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    _POOL=FitPool(load_cfg())
def warm_worker():return os.getpid()
def repair_worker(iid,routes,removed,name,preference,scale,seed,deadline):
    from solver.repair import repair
    instance,*_= _POOL.load(iid)
    return repair(instance,routes,removed,name,preference,scale,seed,deadline)

class Proxy:
    critics=[]
    def __init__(self,runtime,policy):
        from stage2_cpu import CpuHead
        self.runtime=runtime;self.cpu_head=CpuHead(policy)
    def refresh(self,obs,task=0,retain_nodes=True):
        future=Future();self.runtime.requests.put((obs,task,retain_nodes,future))
        return future.result()

class Runtime:
    def __init__(self):
        self.requests=queue.Queue();self.pool=ProcessPoolExecutor(4,mp_context=mp.get_context('spawn'),initializer=worker_init)
        self.threads=ThreadPoolExecutor(4);self.source=ThreadPoolExecutor(1)
        started=time.perf_counter()
        [f.result() for f in [self.pool.submit(warm_worker) for _ in range(12)]]
        self.worker_pids=sorted(self.pool._processes);self.startup_seconds=time.perf_counter()-started
        self.gpu_seconds=0.;self.refresh_calls=0
    def bound_repair(self,iid):
        def dispatch(instance,routes,removed,name,preference,scale,seed,deadline):
            return self.pool.submit(repair_worker,iid,routes,removed,name,preference,scale,seed,deadline).result()
        return dispatch
    def run_group(self,sessions,proxy,task):
        from stage2_session import run_session
        episodes=[[] for _ in range(4)]
        def lane(session,episode):
            run_session(session,proxy,task,episode)
            return time.perf_counter()
        futures=[self.threads.submit(lane,s,e) for s,e in zip(sessions,episodes)]
        ends=[f.result() for f in futures]
        return episodes,max(0.,time.perf_counter()-ends[0])
    def collect(self,policy,pool,searches,iid,refs,target,task,seed,rng,deadline,group_id,require_mature=False):
        from stage2_state_pool import StatePool
        proxy=Proxy(self,policy);store=searches.setdefault('progressive_pool',StatePool(pool));preference=refs[target]
        tw_positive=int(target) in (3,4,14) and len(policy.heads)==1
        def collect_source():
            from solver.parallel_context import use_client
            from stage2_session import run_session
            with use_client(self):
                ready,allowed=store.candidates(iid,target,preference,tw_positive,require_mature)
                if not require_mature:
                    chosen=iid if iid in allowed else allowed[(int(seed)%len(allowed))]
                    slot=store.get(chosen,target,preference)
                    if slot is None:
                        slot=next((v for i in allowed if (v:=store.get(i,target,preference)) is not None),None)
                    if slot is None:return [],dict(valid_complete_group=False,reason='source_horizons_exhausted')
                    ready=[slot] if not slot['needs_advance'] else []
                while not ready and time.perf_counter()+3.<deadline:
                    # Four real CPU lanes advance representative states under the CURRENT preference.
                    existing=[v for key,v in store.slots.items() if key[1]==target and key[0] in allowed and store.available(v)]
                    existing.sort(key=lambda v:(v['last_used'],v['iid']))
                    ids=[v['iid'] for v in existing]+[i for i in allowed if i not in {v['iid'] for v in existing}]
                    slots=[]
                    for i in ids:
                        v=store.get(i,target,preference)
                        if v is not None and store.available(v):slots.append(v)
                        if len(slots)==4:break
                    if not slots:break
                    sessions=[store.session(v,int(rng.integers(0,2**31-1)),deadline) for v in slots]
                    futures=[self.threads.submit(run_session,se,proxy,task,None) for se in sessions]
                    for f in futures:f.result()
                    for v,se in zip(slots,sessions):store.commit(v,se,True)
                    ready,_=store.candidates(iid,target,preference,tw_positive,require_mature)
                if not ready:return [],dict(valid_complete_group=False,reason='genuine_mature_pool_unavailable_before_cutoff')
                slot=ready[0];source_label=store.label(slot)
                if require_mature and source_label not in ('Mid','Late'):raise RuntimeError('Mature quota would be fabricated')
                sessions=[store.session(slot,int(rng.integers(0,2**31-1)),deadline) for _ in range(4)]
                episodes,_=self.run_group(sessions,proxy,task)
                for ep in episodes:
                    for t in ep:t.update(iid=slot['iid'],direction=target,start_group=group_id,source_stage=source_label)
                valid=all(ep and ep[-1]['terminal'] and not ep[-1].get('truncated') and ep[-1]['diagnostics']['ending_reason']=='local_budget' for ep in episodes)
                env=sessions[0].env
                provenance=dict(iid=slot['iid'],direction=target,task=task,weight=list(preference),group_id=group_id,
                    source_routes=sessions[0].group_start_routes,source_label=source_label,required_mature=require_mature,
                    source_initial_stagnation=episodes[0][0]['stagnation'] if episodes[0] else None,
                    source_acceptance_rate=slot['acceptance'],source_initial_vehicles=slot['initial_vehicles'],source_elapsed_before=slot['used'],source_budget=slot['budget'],tw=int(slot['iid'].split('_')[4][2:]),
                    steps=[len(ep) for ep in episodes],valid_complete_group=valid,window_seconds=2.5,
                    dynamic_ideal=env.dynamic_ideal.tolist(),dynamic_span=env.dynamic_span.tolist(),
                    source='real_progressive_elite_pool_no_zero_step_crossover')
                store.commit(slot,sessions[0],False)
                return episodes,provenance
        future=self.source.submit(collect_source)
        try:
            while not future.done() or not self.requests.empty():
                try:obs,t,retain,request=self.requests.get(timeout=.001)
                except queue.Empty:continue
                began=time.perf_counter()
                try:request.set_result(policy.refresh(obs,t,retain))
                except BaseException as exc:request.set_exception(exc);raise
                self.refresh_calls+=1;self.gpu_seconds+=time.perf_counter()-began
            return future.result()
        except BaseException as exc:
            while not self.requests.empty():
                _,_,_,request=self.requests.get_nowait()
                if not request.done():request.set_exception(exc)
            raise
    def close(self):
        self.source.shutdown(wait=True,cancel_futures=True);self.threads.shutdown(wait=True,cancel_futures=True);self.pool.shutdown(wait=True,cancel_futures=True)

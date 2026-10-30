"""Complete the user pilot within45min inclusive of retained technical work."""
from pilot_common import ROOT,read,write,sha,reset,log
import sys,os,time,copy,traceback
sys.path.insert(0,str(ROOT/'step2_repo'))
import torch,numpy as np,psutil
from run import load_config
from solver.data import TrainPool
from solver.model import Policy
from solver.config import directions
from solver.training import cyclic_tasks,update,save_checkpoint
from solver.parallel_runtime import ParallelRuntime

def main():
    process_started=time.time();out=ROOT/'step2_training';out.mkdir(exist_ok=False)
    inc=ROOT/'incidents/step2_source_tail';incident=read(inc/'INCIDENT.json')
    psutil.Process().cpu_affinity(list(range(psutil.cpu_count())));psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);reset(20261024)
    cfg=load_config(ROOT/'config.json');pool=TrainPool(cfg);rng=np.random.default_rng(cfg.seed)
    ck=torch.load(inc/'retained_training/meta_pilot.pt',map_location='cuda',weights_only=False)
    meta=Policy(cfg).cuda();meta.load_state_dict(ck['model'])
    events=[read_value for read_value in (read_line(l) for l in (inc/'retained_training/train.log').read_text(encoding='utf-8').splitlines()) if read_value.get('phase')=='meta']
    history=[{k:v for k,v in h.items() if k not in ('time','stage','elapsed')} for h in events]
    assert len(history)==4,'Resume only at the observed complete four-update merge boundary'
    previous=float(incident['previous_total_seconds']);remaining=2700.-previous-(time.time()-process_started)-8.
    if remaining<300.:raise RuntimeError('Insufficient authorized remaining time; no optimization started')
    start=time.perf_counter();deadline=start+remaining
    meta_left=max(0.,1200.-incident['meta_phase_seconds_before_recovery'])
    # Reserve enough active time to obtain one true four-replica update per
    # direction. No report may label an inherited-only direction as adapted.
    meta_end=min(start+meta_left,deadline-240.)
    state={'next_iid':1};iid_rngs={};runtime=None;source_tails=[]
    write(out/'process.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),stage=2,argv=sys.argv,config=cfg.to_dict(),resume_from=str(inc/'retained_training/meta_pilot.pt'),previous_paid_seconds=previous,total_cap_seconds=2700.,optimizer_and_rng_reinitialized_at_complete_meta_boundary=True))
    def notify(value):
        write(out/'progress.json',dict(stage=2,recovery=True,elapsed=time.perf_counter()-start,total_paid_seconds=previous+time.time()-process_started,**value))
    def collect(policy,weights,limit):
        nonlocal state
        iid=state.get('active_iid')
        if iid is not None:
            entry=state.get('search_pool',{}).get(iid)
            if entry:
                available=entry['search']['budget']-entry['search']['elapsed']
                required=10.*len(weights)+1.
                if available<required:
                    source_tails.append(dict(iid=iid,residual_seconds=available,required_seconds=required,reason='incomplete_five_task_window_cannot_fit_actual_source_budget'))
                    state['search_pool'].pop(iid,None);state.pop('active_iid',None);iid=None
        if iid is None:
            cursor=state.get('next_iid',0);iid=pool.split['fit_order'][cursor%180];state['next_iid']=cursor+1
        states={iid:state};rngs={iid:iid_rngs.get(iid,np.random.default_rng(int(rng.integers(0,2**31-1))).bit_generator.state)}
        result=runtime.collect(policy,pool,pool.split,weights,[iid],states,rngs,limit,notify)[0]
        state=result['state'];iid_rngs[iid]=result['rng']
        if not result['complete']:
            save_checkpoint(out/f'unoptimized_source_tail_{len(source_tails):03d}.pt',dict(episodes=result['episodes'],provenance=result['provenance']))
            source_tails.append(dict(iid=iid,reason='incomplete_collection_retained_not_optimized'))
            state.get('search_pool',{}).pop(iid,None);state.pop('active_iid',None)
        return result
    try:
        runtime=ParallelRuntime(cfg);psutil.Process().cpu_affinity([0,1])
        inner=Policy(cfg,tasks=5).cuda();opt=torch.optim.Adam(inner.parameters(),lr=cfg.learning_rate)
        inner_u=0;last=0.;new_batches=0;meta_began=time.perf_counter()
        while time.perf_counter()+max(65.,last*1.1)<meta_end:
            if inner_u==0:inner.start_inner(meta);weights=cyclic_tasks(rng)
            begin=time.perf_counter();inner.eval();result=collect(inner,weights,meta_end)
            if not result['complete']:
                if time.perf_counter()>=meta_end:break
                continue # A source tail is not the end of the training budget.
            inner.train();metric=update(inner,opt,result['episodes'],'ppo',cfg,meta_end)
            if metric['steps']!=cfg.ppo_epochs:break
            if metric['initial_ratio_error']>2e-4:raise RuntimeError('On-policy replay mismatch')
            inner_u+=1;new_batches+=1
            h=dict(phase='meta',batch=4+new_batches,iid=result['iid'],provenance=result['provenance'],recovery=True,**metric)
            history.append(h);log(out/'train.log',dict(elapsed=time.perf_counter()-start,**h));notify(dict(phase='meta',batch=4+new_batches))
            if inner_u==cfg.inner_updates:meta.meta_update(inner,cfg.meta_step_size);inner_u=0
            last=time.perf_counter()-begin
        if inner_u:meta.meta_update(inner,cfg.meta_step_size)
        if not new_batches:raise RuntimeError('No complete resumed meta batches')
        meta_seconds=incident['meta_phase_seconds_before_recovery']+time.perf_counter()-meta_began
        save_checkpoint(out/'meta_pilot.pt',dict(model=meta.actor_state(),config=cfg.to_dict(),history=[h for h in history if h['phase']=='meta'],meta_seconds=meta_seconds,final_partial_round_updates=inner_u,source_checkpoint_sha256=read(ROOT/'plan.json')['checkpoint_sha256'],recovery_note=incident['resume_note']))
        refs=directions();order=[15]+list(range(5,15))+list(range(5))
        parent={15:None,**{k:15 for k in range(5,15)}}
        for k in range(5):parent[k]=next(j for j in range(5,15) if refs[j,k]>0)
        actors={};counts={k:0 for k in order};initializations=[];state={'next_iid':state.get('next_iid',0)}
        (out/'checkpoints').mkdir()
        for k in order:
            if deadline-time.perf_counter()<12.:raise RuntimeError('Authorized budget cannot fit another direction')
            actor=Policy(cfg).cuda();actor.start_inner(meta if parent[k] is None else actors[parent[k]])
            actors[k]=actor;optimizer=torch.optim.Adam(actor.parameters(),lr=cfg.learning_rate)
            actor.eval();result=collect(actor,[refs[k]],deadline)
            if not result['complete']:raise RuntimeError('Incomplete direction collection')
            actor.train();metric=update(actor,optimizer,result['episodes'],'ppo',cfg,deadline)
            if metric['steps']!=cfg.ppo_epochs or metric['initial_ratio_error']>2e-4:raise RuntimeError('Direction update/density check failed')
            counts[k]=1;actor.eval();initializations.append(dict(direction=k,parent=parent[k],parent_updates=0 if parent[k] is None else counts[parent[k]]))
            save_checkpoint(out/'checkpoints'/f'direction_{k:02d}.pt',dict(version=cfg.version,direction=k,weight=refs[k].tolist(),model=actor.actor_state(),updates=1,config=cfg.to_dict(),parent=parent[k],source_checkpoint_sha256=read(ROOT/'plan.json')['checkpoint_sha256']))
            h=dict(phase='adaptation',direction=k,updates=1,iid=result['iid'],provenance=result['provenance'],**metric)
            history.append(h);log(out/'train.log',dict(elapsed=time.perf_counter()-start,**h));notify(dict(phase='direction_complete',direction=k,updates=1))
        paid=previous+time.time()-process_started
        if paid>2700.:raise RuntimeError('Total inclusive pilot ceiling exceeded')
        report=dict(stage=2,wall_seconds=time.perf_counter()-start,total_step2_optimization_and_technical_seconds=paid,budget_seconds=2700.,meta_seconds=meta_seconds,direction_updates=counts,adaptation_initializations=initializations,history=history,source_tails=source_tails,optimized_transitions=sum(h['transitions'] for h in history),fit_ids_used=sorted({h['iid'] for h in history}),all_directions_updated=True,no_validation_or_realworld_training=True,source_checkpoint_sha256=read(ROOT/'plan.json')['checkpoint_sha256'],recovery_note=incident['resume_note'],previous_incident_seconds=previous)
        write(out/'training_report.json',report);write(out/'complete.json',dict(completed=True,stage=2,direction_models=16,seconds=report['wall_seconds'],inclusive_seconds=paid,meta_seconds=meta_seconds));notify(dict(phase='complete'))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:
        if runtime is not None:runtime.close()

def read_line(line):
    import json
    return json.loads(line)
if __name__=='__main__':
    import multiprocessing as mp
    mp.freeze_support();main()

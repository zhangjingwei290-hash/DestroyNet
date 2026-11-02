"""One r3 count-complete training:144 meta batches +16x90 adaptation; no total time cutoff."""
from stage2_common import ROOT,read,write,sha,reset,log,load_cfg,FitPool,require_stage1
import os,sys,time,copy,traceback,gzip,pickle

def atomic_checkpoint(path,payload):
    import torch
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');torch.save(payload,temp);os.replace(temp,path)

def main():
    gate=require_stage1();started=time.perf_counter();deadline=float('inf')  # Completion-count training; only local repair/rollout watchdogs apply.
    import psutil,numpy as np,torch
    from pathlib import Path
    from collections import Counter
    from solver.model import Policy
    from solver.training import optimizer_for,frozen_state,assert_frozen,update
    from solver.config import directions
    from stage2_initialize import initialize
    from stage2_runtime import Runtime
    from stage2_state_pool import StatePool
    from stage2_schedule import schedules,set_lr,cosine
    from stage2_blend import initialize_direction
    pre=read(ROOT/'PREFLIGHT_STAGE2.json');assert pre['passed'] and pre['revision']=='stage2_r3_full12pass_co_adapted'
    psutil.Process().cpu_affinity([0,1]);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);torch.set_num_threads(1);torch.set_num_interop_threads(1);reset(20261024)
    out=ROOT/'stage2_training';out.mkdir(exist_ok=False)
    write(out/'process.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),argv=sys.argv,total_cap_seconds=None,completion_target=dict(meta_batches=144,direction_batches=90),started_unix=time.time(),gate=gate))
    cfg=load_cfg();pool=FitPool(cfg);rng=np.random.default_rng(20261024);schedule=schedules();write(out/'fixed_schedule.json',schedule)
    meta,transfer=initialize(Policy(cfg).cuda());optimizer_for(meta);write(out/'initialization.json',transfer)
    atomic_checkpoint(out/'initialized_meta.pt',dict(model=meta.actor_state(),config=cfg.to_dict(),training_seed=20261024))
    inner=Policy(cfg,tasks=5).cuda();inner.start_inner(meta);opt=optimizer_for(inner);frozen=frozen_state(inner)
    parent=read(ROOT/'PARENT_R2.json');warm=Path(parent['archive'])/'stage2_training/progressive_state_pool.pt';assert sha(warm)==parent['r2_pool_sha256']
    statepool=StatePool(pool);restoration=statepool.restore(torch.load(warm,map_location='cpu',weights_only=False));searches={'progressive_pool':statepool};write(out/'state_pool_restoration.json',restoration)
    runtime=None;history=[];tails=[];counts=Counter();batches=merges=inner_steps=0;actors={};direction_counts={k:0 for k in range(16)};direction_optimizer_steps={k:0 for k in range(16)};initial_direction_mu={};stage='initialization'
    def notify(**v):write(out/'progress.json',dict(elapsed_seconds=time.perf_counter()-started,remaining_seconds=None,**v))
    def trace(phase,index,episodes,provenance,metric):
        p=out/'trace'/f'{phase}_{index:05d}.pkl.gz';p.parent.mkdir(exist_ok=True);temp=p.with_suffix('.tmp')
        with gzip.open(temp,'wb',compresslevel=1) as f:pickle.dump(dict(episodes=episodes,provenance=provenance,metric=metric),f,protocol=5)
        os.replace(temp,p)
    def collect(actor,refs,target,task,row,phase):
        group,p=runtime.collect(actor,pool,searches,row['iid'],refs,target,task,int(rng.integers(2**31-1)),rng,deadline-1.,f"{phase}:{row['batch']}:{task}:{row['iid']}:{target}",require_mature=row['require_mature'])
        p['scheduled']=row
        assert p.get('iid',row['iid'])==row['iid']
        if not p.get('valid_complete_group'):
            tails.append(dict(phase=phase,scheduled=row,provenance=p))
            atomic_checkpoint(out/'unoptimized_tail.pt',dict(episodes=group,provenance=p));return [],p
        if row['require_mature']:assert p['source_label'] in ('Mid','Late')
        return group,p
    try:
        runtime=Runtime();write(out/'workers.json',dict(pids=runtime.worker_pids,affinities=[[2],[4],[6],[8]],startup_seconds=runtime.startup_seconds))
        stage='meta'
        for bi in range(144):
            if inner_steps==0:
                inner.start_inner(meta);base=rng.dirichlet(np.ones(5));weights=np.stack([np.roll(base,k) for k in range(5)])
            refs=directions();refs[:5]=weights;episodes=[];provenance=[];lr=set_lr(opt,bi,144)
            for row in schedule['meta'][bi*5:(bi+1)*5]:
                group,p=collect(inner,refs,row['task'],row['task'],row,'meta');episodes+=group;provenance.append(p)
                if not group:break
            if len(episodes)!=20:raise RuntimeError('Incomplete scheduled meta group: retain state and diagnose local fault')
            metric=update(inner,opt,episodes,cfg,deadline,frozen)
            if metric['optimizer_steps']!=2:raise RuntimeError('Incomplete optimizer batch without total cutoff')
            batches+=1;inner_steps+=1;counts.update(p['iid'] for p in provenance)
            mature=sum(p['source_label'] in ('Mid','Late') for p in provenance)
            if batches>=19:assert mature>=3
            entry=dict(phase='meta',batch=batches,epoch=bi//36+1,learning_rates=lr,mature_groups=mature,total_groups=5,source_provenance=provenance,fit_ids=[p['iid'] for p in provenance],fit_coverage=len(counts),**metric)
            history.append(entry);log(out/'train.log',elapsed_seconds=time.perf_counter()-started,**entry);trace('meta',batches,episodes,provenance,metric)
            if inner_steps==cfg.inner_updates:
                epsilon=cosine(merges,36,.25,.10);meta.meta_update(inner,epsilon);inner_steps=0;merges+=1
            atomic_checkpoint(out/'latest_meta_boundary.pt',dict(model=meta.actor_state(),inner=inner.actor_state(),optimizer=opt.state_dict(),meta_batches=batches,meta_merges=merges,inner_updates=inner_steps,weights=weights,history=history,elapsed_seconds=time.perf_counter()-started,fit_visit_counts=dict(counts),progressive_pool=statepool.snapshot(),rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
            notify(phase='meta',meta_batches=batches,fit_covered=len(counts),fit_visits=dict(Counter(counts.values())),target_meta_batches=144,required_fit=180)
        if inner_steps:meta.meta_update(inner,cosine(merges,36,.25,.10));merges+=1
        meta_seconds=time.perf_counter()-started;meta_gate=batches==144 and len(counts)==180 and set(counts.values())=={4}
        atomic_checkpoint(out/'meta_retrained.pt',dict(model=meta.actor_state(),config=cfg.to_dict(),meta_batches=batches,fit_visit_counts=dict(counts),source_sha256=transfer['source_sha256']))
        assert meta_gate,'Meta coverage incomplete: no time-based normal completion is allowed'
        stage='adaptation';adapt_start=time.perf_counter();refs=directions();order=[15]+list(range(5,15))+list(range(5));parents={15:None,**{k:15 for k in range(5,15)}}
        for k in range(5):parents[k]=next(j for j in range(5,15) if refs[j,k]>0)
        for k in range(16):
            actor=Policy(cfg).cuda();actor.start_inner(meta);optimizer_for(actor);actors[k]=actor;initial_direction_mu[k]=actor.heads[0].q_mu.weight.detach().clone()
        if meta_gate:
            for k in order:
                actor=actors[k]
                initialize_direction(actor,None if parents[k] is None else actors[parents[k]],meta)
                atomic_checkpoint(out/'direction_initialization'/f'direction_{k:02d}.pt',dict(model=actor.actor_state(),direction=k,parent=parents[k],parent_blend=.7 if parents[k] is not None else None,meta_blend=.3 if parents[k] is not None else 1.))
                initial_direction_mu[k]=actor.heads[0].q_mu.weight.detach().clone();opt=optimizer_for(actor);frozen=frozen_state(actor)
                for row in schedule['adaptation'][str(k)]:
                    lr=set_lr(opt,row['batch']-1,90);episodes,p=collect(actor,refs,k,0,row,'adaptation')
                    if not episodes:raise RuntimeError('Incomplete scheduled adaptation group: preserve state')
                    metric=update(actor,opt,episodes,cfg,deadline,frozen)
                    if metric['optimizer_steps']!=2:raise RuntimeError('Incomplete adaptation optimizer without total cutoff')
                    direction_counts[k]+=1;direction_optimizer_steps[k]+=2
                    entry=dict(phase='adaptation',direction=k,update_batches=direction_counts[k],learning_rates=lr,mature_groups=int(p['source_label'] in ('Mid','Late')),total_groups=1,source_provenance=[p],**metric)
                    history.append(entry);log(out/'train.log',elapsed_seconds=time.perf_counter()-started,**entry);trace(f'adapt{k:02d}',direction_counts[k],episodes,[p],metric)
                    if direction_counts[k]%15==0:
                        atomic_checkpoint(out/'adaptation_boundaries'/f'direction_{k:02d}_batch_{direction_counts[k]:03d}.pt',dict(model=actor.actor_state(),optimizer=opt.state_dict(),direction=k,updates=direction_counts[k],optimizer_steps=direction_optimizer_steps[k],parent=parents[k],parent_blend=.7,meta_blend=.3,progressive_pool=statepool.snapshot(),rng=rng.bit_generator.state,history=history,direction_counts=direction_counts,direction_optimizer_steps=direction_optimizer_steps,elapsed_seconds=time.perf_counter()-started,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
                    notify(phase='adaptation',direction=k,direction_updates=direction_counts,meta_batches=batches,fit_covered=len(counts))
                atomic_checkpoint(out/'checkpoints'/f'direction_{k:02d}.pt',dict(model=actor.actor_state(),direction=k,weight=refs[k].tolist(),updates=direction_counts[k],optimizer_steps=direction_optimizer_steps[k],config=cfg.to_dict(),source_sha256=transfer['source_sha256'],parent=parents[k]))
        assert all(v==90 for v in direction_counts.values()),'Schedule incomplete: this is a fault, never normal completion'
        late=[h for h in history if h['phase']=='meta' and h['batch']>=19];adapt=[h for h in history if h['phase']=='adaptation']
        late_frac=sum(h['mature_groups'] for h in late)/max(1,5*len(late));adapt_frac=sum(h['mature_groups'] for h in adapt)/max(1,len(adapt));cells={str(k):sorted({tuple(p['scheduled']['cell']) for h in adapt if h['direction']==k for p in h['source_provenance']}) for k in range(16)}
        report=dict(revision='stage2_r3_full12pass_co_adapted',training_seed=20261024,hard_limit_seconds=None,completion_criterion='144_meta_and16x90_adaptation_no_total_time_cutoff',active_training_wall_seconds=time.perf_counter()-started,finalization_reserve_seconds=None,meta_seconds=meta_seconds,adaptation_seconds=time.perf_counter()-adapt_start,meta_batches=batches,meta_merges=merges,fit_ids_optimized=sorted(counts),fit_visit_counts=dict(counts),fit_required=180,full_fit_coverage=len(counts)==180,meta_switch_gate=meta_gate,direction_updates=direction_counts,direction_optimizer_steps=direction_optimizer_steps,direction_cells=cells,all_directions_updated=all(v>0 for v in direction_counts.values()),all_direction_targets_met=all(v==90 for v in direction_counts.values()),q_mu_weight_delta_norm={str(k):float((actors[k].heads[0].q_mu.weight-initial_direction_mu[k]).detach().norm()) for k in range(16)},optimized_transitions=sum(h['transitions'] for h in history),history=history,tails=tails,encoder_frozen=True,context_and_history_trainable=True,no_validation_or_test_training=True,legacy_u_calibration=False,density='mode_categorical_selected_beta_and_exact_selected_prefix_PL_dual_ratio',source_sha256=transfer['source_sha256'],warm_meta_sha256=transfer['warm_meta_sha256'],late_meta_mature_fraction=late_frac,adaptation_mature_fraction=adapt_frac,mature_quota_passed=late_frac>=.6 and adapt_frac>=.6,state_pool_resets=statepool.resets,state_pool_advances=len(statepool.advances),gpu_refresh_seconds=runtime.gpu_seconds,cutoff_reached=False,parent_initialization='0.7_parent_plus0.3_meta')
        atomic_checkpoint(out/'progressive_state_pool.pt',statepool.snapshot());write(out/'training_report.json',report)
        write(out/'complete.json',dict(completed=True,all_directions_updated=report['all_directions_updated'],all_direction_targets_met=report['all_direction_targets_met'],meta_gate_passed=meta_gate,active_training_wall_seconds=report['active_training_wall_seconds']))
        notify(phase='complete',all_direction_targets_met=report['all_direction_targets_met'],direction_updates=direction_counts)
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');atomic_checkpoint(out/'failed_training_state.pt',dict(meta=meta.actor_state(),inner=inner.actor_state(),history=history,elapsed=time.perf_counter()-started,phase=stage,progressive_pool=statepool.snapshot()))
        write(out/'failure_budget.json',dict(elapsed_seconds=time.perf_counter()-started,hard_limit_seconds=None,completion_criterion='144_meta_and16x90_adaptation_no_total_time_cutoff',phase=stage,meta_batches=batches,direction_updates=direction_counts));raise
    finally:
        if runtime is not None:runtime.close()
if __name__=='__main__':
    import multiprocessing as mp
    mp.freeze_support();main()

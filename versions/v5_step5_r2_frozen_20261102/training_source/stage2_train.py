"""One bounded GPU training run, immediately switching at60 meta+FIT coverage."""
from stage2_common import ROOT,read,write,sha,reset,log,load_cfg,FitPool,require_stage1
import os,sys,time,copy,traceback,gzip,pickle

def atomic_checkpoint(path,payload):
    import torch
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');torch.save(payload,temp);os.replace(temp,path)

def main():
    gate=require_stage1();started=time.perf_counter();hard_deadline=started+4500.;deadline=hard_deadline-20.  # reserve final checkpoint/shutdown inside75min
    import psutil,numpy as np,torch
    from solver.model import Policy
    from solver.training import optimizer_for,frozen_state,assert_frozen,update
    from solver.config import directions
    from stage2_initialize import initialize
    from stage2_runtime import Runtime
    assert read(ROOT/'PREFLIGHT_STAGE2.json')['passed']
    assert read(ROOT/'PREFLIGHT_STAGE2.json')['revision']=='stage2_r2_structured_modes_unified_scalar'
    psutil.Process().cpu_affinity([0,1]);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);assert torch.cuda.is_available();reset(20261024)
    out=ROOT/'stage2_training';out.mkdir(exist_ok=False)
    write(out/'process.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),argv=sys.argv,total_cap_seconds=4500,started_unix=time.time(),gate=gate))
    cfg=load_cfg();pool=FitPool(cfg);rng=np.random.default_rng(20261024);order=pool.manifest['fit_order'];cursor=0
    meta,transfer=initialize(Policy(cfg).cuda());optimizer_for(meta)
    initial_meta=meta.actor_state();write(out/'initialization.json',transfer)
    atomic_checkpoint(out/'initialized_meta.pt',dict(model=initial_meta,config=cfg.to_dict(),training_seed=20261024))
    inner=Policy(cfg,tasks=5).cuda();opt=optimizer_for(inner);inner.start_inner(meta);frozen=frozen_state(inner)
    runtime=None;history=[];covered=set();searches={};batches=merges=inner_steps=0;tails=[];group_counter=0;weights=None
    direction_counts={k:0 for k in range(16)};direction_optimizer_steps={k:0 for k in range(16)};actors={};initial_direction_mu={};stage='initialization'
    def notify(**values):write(out/'progress.json',dict(elapsed_seconds=time.perf_counter()-started,remaining_seconds=max(0.,deadline-time.perf_counter()),**values))
    def pool_snapshot():
        return searches['progressive_pool'].snapshot() if 'progressive_pool' in searches else {}
    reserved=set()
    def collect(actor,refs,target,task,phase,require_mature=False):
        nonlocal cursor,group_counter
        n=(50,150,250)[cursor%3];cursor+=1;group_counter+=1
        eligible=[i for i in order if int(i.split('_')[2])==n]
        fresh=[i for i in eligible if i not in covered and i not in reserved]
        iid=(fresh[0] if phase=='meta' and not require_mature and fresh else eligible[(cursor//3)%len(eligible)])
        reserved.add(iid)
        group,provenance=runtime.collect(actor,pool,searches,iid,refs,target,task,int(rng.integers(0,2**31-1)),rng,deadline-1.,f'{phase}:{group_counter}:{iid}:{target}',require_mature=require_mature)
        valid=bool(provenance.get('valid_complete_group'))
        if not valid:
            tails.append(dict(iid=iid,phase=phase,provenance=provenance,steps=[len(ep) for ep in group]))
            if group:atomic_checkpoint(out/'last_incomplete_group.pt',dict(episodes=group,provenance=provenance))
        return group if valid else [],provenance
    def trace(phase,index,episodes,provenance,metric):
        # Frozen caches and actual selected-prefix support are retained for replay audit.
        folder=out/'trace';folder.mkdir(exist_ok=True)
        p=folder/f'{phase}_{index:05d}.pkl.gz';tmp=p.with_suffix('.tmp')
        with gzip.open(tmp,'wb',compresslevel=1) as f:pickle.dump(dict(episodes=episodes,provenance=provenance,metric=metric),f,protocol=5)
        os.replace(tmp,p)
    try:
        runtime=Runtime();write(out/'workers.json',dict(pids=runtime.worker_pids,affinities=[[2],[4],[6],[8]],startup_seconds=runtime.startup_seconds))
        stage='meta';last_seconds=0.
        while time.perf_counter()+max(4.,last_seconds)+1.<deadline:
            if batches>=60 and len(covered)==180:break
            if inner_steps==0:
                inner.start_inner(meta);base=rng.dirichlet(np.ones(5));weights=np.stack([np.roll(base,k) for k in range(5)])
            refs=directions();refs[:5]=weights;episodes=[];provenance=[];began=time.perf_counter();reserved.clear()
            for task in range(5):
                if time.perf_counter()+3.>=deadline-1.:break
                group,p=collect(inner,refs,task,task,'meta',require_mature=(batches>=30 and task in (0,2,4)));episodes.extend(group);provenance.append(p)
            if len(episodes)!=20:
                tails.append(dict(phase='meta_incomplete_five_task_batch',trajectories=len(episodes)))
                if episodes:atomic_checkpoint(out/'unoptimized_meta_tail.pt',dict(episodes=episodes,provenance=provenance))
                if time.perf_counter()+15.>=deadline:break
                continue
            metric=update(inner,opt,episodes,cfg,deadline,frozen)
            if metric['optimizer_steps']!=cfg.ppo_epochs:tails.append(dict(phase='meta',metric=metric));break
            batches+=1;inner_steps+=1;covered.update(ep[0]['iid'] for ep in episodes)
            mature=sum(p['source_label'] in ('Mid','Late') for p in provenance)
            if batches>=31:assert mature>=3
            entry=dict(phase='meta',batch=batches,mature_groups=mature,total_groups=5,source_provenance=provenance,fit_ids=sorted({ep[0]['iid'] for ep in episodes}),fit_coverage=len(covered),**metric)
            history.append(entry);log(out/'train.log',elapsed_seconds=time.perf_counter()-started,**entry)
            trace('meta',batches,episodes,provenance,metric)
            if inner_steps==cfg.inner_updates:
                meta.meta_update(inner,cfg.meta_step_size);inner_steps=0;merges+=1
            atomic_checkpoint(out/'latest_meta_boundary.pt',dict(model=meta.actor_state(),inner=inner.actor_state(),optimizer=opt.state_dict(),inner_updates=inner_steps,meta_batches=batches,fit_covered=sorted(covered),cursor=cursor,rng=rng.bit_generator.state,progressive_pool=pool_snapshot()))
            last_seconds=time.perf_counter()-began
            notify(phase='meta',meta_batches=batches,fit_covered=len(covered),target_meta_batches=60,required_fit=180)
        if inner_steps:meta.meta_update(inner,cfg.meta_step_size);merges+=1
        meta_seconds=time.perf_counter()-started
        meta_gate=batches>=60 and len(covered)==180
        atomic_checkpoint(out/'meta_retrained.pt',dict(model=meta.actor_state(),config=cfg.to_dict(),meta_batches=batches,fit_covered=sorted(covered),meta_seconds=meta_seconds,source_sha256=transfer['source_sha256']))
        notify(phase='meta_complete',meta_batches=batches,fit_covered=len(covered),meta_gate_passed=meta_gate,remaining_for_adaptation_seconds=max(0.,deadline-time.perf_counter()))
        # No extra meta batch after the conjunction passes. Every remaining second is adaptation budget.
        stage='adaptation';adapt_start=time.perf_counter();refs=directions();schedule=[15]+list(range(5,15))+list(range(5))
        parent={15:None,**{k:15 for k in range(5,15)}}
        for k in range(5):parent[k]=next(j for j in range(5,15) if refs[j,k]>0)
        for k in range(16):
            actor=Policy(cfg).cuda();actor.start_inner(meta);optimizer_for(actor);actors[k]=actor
            initial_direction_mu[k]=actor.heads[0].q_mu.weight.detach().clone()
        if meta_gate:
            for k in schedule:
                actor=actors[k];actor.start_inner(meta if parent[k] is None else actors[parent[k]])
                initial_direction_mu[k]=actor.heads[0].q_mu.weight.detach().clone()
                opt=optimizer_for(actor);frozen=frozen_state(actor)
                while direction_counts[k]<20 and time.perf_counter()+4.<deadline:
                    episodes,provenance=collect(actor,refs,k,0,'adaptation',require_mature=(direction_counts[k]%2==0))
                    if not episodes:
                        if time.perf_counter()+4.>=deadline:break
                        continue
                    metric=update(actor,opt,episodes,cfg,deadline,frozen)
                    if metric['optimizer_steps']!=cfg.ppo_epochs:tails.append(dict(phase='adaptation',direction=k,metric=metric));break
                    direction_counts[k]+=1;direction_optimizer_steps[k]+=metric['optimizer_steps']
                    entry=dict(phase='adaptation',direction=k,mature_groups=int(provenance['source_label'] in ('Mid','Late')),total_groups=1,source_provenance=[provenance],update_batches=direction_counts[k],**metric)
                    history.append(entry);log(out/'train.log',elapsed_seconds=time.perf_counter()-started,**entry)
                    trace(f'adapt{k:02d}',direction_counts[k],episodes,[provenance],metric)
                    notify(phase='adaptation',direction=k,direction_updates=direction_counts,meta_batches=batches,fit_covered=len(covered))
                atomic_checkpoint(out/'checkpoints'/f'direction_{k:02d}.pt',dict(model=actor.actor_state(),direction=k,weight=refs[k].tolist(),updates=direction_counts[k],optimizer_steps=direction_optimizer_steps[k],config=cfg.to_dict(),source_sha256=transfer['source_sha256'],parent=parent[k]))
                if time.perf_counter()+4.>=deadline:break
        # Save zero-update directions explicitly if cutoff happened; never label them trained.
        for k in range(16):
            path=out/'checkpoints'/f'direction_{k:02d}.pt'
            if not path.exists():atomic_checkpoint(path,dict(model=actors[k].actor_state(),direction=k,weight=refs[k].tolist(),updates=0,optimizer_steps=0,config=cfg.to_dict(),source_sha256=transfer['source_sha256']))
        active_seconds=time.perf_counter()-started
        change={str(k):float((actors[k].heads[0].q_mu.weight-initial_direction_mu[k]).norm()) for k in range(16)}
        report=dict(training_seed=20261024,hard_limit_seconds=4500,active_training_wall_seconds=active_seconds,finalization_reserve_seconds=20.,
            meta_seconds=meta_seconds,adaptation_seconds=time.perf_counter()-adapt_start,meta_batches=batches,meta_merges=merges,
            fit_ids_optimized=sorted(covered),fit_required=180,full_fit_coverage=len(covered)==180,meta_switch_gate=meta_gate,
            direction_updates=direction_counts,direction_optimizer_steps=direction_optimizer_steps,
            direction_update_counter_definition='complete PPO batch with2 Adam steps; target20 batches per direction',
            q_mu_weight_delta_norm=change,all_directions_updated=all(v>0 for v in direction_counts.values()),
            all_direction_targets_met=all(v>=20 for v in direction_counts.values()),
            optimized_transitions=sum(h['transitions'] for h in history),history=history,tails=tails,
            backbone_and_context_frozen=True,no_validation_or_test_training=True,legacy_u_calibration=False,
            revision='stage2_r2_structured_modes_unified_scalar',density='mode_categorical_selected_beta_and_exact_selected_prefix_PL_dual_ratio',source_sha256=transfer['source_sha256'],
            gpu_refresh_seconds=runtime.gpu_seconds,source_fit_cursor=cursor,cutoff_reached=time.perf_counter()>=deadline)
        late=[h for h in history if h['phase']=='meta' and h['batch']>=31];adapt=[h for h in history if h['phase']=='adaptation']
        report['late_meta_mature_fraction']=sum(h['mature_groups'] for h in late)/max(1,sum(h['total_groups'] for h in late))
        report['adaptation_mature_fraction']=sum(h['mature_groups'] for h in adapt)/max(1,len(adapt))
        report['mature_quota_passed']=bool(late and adapt and report['late_meta_mature_fraction']>=.5 and report['adaptation_mature_fraction']>=.5)
        report['state_pool_advances']=len(searches['progressive_pool'].advances) if 'progressive_pool' in searches else 0
        atomic_checkpoint(out/'progressive_state_pool.pt',pool_snapshot())
        write(out/'training_report.json',report)
        write(out/'complete.json',dict(completed=True,all_directions_updated=report['all_directions_updated'],all_direction_targets_met=report['all_direction_targets_met'],meta_gate_passed=meta_gate,active_training_wall_seconds=active_seconds))
        notify(phase='complete',all_directions_updated=report['all_directions_updated'],direction_updates=direction_counts)
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        atomic_checkpoint(out/'failed_training_state.pt',dict(meta=meta.actor_state(),inner=inner.actor_state(),history=history,fit_covered=sorted(covered),direction_updates=direction_counts,elapsed=time.perf_counter()-started,phase=stage,progressive_pool=pool_snapshot()))
        write(out/'failure_budget.json',dict(elapsed_seconds=time.perf_counter()-started,hard_limit_seconds=4500,phase=stage,meta_batches=batches,direction_updates=direction_counts))
        raise
    finally:
        if runtime is not None:runtime.close()

if __name__=='__main__':
    import multiprocessing as mp
    mp.freeze_support();main()

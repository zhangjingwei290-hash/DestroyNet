"""Continue from verified recovered direction05 batch002; no meta/dir15 restart."""
from stage2_common import ROOT,read,write,sha,load_cfg,FitPool,log
from stage2_train import atomic_checkpoint
import os,sys,time,copy,gzip,pickle,traceback

def main():
    began=time.perf_counter()
    import psutil,numpy as np,torch
    from collections import Counter
    from solver.model import Policy
    from solver.training import optimizer_for,frozen_state,update
    from solver.config import directions
    from stage2_runtime import Runtime
    from stage2_state_pool import StatePool
    from stage2_schedule import schedules,set_lr
    from stage2_blend import initialize_direction
    from pathlib import Path
    psutil.Process().cpu_affinity([0,1]);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out=ROOT/'stage2_training';incident=ROOT/'engineering/replay_guard_direction05_batch003';assert read(incident/'PRECISION_REPAIR_VALIDATION.json')['passed']
    failure=torch.load(out/'failed_training_state.pt',map_location='cpu',weights_only=False);prior=read(incident/'stage2_training/failure_budget.json');previous_elapsed=prior['elapsed_seconds']
    cfg=load_cfg();pool=FitPool(cfg);statepool=StatePool(pool);statepool.restore(failure['progressive_pool']);statepool.advances=copy.deepcopy(failure['progressive_pool']['advances']);statepool.resets=copy.deepcopy(failure['progressive_pool'].get('resets',[]));searches={'progressive_pool':statepool}
    boundary=torch.load(out/'adaptation_boundaries/direction_15_batch_090.pt',map_location='cpu',weights_only=False);prefix=boundary['progressive_pool']['advances'];suffix=statepool.advances[len(prefix):];assert statepool.advances[:len(prefix)]==prefix and len(suffix)==3
    schedule=schedules();assert [x['iid'] for x in suffix]==[x['iid'] for x in schedule['adaptation']['5'][:3]]
    rng=np.random.default_rng();rng.bit_generator.state=boundary['rng'];rng_draws=3*5+4*len(suffix)
    for _ in range(rng_draws):rng.integers(2**31-1)
    torch.set_rng_state(boundary['torch_rng']);torch.cuda.set_rng_state_all(boundary['cuda_rng'])
    meta=Policy(cfg).cuda();meta.load_state_dict(torch.load(out/'meta_retrained.pt',map_location='cuda',weights_only=False)['model']);optimizer_for(meta)
    actors={};initial_direction_mu={};history=failure['history'];counts=Counter();direction_counts={k:0 for k in range(16)};direction_steps={k:0 for k in range(16)}
    for h in history:
        if h['phase']=='meta':counts.update(h['fit_ids'])
        else:direction_counts[h['direction']]+=1;direction_steps[h['direction']]+=h['optimizer_steps']
    assert len([h for h in history if h['phase']=='meta'])==144 and set(counts.values())=={4} and direction_counts[15]==90 and direction_counts[5]==2
    for k in range(16):
        actor=Policy(cfg).cuda();actor.start_inner(meta);optimizer_for(actor);actors[k]=actor
        ip=out/'direction_initialization'/f'direction_{k:02d}.pt'
        initial_direction_mu[k]=torch.load(ip,map_location='cuda',weights_only=False)['model']['heads.0.q_mu.weight'].detach().clone() if ip.exists() else actor.heads[0].q_mu.weight.detach().clone()
    actors[15].load_state_dict(torch.load(out/'checkpoints/direction_15.pt',map_location='cuda',weights_only=False)['model'])
    recovered=torch.load(incident/'recovered_direction05_batch002.pt',map_location='cuda',weights_only=False);actors[5].load_state_dict(recovered['model'])
    refs=directions();order=[15]+list(range(5,15))+list(range(5));parents={15:None,**{k:15 for k in range(5,15)}}
    for k in range(5):parents[k]=next(j for j in range(5,15) if refs[j,k]>0)
    runtime=None;current_actor=actors[5];current_opt=None;current_direction=5;provenance=None
    def elapsed():return previous_elapsed+time.perf_counter()-began
    def notify(**kw):write(out/'progress.json',dict(elapsed_seconds=elapsed(),remaining_seconds=None,engineering_resume=True,**kw))
    def trace(k,index,episodes,p,metric):
        path=out/'trace'/f'adapt{k:02d}_{index:05d}.pkl.gz';assert not path.exists()
        temp=path.with_suffix('.tmp')
        with gzip.open(temp,'wb',compresslevel=1) as f:pickle.dump(dict(episodes=episodes,provenance=[p],metric=metric),f,protocol=5)
        os.replace(temp,path)
    try:
        write(out/'process.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),argv=sys.argv,total_cap_seconds=None,completion_target=dict(meta_batches=144,direction_batches=90),previous_training_seconds=previous_elapsed,resuming_direction=5,resuming_next_batch=3,started_unix=time.time()))
        write(out/'ENGINEERING_RESUME.json',dict(verified_recovery_loss_deltas=[x['loss_delta'] for x in read(incident/'RECOVERY_INSPECTION.json')['reconstructed']],preserved_meta_batches=144,preserved_direction15_batches=90,preserved_direction05_batches=2,original_unoptimized_fault_group=1,original_fault_transitions_not_retained=True,source_state_after_fault_preserved=True,restored_rng_seed_draws=rng_draws,previous_active_training_seconds=previous_elapsed,diagnostic_transitions_excluded=1420,precision_patch='nonlearned_physical_statistics_float64_to_float32',thresholds_unchanged=True))
        runtime=Runtime();write(out/'workers.json',dict(pids=runtime.worker_pids,affinities=[[2],[4],[6],[8]],startup_seconds=runtime.startup_seconds))
        for k in order:
            if direction_counts[k]==90:continue
            current_direction=k;actor=actors[k];current_actor=actor
            if k!=5:
                initialize_direction(actor,None if parents[k] is None else actors[parents[k]],meta)
                initial_direction_mu[k]=actor.heads[0].q_mu.weight.detach().clone()
                path=out/'direction_initialization'/f'direction_{k:02d}.pt';assert not path.exists();atomic_checkpoint(path,dict(model=actor.actor_state(),direction=k,parent=parents[k],parent_blend=.7,meta_blend=.3))
            opt=optimizer_for(actor);current_opt=opt
            if k==5:opt.load_state_dict(recovered['optimizer'])
            frozen=frozen_state(actor)
            for row in schedule['adaptation'][str(k)][direction_counts[k]:]:
                lr=set_lr(opt,row['batch']-1,90)
                episodes,p=runtime.collect(actor,pool,searches,row['iid'],refs,k,0,int(rng.integers(2**31-1)),rng,float('inf'),f"adaptation:{row['batch']}:0:{row['iid']}:{k}",require_mature=row['require_mature']);p['scheduled']=row;provenance=p
                assert p['valid_complete_group'] and p['iid']==row['iid']
                if row['require_mature']:assert p['source_label'] in ('Mid','Late')
                metric=update(actor,opt,episodes,cfg,float('inf'),frozen);assert metric['optimizer_steps']==2
                direction_counts[k]+=1;direction_steps[k]+=2
                entry=dict(phase='adaptation',direction=k,update_batches=direction_counts[k],learning_rates=lr,mature_groups=int(p['source_label'] in ('Mid','Late')),total_groups=1,source_provenance=[p],engineering_revision='stable_nonlearned_feature_reductions',**metric)
                history.append(entry);trace(k,direction_counts[k],episodes,p,metric);log(out/'train.log',elapsed_seconds=elapsed(),**entry)
                payload=dict(model=actor.actor_state(),optimizer=opt.state_dict(),direction=k,updates=direction_counts[k],optimizer_steps=direction_steps[k],parent=parents[k],parent_blend=.7,meta_blend=.3,progressive_pool=statepool.snapshot(),rng=rng.bit_generator.state,history=history,direction_counts=direction_counts,direction_optimizer_steps=direction_steps,elapsed_seconds=elapsed(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
                atomic_checkpoint(out/'latest_resume_boundary.pt',payload)
                if direction_counts[k]%15==0:atomic_checkpoint(out/'adaptation_boundaries'/f'direction_{k:02d}_batch_{direction_counts[k]:03d}.pt',payload)
                notify(phase='adaptation',direction=k,direction_updates=direction_counts,meta_batches=144,fit_covered=180)
            atomic_checkpoint(out/'checkpoints'/f'direction_{k:02d}.pt',dict(model=actor.actor_state(),direction=k,weight=refs[k].tolist(),updates=90,optimizer_steps=180,config=cfg.to_dict(),source_sha256=read(ROOT/'STAGE2_PLAN.json')['original_meta_sha256'],parent=parents[k],engineering_revision='stable_nonlearned_feature_reductions'))
        assert all(v==90 for v in direction_counts.values())
        meta_boundary=torch.load(out/'latest_meta_boundary.pt',map_location='cpu',weights_only=False);meta_seconds=meta_boundary['elapsed_seconds'];late=[h for h in history if h['phase']=='meta' and h['batch']>=19];adapt=[h for h in history if h['phase']=='adaptation'];lf=sum(h['mature_groups'] for h in late)/(5*len(late));af=sum(h['mature_groups'] for h in adapt)/len(adapt)
        report=dict(revision='stage2_r3_full12pass_co_adapted',training_seed=20261024,hard_limit_seconds=None,completion_criterion='144_meta_and16x90_adaptation_no_total_time_cutoff',active_training_wall_seconds=elapsed(),meta_seconds=meta_seconds,meta_seconds_definition='elapsed_at_original_last_complete_meta_boundary; exact phase-transition timestamp was not serialized',adaptation_seconds=elapsed()-meta_seconds,meta_batches=144,meta_merges=36,fit_ids_optimized=sorted(counts),fit_visit_counts=dict(counts),fit_required=180,full_fit_coverage=True,meta_switch_gate=True,direction_updates=direction_counts,direction_optimizer_steps=direction_steps,direction_cells={str(k):sorted({tuple(p['scheduled']['cell']) for h in adapt if h['direction']==k for p in h['source_provenance']}) for k in range(16)},all_directions_updated=True,all_direction_targets_met=True,q_mu_weight_delta_norm={str(k):float((actors[k].heads[0].q_mu.weight-initial_direction_mu[k]).detach().norm()) for k in range(16)},optimized_transitions=sum(h['transitions'] for h in history),history=history,tails=[],encoder_frozen=True,context_and_history_trainable=True,no_validation_or_test_training=True,legacy_u_calibration=False,density='mode_categorical_selected_beta_and_exact_selected_prefix_PL_dual_ratio',source_sha256=read(ROOT/'STAGE2_PLAN.json')['original_meta_sha256'],warm_meta_sha256=read(ROOT/'PARENT_R2.json')['r2_meta_sha256'],late_meta_mature_fraction=lf,adaptation_mature_fraction=af,mature_quota_passed=lf>=.6 and af>=.6,state_pool_resets=statepool.resets,state_pool_advances=len(statepool.advances),gpu_refresh_seconds=runtime.gpu_seconds,gpu_refresh_seconds_scope='resumed segment; original total not serialized at failure',cutoff_reached=False,parent_initialization='0.7_parent_plus0.3_meta',engineering_resume=read(out/'ENGINEERING_RESUME.json'),original_failed_group_optimized=False,engineering_repair_receipt='engineering/replay_guard_direction05_batch003/REPAIR.json')
        atomic_checkpoint(out/'progressive_state_pool.pt',statepool.snapshot());write(out/'training_report.json',report);write(out/'complete.json',dict(completed=True,all_directions_updated=True,all_direction_targets_met=True,meta_gate_passed=True,active_training_wall_seconds=report['active_training_wall_seconds'],engineering_resume=True));notify(phase='complete',direction_updates=direction_counts)
    except BaseException:
        (out/'resume_failure.txt').write_text(traceback.format_exc(),encoding='utf-8');atomic_checkpoint(out/'resume_failed_training_state.pt',dict(model=current_actor.actor_state(),optimizer=current_opt.state_dict() if current_opt else None,direction=current_direction,history=history,progressive_pool=statepool.snapshot(),rng=rng.bit_generator.state,direction_counts=direction_counts,direction_optimizer_steps=direction_steps,elapsed_seconds=elapsed(),provenance=provenance))
        write(out/'resume_failure_budget.json',dict(elapsed_seconds=elapsed(),direction=current_direction,direction_counts=direction_counts,total_cap_seconds=None));raise
    finally:
        if runtime is not None:runtime.close()
if __name__=='__main__':
    import multiprocessing as mp
    mp.freeze_support();main()

"""Bounded pilots only: stage1 tree adaptation, stage2 neutral-q meta plus adaptation."""
from pilot_common import ROOT,read,write,sha,reset,log
import argparse,sys,time,copy,traceback,os
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',type=int,choices=(1,2),required=True)
    ap.add_argument('--data-dir',default=str(ROOT/'data'))
    ap.add_argument('--checkpoint-dir',default=str(ROOT/'checkpoints'))
    ap.add_argument('--output-dir')
    ap.add_argument('--seed',type=int,default=20261024)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();stage=args.stage
    out=Path(args.output_dir or ROOT/f'step{stage}_training');out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/('repo' if stage==1 else 'step2_repo')))
    import numpy as np,torch,psutil
    from run import load_config
    from solver.data import TrainPool
    from solver.model import Policy
    from solver.config import directions
    from solver.training import update,cyclic_tasks,save_checkpoint
    from solver.parallel_runtime import ParallelRuntime
    cfg=load_config(ROOT/'config.json');cfg.study_root=str(Path(args.data_dir));cfg.split_manifest=str(Path(args.data_dir)/'split_v5.json')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('GPU required for training and adaptation')
    psutil.Process().cpu_affinity(list(range(psutil.cpu_count())));psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    reset(args.seed);rng=np.random.default_rng(args.seed);pool=TrainPool(cfg)
    source=Path(args.checkpoint_dir)/'meta_complete.pt'
    assert sha(source)==read(ROOT/'plan.json')['checkpoint_sha256']
    ck=torch.load(source,map_location='cuda',weights_only=False)
    meta=Policy(cfg).cuda()
    source_state=ck['model']
    if stage==2:
        removed={k for k in source_state if k not in meta.state_dict()}
        if removed!={'heads.0.q_regime_mu'}:raise ValueError('Unexpected checkpoint migration keys '+str(removed))
        source_state={k:v for k,v in source_state.items() if k in meta.state_dict()}
    meta.load_state_dict(source_state);del ck
    if stage==1:
        meta.encoder.requires_grad_(False)
    else:
        # Only the granularity branch is reset. Encoder, node policy and all
        # remaining checkpoint tensors are transferred without random reinitialization.
        with torch.no_grad():
            for head in meta.heads:
                head.q_mu.weight.zero_();head.q_mu.bias.fill_(-1.0);head.q_state_mu.zero_()
        write(out/'initialization.json',dict(source_sha256=sha(source),q_mu_weight='zero',q_mu_bias=-1.,q_state_mu='zero',other_tensors='copied',target_mean_q_over_n='18%-20%; measured in preflight, not enforced during learning'))
    encoder_source={k:v.detach().cpu().clone() for k,v in meta.encoder.state_dict().items()}
    write(out/'process.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),stage=stage,argv=sys.argv,device=torch.cuda.get_device_name(),config=cfg.to_dict(),source_checkpoint_sha256=sha(source)))
    runtime=None;history=[];state={'next_iid':0};iid_rngs={};wall_start=time.perf_counter()
    total_budget=120. if args.smoke else 1800.
    deadline=wall_start+total_budget
    def notify(v):
        data=dict(stage=stage,elapsed=time.perf_counter()-wall_start,**v)
        write(out/'progress.json',data)
        if v.get('phase')!='collecting':log(out/'train.log',data)
    def collect(policy,weights):
        nonlocal state
        iid=state.get('active_iid')
        if iid is None:
            # ParallelRuntime sets active_iid before collect_closed sees it;
            # advance the shared FIT cursor here, at actual source admission.
            cursor=state.get('next_iid',0)
            iid=pool.split['fit_order'][cursor%180]
            state['next_iid']=cursor+1
        states={iid:state};rngs={iid:iid_rngs.get(iid,np.random.default_rng(int(rng.integers(0,2**31-1))).bit_generator.state)}
        result=runtime.collect(policy,pool,pool.split,weights,[iid],states,rngs,deadline,notify)[0]
        state=result['state'];iid_rngs[iid]=result['rng']
        return result
    try:
        runtime=ParallelRuntime(cfg);psutil.Process().cpu_affinity([0,1])
        write(out/'workers.json',dict(pids=runtime.worker_pids,cpu_workers=cfg.cpu_workers,startup_seconds=runtime.startup_seconds))
        notify(dict(phase='startup',budget_seconds=total_budget))
        if stage==2:
            inner=Policy(cfg,tasks=5).cuda();opt=torch.optim.Adam(inner.parameters(),lr=cfg.learning_rate)
            meta_end=min(deadline,wall_start+(30. if args.smoke else 1200.));batches=0;committed=0;last_duration=0.
            while time.perf_counter()+max(15.,last_duration*1.1)<meta_end:
                inner.start_inner(meta);weights=cyclic_tasks(rng);inner.eval();begin=time.perf_counter()
                old_deadline=deadline;deadline=meta_end
                result=collect(inner,weights)
                deadline=old_deadline
                if not result['complete']:
                    save_checkpoint(out/'meta_unoptimized_tail.pt',dict(episodes=result['episodes'],provenance=result['provenance']));break
                inner.train();metric=update(inner,opt,result['episodes'],'ppo',cfg,meta_end)
                if metric['steps']!=cfg.ppo_epochs:break
                if metric['initial_ratio_error']>2e-4:raise RuntimeError('On-policy density mismatch')
                meta.meta_update(inner,cfg.meta_step_size);committed+=1;batches+=1
                record=dict(phase='meta',batch=batches,iid=result['iid'],provenance=result['provenance'],**metric)
                history.append(record);notify(record)
                save_checkpoint(out/'meta_pilot.pt',dict(model=meta.actor_state(),config=cfg.to_dict(),history=history,source_checkpoint_sha256=sha(source)))
                last_duration=time.perf_counter()-begin
            if not committed:raise RuntimeError('No complete stage2 meta update within budget')
            # No extra meta training after its soft quota. The remaining time
            # up to the same absolute30-minute ceiling belongs to adaptation.
            state={'next_iid':state.get('next_iid',0)}
        refs=directions();order=[15] if args.smoke else [15]+list(range(5,15))+list(range(5))
        parent={15:None,**{k:15 for k in range(5,15)}}
        for k in range(5):parent[k]=next(j for j in range(5,15) if refs[j,k]>0)
        actors={};counts={k:0 for k in order};initializations=[];batch_times=[]
        (out/'checkpoints').mkdir()
        for index,k in enumerate(order):
            quota=max(0.,deadline-time.perf_counter())/(len(order)-index)
            began_dir=time.perf_counter();actor=Policy(cfg).cuda();actor.start_inner(meta if parent[k] is None else actors[parent[k]])
            if stage==1:actor.encoder.requires_grad_(False)
            actors[k]=actor
            initializations.append(dict(direction=k,parent=parent[k],parent_updates=0 if parent[k] is None else counts[parent[k]],quota=quota))
            opt=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=cfg.learning_rate)
            while time.perf_counter()<deadline:
                if args.smoke and counts[k]>=1:break
                estimate=max(batch_times[-3:],default=16.)
                if deadline-time.perf_counter()<estimate+3.:break
                if counts[k]>0 and time.perf_counter()-began_dir+.5*estimate>=quota:break
                begin=time.perf_counter();actor.eval();result=collect(actor,[refs[k]])
                if not result['complete']:
                    save_checkpoint(out/f'adapt_unoptimized_tail_{k:02d}.pt',dict(episodes=result['episodes'],provenance=result['provenance']));break
                actor.train();metric=update(actor,opt,result['episodes'],'ppo',cfg,deadline)
                if metric['initial_ratio_error']>2e-4:raise RuntimeError('Adaptation actor/replay density mismatch')
                if metric['steps']!=cfg.ppo_epochs:break
                counts[k]+=1;batch_times.append(time.perf_counter()-begin)
                record=dict(phase='adaptation',direction=k,updates=counts[k],iid=result['iid'],provenance=result['provenance'],**metric)
                history.append(record);notify(record)
            if stage==1:
                for name,tensor in actor.encoder.state_dict().items():
                    if not torch.equal(tensor.detach().cpu(),encoder_source[name]):raise RuntimeError('Frozen encoder changed')
            actor.eval()
            save_checkpoint(out/'checkpoints'/f'direction_{k:02d}.pt',dict(version=cfg.version,direction=k,weight=refs[k].tolist(),model=actor.actor_state(),updates=counts[k],config=cfg.to_dict(),source_checkpoint_sha256=sha(source),parent=parent[k]))
            notify(dict(phase='direction_complete',direction=k,updates=counts[k]))
        elapsed=time.perf_counter()-wall_start
        report=dict(stage=stage,wall_seconds=elapsed,budget_seconds=total_budget,direction_updates=counts,adaptation_initializations=initializations,history=history,encoder_frozen_verified=stage==1,optimized_transitions=sum(h['transitions'] for h in history),actor_calls=runtime.calls,gpu_service_seconds=runtime.gpu_seconds,fit_ids_used=sorted({h['iid'] for h in history}),no_validation_or_realworld_training=True,all_directions_updated=all(counts.values()),source_checkpoint_sha256=sha(source))
        write(out/'training_report.json',report)
        if not all(counts.values()):raise RuntimeError('Some directions have zero optimizer updates; do not evaluate as adapted')
        write(out/'complete.json',dict(completed=True,stage=stage,seconds=elapsed,direction_models=len(order),technical_smoke=args.smoke))
        notify(dict(phase='complete',direction_updates=counts,wall_seconds=elapsed))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:
        if runtime is not None:runtime.close()

if __name__=='__main__':
    import multiprocessing as mp
    mp.freeze_support();main()

"""V5 single-seed, three full FIT epochs with7 CPU repair workers / central GPU.

No test evaluation or adaptation is launched by this entry. Latest checkpoint
contains every search/RNG plus EMNH inner position; --resume is explicit only.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import argparse,copy,json,time,traceback,shutil,sys
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import torch
from run import load_config,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--root',required=True)
    ap.add_argument('--smoke-batches',type=int,default=0);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--smoke-local-seconds',type=float,default=None)
    args=ap.parse_args();cfg=load_config(args.config);cfg.validate(require_budget=True)
    if cfg.independent_iid_batch!=8 or cfg.cpu_workers!=7 or cfg.meta_target_epochs!=3:raise ValueError('Reviewed8IID/7workers/3epochs required')
    if args.smoke_local_seconds is not None:
        if not args.smoke_batches:raise ValueError('Short segment only permitted in named smoke')
        cfg.local_seconds=args.smoke_local_seconds
    if cfg.device!='cuda' or not torch.cuda.is_available():raise RuntimeError('CUDA required')
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True);checkpoint_dir=root/'checkpoints';checkpoint_dir.mkdir(exist_ok=True)
    if (root/'process.json').exists() and not args.resume:raise FileExistsError('Run already initialized')
    from solver.data import TrainPool
    from solver.model import Policy
    from solver.training import cyclic_tasks,update,save_checkpoint,rng_state
    from solver.parallel_runtime import ParallelRuntime
    import psutil
    pool=TrainPool(cfg);split=pool.split;order=split['fit_order']
    if len(order)!=180 or set(order)!=set(split['fit_ids']):raise ValueError('Invalid180FIT order')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(cfg.seed);torch.cuda.manual_seed_all(cfg.seed);rng=np.random.default_rng(cfg.seed)
    meta=Policy(cfg).cuda();inner=Policy(cfg,tasks=5).cuda()
    optimizer=torch.optim.Adam(inner.parameters(),lr=cfg.learning_rate)
    states={i:dict(next_iid=0) for i in order}
    rngs={i:np.random.default_rng(cfg.seed+100000+j).bit_generator.state for j,i in enumerate(order)}
    counts={i:0 for i in order};active=[];next_index=0;epoch=0;round_i=0;inner_u=0;weights=None
    batches=0;history=[];spent_before=0.;unoptimized=[]
    latest=checkpoint_dir/'latest_checkpoint.pt'
    if args.resume:
        ck=torch.load(latest,map_location='cuda',weights_only=False)
        if ck['config']!=cfg.to_dict():raise ValueError('Resume config differs')
        meta.load_state_dict(ck['meta']);inner.load_state_dict(ck['inner']);optimizer.load_state_dict(ck['optimizer'])
        states,rngs,counts=ck['states'],ck['iid_rngs'],ck['counts']
        active,next_index,epoch=ck['active'],ck['next_index'],ck['epoch']
        round_i,inner_u,weights=ck['round_i'],ck['inner_u'],ck['weights']
        batches,history,spent_before=ck['batches'],ck['history'],ck['spent']
        rng.bit_generator.state=ck['rng']['numpy_generator'];np.random.set_state(ck['rng']['numpy_global'])
        torch.set_rng_state(ck['rng']['torch_cpu'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in ck['rng']['torch_cuda']])
    process=psutil.Process();started=time.perf_counter();deadline=started+max(0.,cfg.meta_seconds-spent_before)
    if args.smoke_batches:deadline=min(deadline,started+1200.)
    clock=lambda:spent_before+time.perf_counter()-started
    write_json(root/'process.json',dict(pid=os.getpid(),created=process.create_time(),command=sys.argv,
        config=cfg.to_dict(),scope='technical_smoke_discard_weights' if args.smoke_batches else 'formal_meta_only',
        training_instances=180,validation_instances_not_trained=45,realworld_test_instances_not_trained=45,
        target_full_epochs=3,target_completed_cycles=540))
    def notify(value):
        value=dict(value,wall_seconds=clock(),completed_cycles=sum(counts.values()),completed_epochs=epoch,
            optimized_iids=len({i for h in history for i in h['iids']}),batches=batches)
        write_json(root/'progress.json',value)
        stamp=datetime.now(timezone(timedelta(hours=8))).isoformat()
        # Direct file handles: no dependence on a persistent console.
        with (root/'worker.log').open('a',encoding='utf-8') as f:f.write(stamp+' '+json.dumps(value,ensure_ascii=False,default=str)+'\n')
    runtime=None
    try:
        runtime=ParallelRuntime(cfg)
        write_json(root/'workers.json',dict(pids=runtime.worker_pids,startup_seconds=runtime.startup_seconds,
            per_worker_cpu_threads=1,active_trajectory_slots=cfg.cpu_workers))
        notify(dict(phase='startup',cuda=torch.cuda.get_device_name(),gpu_memory_GiB=torch.cuda.get_device_properties(0).total_memory/2**30))
        while epoch<cfg.meta_target_epochs and time.perf_counter()<deadline:
            if inner_u==0:
                inner.start_inner(meta);weights=cyclic_tasks(rng)
            if not active:
                active=order[next_index:next_index+cfg.independent_iid_batch];next_index+=len(active)
                if not active:raise RuntimeError('FIT scheduler exhausted before epoch accounting')
            began=time.perf_counter();before={i:counts[i] for i in active}
            inner.eval()
            results=runtime.collect(inner,pool,split,weights,active,states,rngs,deadline,notify)
            for result in results:
                states[result['iid']]=result['state'];rngs[result['iid']]=result['rng']
            if not all(r['complete'] for r in results):
                unoptimized=results;break
            episodes=[ep for r in results for ep in r['episodes']]
            if len(episodes)!=len(active)*cfg.task_count*cfg.replicas:raise ValueError('Not exactly5 tasks x4 replicas per IID')
            # Verify the four source states before optimizing. Repair ID and
            # remaining measured milliseconds may differ; physical/history may not.
            starts={}
            for ep in episodes:starts.setdefault((ep[0]['iid'],ep[0]['task']),[]).append(ep[0]['observation'])
            for group in starts.values():
                if len(group)!=4:raise ValueError('Wrong same-start group')
                for obs in group[1:]:
                    if obs['routes']!=group[0]['routes']:raise ValueError('Replica source routes differ')
                    for key in ('x','edge_index_n','edge_index_r0','edge_index_r1','state_'):
                        if not torch.equal(obs[key],group[0][key]):raise ValueError('Replica source state differs')
            inner.train();opt_began=time.perf_counter();metric=update(inner,optimizer,episodes,'ppo',cfg,deadline)
            opt_seconds=time.perf_counter()-opt_began
            if metric['steps']!=cfg.ppo_epochs:
                unoptimized=results;break
            if metric['initial_ratio_error']>2e-4:raise ValueError('On-policy actor/replay density mismatch')
            for result in results:counts[result['iid']]=result['state'].get('completed_cycles',0)
            record=dict(batch=batches,round=round_i,inner_u=inner_u,iids=list(active),weights=weights.tolist(),
                collection_seconds=opt_began-began,update_seconds=opt_seconds,
                continuations=[r['continuation'] for r in results],
                provenance=[p for r in results for p in r['provenance']],
                by_n={str(n):dict(transitions=sum(len(ep) for ep in episodes if len(ep[0]['observation']['x'])==n+1),
                    timeouts=sum(t['diagnostics']['timeout'] for ep in episodes if len(ep[0]['observation']['x'])==n+1 for t in ep),
                    positive_steps=sum(t['reward']>0 for ep in episodes if len(ep[0]['observation']['x'])==n+1 for t in ep)) for n in (50,150,250)},**metric)
            history.append(record);batches+=1;inner_u+=1
            active=[i for i in active if counts[i]==before[i]]
            if not active and next_index==len(order):
                epoch+=1;next_index=0
                if any(v!=epoch for v in counts.values()):raise ValueError('Epoch counts not equal across180FIT')
            if inner_u==cfg.inner_updates or epoch==cfg.meta_target_epochs:
                meta.meta_update(inner,cfg.meta_step_size);round_i+=1;inner_u=0;weights=None
            save_checkpoint(latest,dict(config=cfg.to_dict(),meta=meta.actor_state(),inner=inner.actor_state(),
                optimizer=optimizer.state_dict(),states=states,iid_rngs=rngs,counts=counts,active=active,
                next_index=next_index,epoch=epoch,round_i=round_i,inner_u=inner_u,weights=weights,
                batches=batches,history=history,spent=clock(),rng=rng_state(rng)))
            estimate=None
            if sum(counts.values())>=8:
                # A rate extrapolation, not a convergence or completion guarantee.
                estimate=clock()/sum(counts.values())*(cfg.meta_target_epochs*len(order)-sum(counts.values()))
            notify(dict(phase='meta',round=round_i,inner_u=inner_u,transitions=metric['transitions'],
                optimizer_steps=metric['steps'],collection_seconds=record['collection_seconds'],update_seconds=opt_seconds,
                initial_ratio_error=metric['initial_ratio_error'],timeout_fraction=metric['timeout_fraction'],
                eta_seconds_rate_extrapolation=estimate))
            if args.smoke_batches and batches>=args.smoke_batches:break
        if unoptimized:torch.save(unoptimized,checkpoint_dir/'unoptimized_tail.pt')
        complete=epoch==cfg.meta_target_epochs
        if not args.smoke_batches and not complete:raise RuntimeError('Meta hard deadline reached before3 full epochs; preserved latest checkpoint/tail')
        if args.smoke_batches and batches<args.smoke_batches:raise RuntimeError('Smoke did not complete requested optimizer batches')
        if complete:
            save_checkpoint(checkpoint_dir/'meta_complete.pt',dict(config=cfg.to_dict(),model=meta.actor_state(),
                counts=counts,history=history,spent=clock(),rng=rng_state(rng)))
        report=dict(config=cfg.to_dict(),scope='smoke_only' if args.smoke_batches else 'meta_only',
            completed_epochs=epoch,completed_cycles=sum(counts.values()),cycles_by_iid=counts,
            optimized_iids=sorted({i for h in history for i in h['iids']}),fit_total=180,
            batches=batches,optimizer_steps=sum(h['steps'] for h in history),
            optimized_transitions=sum(h['transitions'] for h in history),wall_seconds=clock(),
            actor_calls=runtime.calls,gpu_service_seconds=runtime.gpu_seconds,gpu_batch_histogram=runtime.batch_hist,
            inactive_lane_wait_seconds=runtime.inactive_wait,history=history,
            no_realworld_training=True,validation_not_used_for_model_selection=True)
        write_json(root/'training_report.json',report)
        write_json(root/'complete.json',dict(scope=report['scope'],batches=batches,completed_epochs=epoch,
            completed_cycles=sum(counts.values()),wall_seconds=clock(),formal_weights=complete and not args.smoke_batches))
        notify(dict(phase='complete',scope=report['scope']))
    except BaseException:
        (root/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');notify(dict(phase='failed'));raise
    finally:
        if runtime is not None:runtime.close()

if __name__=='__main__':
    import multiprocessing as mp
    mp.freeze_support();main()

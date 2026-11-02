"""36 Stage2 frozen searches only; old baselines are immutable references."""
from stage2_common import ROOT,read,write,sha,reset,log,EvalPool,load_cfg
import os,sys,time,copy,argparse,traceback,multiprocessing as mp,math
from stage1_mechanisms import refresh_required
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
POOL=CFG=OUT=PLAN=ACTORS=FROZEN=None
def initialize(data_dir,checkpoint_dir,out,method,model_source):
    global POOL,CFG,OUT,PLAN,ACTORS,FROZEN
    import psutil,torch
    from solver.model import Policy
    from stage2_cpu import CpuHead
    PLAN=read(ROOT/'plan.json');PLAN['method']=method;PLAN['model_source']=model_source;PLAN['checkpoint_sha256']=[sha(Path(checkpoint_dir)/f'direction_{k:02d}.pt') for k in range(16)];index=(mp.current_process()._identity[0]-1)%3
    psutil.Process().cpu_affinity(PLAN['worker_affinities'][index]);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU fallback forbidden')
    begin=time.perf_counter();CFG=load_cfg();POOL=EvalPool(CFG,data_dir);OUT=Path(out);ACTORS=[];FROZEN=[]
    for k,digest in enumerate(PLAN['checkpoint_sha256']):
        path=Path(checkpoint_dir)/f'direction_{k:02d}.pt';assert sha(path)==digest
        ck=torch.load(path,map_location='cuda',weights_only=False)
        assert ck['direction']==k and ck['updates']>=1
        model=Policy(CFG).cuda();model.load_state_dict(ck['model'],strict=True);model.eval()
        assert len(model.heads)==1
        model.cpu_head=CpuHead(model,legacy_feature_reductions=(model_source=='r2'));ACTORS.append(model);FROZEN.append({key:v.detach().clone() for key,v in model.state_dict().items()})
    torch.cuda.synchronize()
    write(OUT/f'worker_{os.getpid()}.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),phase='ready',affinity=psutil.Process().cpu_affinity(),initialization_seconds=time.perf_counter()-begin,cuda=True,device=torch.cuda.get_device_name(0),dtype='float32',numeric_threads=1,checkpoint_sha256=PLAN['checkpoint_sha256']))
def frozen_equal():
    import torch
    return all(torch.equal(value,ACTORS[k].state_dict()[key]) for k,state in enumerate(FROZEN) for key,value in state.items())
def job(task):
    import psutil
    from solver.outer import Search
    from step3_diagnostics import records_q
    iid,seed,smoke=task;method=PLAN['method'];path=OUT/f'{iid}_seed{seed}_{method}.json'
    if path.exists():raise FileExistsError('Refusing to overwrite '+str(path))
    info=dict(pid=os.getpid(),created=psutil.Process().create_time(),iid=iid,seed=seed,phase='running',affinity=psutil.Process().cpu_affinity(),started_unix=time.time())
    write(OUT/f'worker_{os.getpid()}.json',info)
    # Immutable baseline IO occurs before starting the solver clock.
    old=[]
    for m in ('alns_tuned','v5_meta_frozen'):
        ref_seed=20261024 if smoke else seed
        ref=read(Path(PLAN['baseline_root'])/'metric_inputs/results'/f'{iid}_seed{ref_seed}_{m}.json')
        assert (ref['iid'],ref['seed'],ref['method'])==(iid,ref_seed,m);old.append(ref)
    assert frozen_equal();reset(seed)
    search=Search(POOL,iid,CFG,seed,policies=ACTORS,seconds=16. if smoke else None)
    initial=dict(routes=[copy.deepcopy(p.X[0].routes) for p in search.algorithm.pop],objectives=[p.F.tolist() for p in search.algorithm.pop])
    for ref in old:
        if not smoke:assert ref['initial_population']==initial and ref['budget_seconds']==search.budget
        assert ref['physical_offset']==search.base.offset.tolist() and ref['physical_scale']==search.base.scales.tolist()
    search.run();report=search.report();assert frozen_equal()
    for point in report['archive']:
        assert POOL.api.feasible(search.base.solution(point['routes']),search.base.instance)
    report['dynamic_scalarization_snapshots']=search.dynamic_scalarization_snapshots
    report['clone_rejections']=getattr(search.algorithm,'clone_rejections',0)
    records=report['records'];hits=sum(bool(x['cache_hit']) for x in records)
    pulse=[x for x in records if x['use_pulse']];refine=[x for x in records if not x['use_pulse']]
    for x in records:
        expected=(x['cache_step']==0 or (x['dirty_since_refresh_before'] and x['repair_seconds_since_refresh_before']>=5.*max(x['last_gnn_refresh_seconds_before'],1e-4)) or (not x['dirty_since_refresh_before'] and x['cache_age_before']>=8))
        assert x['cache_refresh']==expected
        assert x['cache_hit']==(not x['cache_refresh'])
        assert x['cache_age_after']==(1 if x['cache_refresh'] else x['cache_age_before']+1)
        assert not x['use_pulse']
        from stage2_structured import q_for_mode
        a=x['action'];n=int(iid.split('-')[0]);size,qc=q_for_mode(a['mode'],a['u'],n,len(x['sampling_routes']))
        assert a['q_condition']==qc and len(x['removed'])==len(set(x['removed']))==x['q']
        if a['mode']<2:assert x['q']==size
        else:
            assert len(a['selected_routes'])==size and set(x['sampling_routes'][a['selected_routes'][0]])<=set(x['removed'])
            assert x['q']<=a['q_cap'] or a['first_route_exceeds_cap']
    n=int(iid.split('-')[0])
    cache=dict(refresh_rule="repair_amortization_or_clean_age8",hits=hits,refreshes=len(records)-hits,hit_rate=hits/len(records) if records else 0.,pulse_steps=len(pulse),refinement_steps=len(refine),pulse_mean_q_ratio=sum(x['q']/n for x in pulse)/len(pulse) if pulse else None,refinement_mean_q_ratio=sum(x['q']/n for x in refine)/len(refine) if refine else None,gnn_refresh_seconds=sum(x['gnn_refresh_seconds'] for x in records),cached_sampling_seconds=sum(x['cached_sampling_seconds'] for x in records),physical_refreshes=sum(x['physical_refresh'] for x in records),physical_refresh_seconds=sum(x['physical_refresh_seconds'] for x in records),q_head_seconds=sum(x['q_head_seconds'] for x in records),logits_condition='fixed_lo_hi_interpolated_topology_plus_live_physical_residual',hidden_update='refresh_only',sampling_backend=PLAN['sampling'])
    report['state_awareness']=[dict(stagnation=st,count=sum(min(9,x['stagnation_before'])==st for x in records),q_sum=sum(x['q'] for x in records if min(9,x['stagnation_before'])==st),accepted=sum(x['accepted'] for x in records if min(9,x['stagnation_before'])==st),mode_counts=[sum(x['mode']==m and min(9,x['stagnation_before'])==st for x in records) for m in range(3)]) for st in range(10)]
    report['direction_statistics']=[dict(direction=k,count=sum(x['direction']==k for x in records),q_sum=sum(x['q'] for x in records if x['direction']==k),accepted=sum(x['accepted'] for x in records if x['direction']==k),mode_counts=[sum(x['mode']==m and x['direction']==k for x in records) for m in range(3)]) for k in range(16)]
    report['mode_diagnostics']=[dict(mode=m,steps=sum(x['mode']==m for x in records),accepted=sum(x['mode']==m and x['accepted'] for x in records),vehicle_reductions=sum(x['mode']==m and x['candidate_objectives'] is not None and x['candidate_objectives'][0]<x['before_objectives'][0] for x in records),accepted_vehicle_reductions=sum(x['mode']==m and x['accepted'] and x['current_objectives'][0]<x['before_objectives'][0] for x in records),first_route_cap_exceptions=sum(x['mode']==m and x['first_route_exceeds_cap'] for x in records)) for m in range(3)]
    report.update(feature_reduction_precision='legacy_float32_r2' if PLAN['model_source']=='r2' else 'stable_float64_reduction_float32_network',method=method,legacy_u_calibration=False,q_head_every_step=True,external_pulse=False,mechanism_revision='stage2_r3_full12pass_co_adapted' if PLAN['model_source']=='r3' else 'stage2_r2_structured_modes_unified_scalar',model_source=PLAN['model_source'],mechanism_decisions=read(ROOT/'DECISIONS_PENDING.json'),initial_population=initial,scope=PLAN['scope'],technical_smoke=smoke,checkpoint_sha256=PLAN['checkpoint_sha256'],evaluated_checkpoint_sha256=PLAN['checkpoint_sha256'],checkpoint_source_sha256=PLAN['original_meta_source_sha256'],frozen_tensors_verified=True,worker_pid=os.getpid(),worker_created=psutil.Process().create_time(),worker_affinity=psutil.Process().cpu_affinity(),gpu_used=True,hardware='Same desktop later load epoch; three pinned numerical-single-thread CUDA FP32 workers',cache_diagnostics=cache,q_diagnostics=records_q(records,n),steps=len(records))
    write(path,report)
    compact={key:value for key,value in report.items() if key not in ('records','variation_events','point_events')}
    compact['raw_bytes']=path.stat().st_size;write(OUT/'metric_inputs'/path.name,compact)
    result=dict(iid=iid,seed=seed,method=method,steps=len(records),search_seconds=report['search_seconds'],initial_seconds=report['initial_seconds'],inference_seconds=report['inference_seconds'],archive=len(report['archive']),cache_hit_rate=cache['hit_rate'])
    write(OUT/f'worker_{os.getpid()}.json',{**info,'phase':'complete','completed_unix':time.time(),'result':result});return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default=str(ROOT/'inputs'));ap.add_argument('--checkpoint-dir',default=str(ROOT/'stage2_training/checkpoints'));ap.add_argument('--output-dir',default=str(ROOT/'stage2_evaluation'));ap.add_argument('--smoke',action='store_true');ap.add_argument('--ids',nargs='+');ap.add_argument('--seeds',nargs='+',type=int,default=[20261024,20261025,20261026]);ap.add_argument('--method',default='step5_stage2_r3');ap.add_argument('--model-source',choices=['r2','r3'],default='r3');args=ap.parse_args()
    import psutil
    psutil.Process().cpu_affinity([14,15]);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    from stage2_common import require_stage1
    require_stage1()
    assert read(ROOT/'stage2_training/complete.json')['all_directions_updated']
    plan=read(ROOT/'plan.json');plan['method']=args.method;assert plan['new_runs']==900
    decisions=read(ROOT/'DECISIONS_PENDING.json')
    if not (decisions.get('stage1_approved') and decisions['stage1_reference_penalty'] is not None and decisions['stage1_pulse_mapping']=='linear' and decisions['stage1_candidate_anchor_included'] is True):
        raise RuntimeError('Stage1 mechanism definitions still await explicit user selection')
    assert read(ROOT/'PREFLIGHT_STAGE2.json')['passed']
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    ids=args.ids or plan['test_ids'];assert set(ids)<=set(plan['test_ids']) and set(args.seeds)<=set(plan['search_seeds'])
    requested=[(i,s,False) for s in args.seeds for i in ids]
    cases=[];reused=[]
    for iid,seed,smoke in requested:
        p=out/f'{iid}_seed{seed}_{args.method}.json';compact=out/'metric_inputs'/p.name
        if p.exists():
            if not compact.exists():raise RuntimeError('Incomplete raw/compact boundary; preserve and inspect '+str(p))
            d=read(compact);assert (d['iid'],d['seed'],d['method'],d['model_source'])==(iid,seed,args.method,args.model_source);reused.append((iid,seed))
        else:cases.append((iid,seed,False))
    if args.smoke:cases=[('150-0-0',20261022,True)]
    write(out/'process.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),argv=sys.argv,expected_runs=len(cases),requested_runs=len(requested),reused_boundary_runs=len(reused),smoke=args.smoke,method=plan['method'],training_dispatched=False))
    started=time.time();done=[]
    try:
        with ProcessPoolExecutor(3,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(args.data_dir,args.checkpoint_dir,str(out),args.method,args.model_source)) as ex:
            pending={ex.submit(job,t):t for t in cases}
            for future in as_completed(pending):
                result=future.result();done.append(result)
                write(out/'progress.json',dict(completed=len(done),total=len(cases),updated_unix=time.time(),last=result,wall_seconds=time.time()-started))
                log(out/'evaluation.log',completed=len(done),total=len(cases),result=result)
        assert len(done)==len(cases)
        write(out/'complete.json',dict(completed=True,runs=len(done)+len(reused),new_runs=len(done),reused_runs=len(reused),smoke=args.smoke,wall_seconds=time.time()-started,frozen_tensors_verified=True,method=plan['method'],training_seconds=0))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':mp.freeze_support();main()

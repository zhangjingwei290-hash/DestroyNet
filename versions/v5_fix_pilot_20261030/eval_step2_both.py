"""Same-worker paired evaluation of shared meta and direction-adapted step2."""
from pilot_common import ROOT,read,write,sha,reset,log
import os,sys,time,copy,traceback,argparse,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
_POOL=None;_CFG=None;_OUT=None;_POLICIES=None;_DIGESTS=None

def initialize(checkpoint_dir,data_dir,out):
    global _POOL,_CFG,_OUT,_POLICIES,_DIGESTS
    sys.path.insert(0,str(ROOT/'step2_repo'))
    import torch,psutil
    from run import load_config
    from solver.model import Policy
    from eval import EvalPool
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    index=(mp.current_process()._identity[0]-1)%3
    psutil.Process().cpu_affinity([[0,1],[2,3],[4,5]][index]);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    _CFG=load_config(ROOT/'config.json');_POOL=EvalPool(_CFG,data_dir);_OUT=Path(out)
    actors=[];digests=[]
    for k in range(16):
        path=Path(checkpoint_dir)/f'direction_{k:02d}.pt'
        ck=torch.load(path,map_location='cuda',weights_only=False)
        if ck['direction']!=k or ck['updates']<1:raise ValueError('Missing direction optimization')
        policy=Policy(_CFG).cuda();policy.load_state_dict(ck['model']);policy.eval();actors.append(policy);digests.append(sha(path))
    meta_path=Path(checkpoint_dir).parent/'meta_pilot.pt'
    ck=torch.load(meta_path,map_location='cuda',weights_only=False)
    meta=Policy(_CFG).cuda();meta.load_state_dict(ck['model']);meta.eval()
    assert len(meta.heads)==1
    _POLICIES={'step2_meta_only':[meta]*16,'step2_adapted':actors}
    _DIGESTS={'step2_meta_only':[sha(meta_path)],'step2_adapted':digests}
    write(_OUT/f'worker_{os.getpid()}.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),phase='ready',affinity=psutil.Process().cpu_affinity(),checkpoint_sha256=_DIGESTS))

def one(iid,seed,method,smoke):
    from solver.outer import Search
    from diagnostics import records_q
    path=_OUT/f'{iid}_seed{seed}_{method}.json'
    if path.exists():raise FileExistsError('Refuse to overwrite '+str(path))
    reset(seed)
    search=Search(_POOL,iid,_CFG,seed,policies=_POLICIES[method],seconds=16. if smoke else None)
    initial=dict(routes=[copy.deepcopy(p.X[0].routes) for p in search.algorithm.pop],objectives=[p.F.tolist() for p in search.algorithm.pop])
    base=Path(read(ROOT/'plan.json')['baseline_root'])
    for old_method in ('alns_tuned','v5_meta_frozen'):
        old=read(base/'metric_inputs/results'/f'{iid}_seed{seed}_{old_method}.json')
        if old['initial_population']!=initial:raise RuntimeError('Initial populations are not paired')
        if not smoke and old['budget_seconds']!=search.budget:raise RuntimeError('Budget differs')
        if old['physical_offset']!=search.base.offset.tolist() or old['physical_scale']!=search.base.scales.tolist():raise RuntimeError('Physical normalization differs')
    search.run();report=search.report()
    for p in report['archive']:
        if not _POOL.api.feasible(search.base.solution(p['routes']),search.base.instance):raise RuntimeError('Infeasible final archive')
    report.update(initial_population=initial,method=method,stage=2,technical_smoke=smoke,
        scope='authorized_realworld9x3_pilot_only',hardware='same desktop, three pinned paired workers, numerical threads1, CUDA0',
        checkpoint_source_sha256=read(ROOT/'plan.json')['checkpoint_sha256'],evaluated_checkpoint_sha256=_DIGESTS[method],
        q_diagnostics=records_q(report['records'],int(iid.split('-')[0])))
    write(path,report)
    compact={k:v for k,v in report.items() if k not in ('records','variation_events','point_events')}
    compact['raw_bytes']=path.stat().st_size;write(_OUT/'metric_inputs'/path.name,compact)
    return dict(iid=iid,seed=seed,method=method,steps=len(report['records']),search_seconds=report['search_seconds'],archive=len(report['archive']))

def pair(task):
    import psutil
    iid,seed,index,smoke=task
    order=['step2_meta_only','step2_adapted']
    if index%2:order.reverse()
    info=dict(pid=os.getpid(),created=psutil.Process().create_time(),phase='running',iid=iid,seed=seed,method_order=order,started_unix=time.time())
    write(_OUT/f'worker_{os.getpid()}.json',info)
    rows=[]
    for method in order:
        info['active_method']=method;write(_OUT/f'worker_{os.getpid()}.json',info)
        rows.append(one(iid,seed,method,smoke))
    info.update(phase='pair_complete',completed_unix=time.time());write(_OUT/f'worker_{os.getpid()}.json',info)
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default=str(ROOT/'inputs'))
    ap.add_argument('--checkpoint-dir',default=str(ROOT/'step2_training/checkpoints'))
    ap.add_argument('--output-dir',default=str(ROOT/'step2_evaluation'));ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    import psutil
    psutil.Process().cpu_affinity([14,15]);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    plan=read(ROOT/'plan.json');cases=[(i,s) for s in plan['search_seeds'] for i in plan['test_ids']]
    if args.smoke:cases=[(plan['test_ids'][k],plan['search_seeds'][0]) for k in (0,3,6)]
    tasks=[(i,s,j,args.smoke) for j,(i,s) in enumerate(cases)]
    write(out/'process.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),stage=2,argv=sys.argv,expected_runs=2*len(tasks),expected_blocks=len(tasks),methods=['step2_meta_only','step2_adapted'],smoke=args.smoke))
    done=[];start=time.time()
    try:
        with ProcessPoolExecutor(3,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(args.checkpoint_dir,args.data_dir,str(out))) as executor:
            futures=[executor.submit(pair,t) for t in tasks]
            for future in as_completed(futures):
                results=future.result();done.extend(results)
                write(out/'progress.json',dict(completed=len(done),total=2*len(tasks),completed_blocks=len(done)//2,total_blocks=len(tasks),updated_unix=time.time(),last=results))
                log(out/'evaluation.log',dict(pair=results,completed=len(done)))
        write(out/'complete.json',dict(completed=True,runs=len(done),blocks=len(tasks),smoke=args.smoke,wall_seconds=time.time()-start,methods=['step2_meta_only','step2_adapted'],same_worker_per_pair=True,alternating_order=True))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':mp.freeze_support();main()

"""Three pinned independent searches; same full budgets and initial populations."""
from pilot_common import ROOT,read,write,sha,reset,log
import argparse,sys,os,copy,time,traceback,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
_ACTORS=None
_POOL=None
_CFG=None
_OUT=None
_STAGE=None

class EvalPool:
    def __init__(self,cfg,data_dir):
        from solver.data import project_api
        from solver.config import q_values
        self.api=project_api(cfg.project_root);self.data_dir=Path(data_dir)
        self.boxes=read(ROOT/'boxes.json');self.ids=read(ROOT/'plan.json')['test_ids'];self.cache={}
        self.q_values=q_values
    def load(self,iid):
        if iid not in self.ids:raise ValueError('Outside fixed nine evaluation instances')
        if iid not in self.cache:
            row=read(self.data_dir/f'{iid}.json')
            if row['instance_id']!=iid or row['split']!='test':raise ValueError('Bad evaluation identity')
            instance=self.api.instance(row)
            source=self.api.Solution([self.api.Route(list(r)) for r in row['source_solution']])
            if not self.api.feasible(source,instance):raise ValueError('Infeasible input source')
            self.cache[iid]=instance,row,copy.deepcopy(self.boxes[iid]),self.q_values(instance.n_customers)
        return self.cache[iid]

def initialize(stage,checkpoint_dir,data_dir,out):
    global _ACTORS,_POOL,_CFG,_OUT,_STAGE
    sys.path.insert(0,str(ROOT/('repo' if stage==1 else 'step2_repo')))
    import torch,psutil
    from run import load_config
    from solver.model import Policy
    index=(mp.current_process()._identity[0]-1)%3
    psutil.Process().cpu_affinity([[0,1],[2,3],[4,5]][index]);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for neural evaluation')
    _CFG=load_config(ROOT/'config.json');_OUT=Path(out);_STAGE=stage
    _POOL=EvalPool(_CFG,data_dir);_ACTORS=[];digests=[]
    for k in range(16):
        path=Path(checkpoint_dir)/f'direction_{k:02d}.pt'
        checkpoint=torch.load(path,map_location='cuda',weights_only=False)
        if checkpoint['direction']!=k or checkpoint['updates']<1:raise ValueError('Direction is missing adaptation')
        actor=Policy(_CFG).cuda();actor.load_state_dict(checkpoint['model']);actor.eval()
        _ACTORS.append(actor);digests.append(sha(path))
    write(_OUT/f'worker_{os.getpid()}.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),affinity=psutil.Process().cpu_affinity(),stage=stage,checkpoint_sha256=digests,phase='ready'))

def evaluate(task):
    import torch,psutil,numpy as np
    from solver.outer import Search
    from diagnostics import records_q
    iid,seed,smoke=task;method=f'step{_STAGE}_adapted'
    path=_OUT/f'{iid}_seed{seed}_{method}.json'
    if path.exists():raise FileExistsError('Refuse to overwrite evaluation '+str(path))
    info=dict(pid=os.getpid(),created=psutil.Process().create_time(),iid=iid,seed=seed,stage=_STAGE,phase='running',started_unix=time.time())
    write(_OUT/f'worker_{os.getpid()}.json',info)
    reset(seed)
    search=Search(_POOL,iid,_CFG,seed,policies=_ACTORS,seconds=16. if smoke else None)
    initial=dict(routes=[copy.deepcopy(p.X[0].routes) for p in search.algorithm.pop],objectives=[p.F.tolist() for p in search.algorithm.pop])
    base=Path(read(ROOT/'plan.json')['baseline_root'])
    for baseline_method in ('alns_tuned','v5_meta_frozen'):
        old=read(base/'metric_inputs/results'/f'{iid}_seed{seed}_{baseline_method}.json')
        if old['initial_population']!=initial:raise RuntimeError('Initial population differs from reused baseline')
        if not smoke:
            if old['budget_seconds']!=search.budget:raise RuntimeError('Outer budget differs')
        for key,value in [('physical_offset',search.base.offset.tolist()),('physical_scale',search.base.scales.tolist())]:
            if old[key]!=value:raise RuntimeError('Physical normalization differs')
    search.run();report=search.report()
    for point in report['archive']:
        if not _POOL.api.feasible(search.base.solution(point['routes']),search.base.instance):raise RuntimeError('Infeasible final archive')
    report.update(initial_population=initial,method=method,stage=_STAGE,technical_smoke=smoke,
        scope='authorized_realworld9x3_pilot_only',
        hardware='same desktop; three pinned numerical-single-thread CUDA search processes; later load epoch',
        checkpoint_source_sha256=read(ROOT/'plan.json')['checkpoint_sha256'],
        q_diagnostics=records_q(report['records'],int(iid.split('-')[0])))
    write(path,report)
    compact={k:v for k,v in report.items() if k not in ('records','variation_events','point_events')}
    compact['raw_bytes']=path.stat().st_size;write(_OUT/'metric_inputs'/path.name,compact)
    info.update(phase='complete',completed_unix=time.time(),steps=len(report['records']),search_seconds=report['search_seconds'])
    write(_OUT/f'worker_{os.getpid()}.json',info)
    return dict(iid=iid,seed=seed,method=method,steps=len(report['records']),search_seconds=report['search_seconds'],archive=len(report['archive']))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=int,choices=(1,2),required=True)
    ap.add_argument('--data-dir',default=str(ROOT/'inputs'));ap.add_argument('--checkpoint-dir')
    ap.add_argument('--output-dir');ap.add_argument('--seed',type=int);ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();stage=args.stage
    out=Path(args.output_dir or ROOT/f'step{stage}_evaluation');out.mkdir(parents=True,exist_ok=False)
    checkpoints=args.checkpoint_dir or str(ROOT/f'step{stage}_training/checkpoints')
    plan=read(ROOT/'plan.json');seeds=[args.seed] if args.seed is not None else plan['search_seeds']
    tasks=[(iid,seed,False) for seed in seeds for iid in plan['test_ids']]
    if args.smoke:tasks=[(plan['test_ids'][j],seeds[0],True) for j in (0,3,6)]
    import psutil
    psutil.Process().cpu_affinity([14,15]);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    write(out/'process.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),stage=stage,argv=sys.argv,expected_runs=len(tasks),smoke=args.smoke))
    done=[];start=time.time()
    try:
        with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(stage,checkpoints,args.data_dir,str(out))) as executor:
            futures=[executor.submit(evaluate,t) for t in tasks]
            for future in as_completed(futures):
                result=future.result();done.append(result)
                write(out/'progress.json',dict(completed=len(done),total=len(tasks),updated_unix=time.time(),last=result))
                log(out/'evaluation.log',result)
        write(out/'complete.json',dict(completed=True,runs=len(done),smoke=args.smoke,wall_seconds=time.time()-start))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':mp.freeze_support();main()

"""Read-only frozen meta evaluation. Explicit --run required; default preflight only."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse,copy,hashlib,json,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'repo'))
import numpy as np
import torch
import moocore
from run import load_config
from solver.data import TrainPool

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    with tmp.open('w',encoding='utf-8') as handle:
        json.dump(d,handle,ensure_ascii=False,separators=(',',':'),allow_nan=False,default=convert)
    tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class TestPool(TrainPool):
    """Explicit test-only reader; never handed to training or adaptation."""
    def __init__(self,cfg):
        super().__init__(cfg)
        self.ids=tuple(self.split['test_ids']);self.expected_split={i:'test' for i in self.ids}
        self.cache={}

def setup():
    torch.set_num_threads(1)
    cfg=load_config(ROOT/'eval_config.json');pool=TestPool(cfg)
    from solver.model import Policy
    from solver.outer import Search
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    plan=read(ROOT/'plan.json')
    if sha(ROOT/'checkpoints/meta_complete.pt')!=plan['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    for name,digest in read(ROOT/'source_hashes.json').items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed: '+name)
    for iid,item in read(ROOT/'input_hashes.json').items():
        if sha(item['path'])!=item['sha256']:raise ValueError('Test input changed: '+iid)
    ck=torch.load(ROOT/'checkpoints/meta_complete.pt',map_location='cuda',weights_only=False)
    model=Policy(cfg).cuda();model.load_state_dict(ck['model'],strict=True);model.eval()
    model.requires_grad_(False)
    return cfg,pool,model,Search,plan

def reset(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def initial(s):
    return dict(routes=[copy.deepcopy(p.X[0].routes) for p in s.algorithm.pop],objectives=[p.F.tolist() for p in s.algorithm.pop])
def assert_pair(a,b):
    if a['initial_population']!=b['initial_population']:raise ValueError('Paired initial routes/objectives differ')
    for k in ('physical_offset','physical_scale','normalized_reference','budget_seconds'):
        if a[k]!=b[k]:raise ValueError('Pair differs: '+k)

def preflight():
    cfg,pool,model,Search,plan=setup();rows=[]
    for iid in plan['ids']:
        inst,row,box,_=pool.load(iid)
        if box['reference']!=[1.1]*5 or min(box['scale'])<=0:raise ValueError('Invalid box')
        rows.append(dict(iid=iid,n=inst.n_customers,initial_vehicles=len(row['source_solution']),budget=cfg.solve_seconds_by_instance[iid]))
    # Compare full initial routes as well as objectives for one instance of each size.
    checked=[]
    for n in (50,150,250):
        iid=next(x['iid'] for x in rows if x['n']==n);seed=plan['search_seeds'][0]
        reset(seed);a=Search(pool,iid,cfg,seed,None)
        reset(seed);b=Search(pool,iid,cfg,seed,[model]*16)
        if initial(a)!=initial(b):raise ValueError('Initial population differs')
        checked.append(iid)
    # Known 5D rectangle and IGD+ orientation; calls external library only.
    np.testing.assert_allclose(moocore.hypervolume(np.array([[.2]*5]),ref=[1.1]*5),.9**5)
    np.testing.assert_allclose(moocore.igd_plus(np.array([[0.,0.]]),ref=np.array([[1.,1.]])),0.)
    np.testing.assert_allclose(moocore.igd_plus(np.array([[1.,1.]]),ref=np.array([[0.,0.]])),2**.5)
    write(ROOT/'preflight.json',dict(passed=True,source_feasible=len(rows),initial_pair_checks=checked,instances=rows,search_started=False,training_started=False,checkpoint_sha256=plan['checkpoint_sha256']))
    print('PREFLIGHT PASSED: 45 feasible sources; 3 paired initial populations; metric fixtures; no search run')

def run(resume):
    cfg,pool,model,Search,plan=setup();out=ROOT/'results'
    if out.exists() and not resume:raise RuntimeError('Results exist; use explicit --resume')
    out.mkdir(exist_ok=True)
    import psutil
    current=psutil.Process()
    for p in psutil.process_iter(['pid','cmdline']):
        try:
            cmd=p.info['cmdline'] or []
            if p.pid!=current.pid and any(str(ROOT/name).lower()==str(x).lower() for name in ('evaluate.py','parallel_evaluate.py') for x in cmd) and '--run' in cmd:raise RuntimeError('Duplicate evaluation process')
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    frozen={k:v.clone() for k,v in model.state_dict().items()}
    write(out/'process.json',dict(pid=os.getpid(),created=current.create_time(),command=sys.argv,start_unix=time.time(),scope=plan['scope']))
    try:
        done=0
        for si,seed in enumerate(plan['search_seeds']):
            for ii,iid in enumerate(plan['ids']):
                methods=plan['methods'] if (si+ii)%2==0 else list(reversed(plan['methods']));pair={}
                for method in methods:
                    path=out/f'{iid}_seed{seed}_{method}.json'
                    if path.exists():
                        report=read(path)
                        if (report['iid'],report['seed'],report['method'],report['checkpoint_sha256'])!=(iid,seed,method,plan['checkpoint_sha256']):raise ValueError('Resume identity mismatch')
                    else:
                        reset(seed);s=Search(pool,iid,cfg,seed,None if method=='heuristic' else [model]*16)
                        start=initial(s);s.run();report=s.report()
                        report.update(method=method,scope=plan['scope'],initial_population=start,checkpoint_sha256=plan['checkpoint_sha256'])
                        # report() independently verifies every final archived route and objective.
                        vals=(np.array(start['objectives'])-s.base.offset)/s.base.scales
                        report['initial_hv']=float(moocore.hypervolume(vals,ref=s.base.box['reference']))
                        report['objective_minima']=np.array([p['objectives'] for p in report['archive']]).min(0).tolist()
                        if any(not torch.equal(v,model.state_dict()[k]) for k,v in frozen.items()):raise RuntimeError('Frozen weights changed')
                        write(path,report)
                    pair[method]=report;done+=1
                    write(out/'progress.json',dict(phase='evaluation',completed_runs=done,total_runs=1800,iid=iid,seed=seed,method=method,updated_unix=time.time()))
                assert_pair(pair['heuristic'],pair['v5_meta_frozen'])
        from summarize import main
        main()
        write(out/'complete.json',dict(completed=True,runs=1800,pairs=900,frozen_weights_verified=True,completed_unix=time.time()))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.resume and not a.run:p.error('--resume requires --run')
    run(a.resume) if a.run else preflight()

"""Formal frozen evaluation utilities; imports no neural library by default."""
import os,sys,json,time,hashlib,random,copy
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='0'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'repo'))
def read(path):
    for attempt in range(4):
        try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
        except (PermissionError,json.JSONDecodeError):
            if attempt==3:raise
            time.sleep(.1)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def convert(v):
        if hasattr(v,'tolist'):return v.tolist()
        if hasattr(v,'item'):return v.item()
        raise TypeError(type(v).__name__)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,default=convert,separators=(',',':')),encoding='utf-8')
    for attempt in range(5):
        try:os.replace(temp,path);return
        except PermissionError:
            if attempt==4:raise
            time.sleep(.1)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def log(path,**data):
    from datetime import datetime,timezone,timedelta
    value=dict(time=datetime.now(timezone(timedelta(hours=8))).isoformat(),**data)
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False,default=str)+'\n')
    print(json.dumps(value,ensure_ascii=False,default=str),flush=True)
def reset(seed):
    import numpy as np,torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def load_cfg():
    from run import load_config
    cfg=load_config(ROOT/'eval_config.json')
    decisions=read(ROOT/'DECISIONS_PENDING.json')
    cfg.step5_reference_penalty=decisions['stage1_reference_penalty']
    return cfg
class EvalPool:
    def __init__(self,cfg,data_dir=None):
        from solver.data import project_api
        from solver.config import q_values
        self.api=project_api(cfg.project_root);self.q_values=q_values
        self.data_dir=Path(data_dir or ROOT/'inputs');self.ids=read(ROOT/'plan.json')['test_ids']
        self.boxes=read(ROOT/'boxes.json');self.cache={}
    def load(self,iid):
        if iid not in self.ids:raise ValueError('Outside approved12 test inputs')
        if iid not in self.cache:
            row=read(self.data_dir/f'{iid}.json')
            assert row['instance_id']==iid and row['split']=='test'
            inst=self.api.instance(row)
            source=self.api.Solution([self.api.Route(list(v)) for v in row['source_solution']])
            assert self.api.feasible(source,inst)
            self.cache[iid]=inst,row,copy.deepcopy(self.boxes[iid]),self.q_values(inst.n_customers)
        return self.cache[iid]
def readonly_check():
    receipt=read(ROOT/'READONLY_RECEIPT.json');changed=[];counts={}
    for entry in receipt['files']:
        p=Path(receipt['roots'][entry['root']])/entry['path']
        try:
            st=p.stat();ok=(st.st_size,st.st_mtime_ns)==(entry['size'],entry['mtime_ns'])
        except FileNotFoundError:ok=False
        if not ok:changed.append(dict(root=entry['root'],path=entry['path']))
        counts[entry['root']]=counts.get(entry['root'],0)+1
    result=dict(passed=not changed,counts=counts,changed=changed,checked_unix=time.time())
    assert not changed,result
    return result

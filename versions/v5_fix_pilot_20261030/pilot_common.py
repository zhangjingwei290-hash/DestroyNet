import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json,hashlib,random,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,default=convert,separators=(',',':')),encoding='utf-8')
    for retry in range(5):
        try:os.replace(temp,path);return
        except PermissionError:
            if retry==4:raise
            time.sleep(.1)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def reset(seed):
    import numpy as np,torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def log(path,value):
    from datetime import datetime,timezone,timedelta
    value=dict(time=datetime.now(timezone(timedelta(hours=8))).isoformat(),**value)
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False,default=str)+'\n')
    print(json.dumps(value,ensure_ascii=False,default=str),flush=True)

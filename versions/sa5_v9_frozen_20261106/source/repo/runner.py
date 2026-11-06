from pathlib import Path
import os,sys,json,subprocess,time,hashlib
R=Path(__file__).resolve().parent.parent
F=Path('D:/workspace/destroy-repair-V0.01-publication/versions/v5_step5_r2_frozen_20261102')
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2),encoding='utf8');os.replace(tmp,p)
def run(iid,seed,seconds,gate,root,split,capture=False):
 import psutil
 root.mkdir(parents=True,exist_ok=False)
 env=dict(os.environ,CASE_ROOT=str(root),CASE_IID=iid,CASE_SEED=str(seed),CASE_SECONDS=str(seconds),METHODS='["gate_trained"]',METHOD_WEIGHTS='{}',STAGE_MANIFEST=str(R/'SOURCE_MANIFEST.json'),CHECKPOINT_ROOT=str(F/'checkpoints'),DATA_SPLIT=split,GATE_PATH=str(gate))
 if capture:env.update(TRAIN_CAPTURE='1',SESSION_ROLLOUT='1')
 with (root/'stdout.log').open('wb') as out,(root/'stderr.log').open('wb') as err:
  c=subprocess.Popen([sys.executable,'-B','-X','utf8',str(R/'repo/evaluate_case.py')],env=env,stdout=out,stderr=err)
  p=psutil.Process(c.pid);write(R/'active_child.json',dict(pid=p.pid,created=p.create_time(),cmd=p.cmdline(),root=str(root)))
  code=c.wait()
 if code:raise RuntimeError(f'{root}: exit {code}; '+(root/'stderr.log').read_text()[-4000:])
 assert json.loads((root/'analysis/AUDIT.json').read_text())['passed']
 return dict(iid=iid,n=json.loads((F/f'data/{split}/{iid}.json').read_text())['n_customers'],seed=seed,method='v9',path=str(root/'results/gate_trained.json'))
def manifest():
 files=[]
 for p in sorted((R/'repo').glob('*.py')):
  files.append(dict(windows_path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 write(R/'SOURCE_MANIFEST.json',dict(files=files))
if __name__=='__main__':
 import torch,psutil
 from v9_policy import Controller
 psutil.Process().cpu_affinity([6,7]);torch.set_num_threads(1);torch.manual_seed(20261105)
 gate=R/'checkpoints/initial.pt';gate.parent.mkdir(exist_ok=True);torch.save(Controller().state_dict(),gate)
 manifest();receipts=[]
 for n in (50,150,250):
  rec=run(f'v5p_train_{n}_d0_tw0_0',20261107,16,gate,R/f'smoke/N{n}','fit',True)
  result=json.loads(Path(rec['path']).read_text());rows=torch.load(R/f'smoke/N{n}/session_rollout.pt',weights_only=False)['rows']
  receipts.append(dict(n=n,passed=True,actions=len(result['records']),batches=len(result['batches']),complete_training_rows=len(rows),modes=sorted({r['mode'] for r in result['records']}),mean_q=sum(r['q'] for r in result['records'])/len(result['records'])))
 write(R/'SMOKE.json',dict(passed=True,cases=receipts));print(json.dumps(receipts),flush=True)

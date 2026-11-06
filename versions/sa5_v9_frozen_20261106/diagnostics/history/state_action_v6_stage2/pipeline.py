from pathlib import Path
import os,sys,json,time,subprocess,hashlib,traceback
O=Path(__file__).resolve().parent;P=Path('D:/workspace/fuxian/DPR/development/sa5_coverage_v2_20261104')
def write(name,x):
 p=O/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2),encoding='utf-8');os.replace(q,p)
def run(script,label,env=None):
 import psutil
 with (O/f'{label}.stdout.log').open('wb') as out,(O/f'{label}.stderr.log').open('wb') as err:
  c=subprocess.Popen([sys.executable,'-B','-X','utf8',str(O/script)],env=env,stdout=out,stderr=err);p=psutil.Process(c.pid);write('active_child.json',dict(pid=p.pid,created=p.create_time(),ppid=p.ppid(),cmd=p.cmdline(),stage=label));code=c.wait()
 if code:raise RuntimeError(f'{label} exit {code}')
def lock(stage):
 fs=list(O.glob('*.py'))+[O/'PROTOCOL.md',P/'SOURCE_MANIFEST.json',P/'network.py']+list((P/'training').glob('direction_*.pt'))
 if stage=='evaluation':fs+=[O/'training/gate.pt']
 write(stage+'_SOURCE_MANIFEST.json',dict(files=[dict(windows_path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in fs]));return O/(stage+'_SOURCE_MANIFEST.json')
def case(iid,seed,methods,root,manifest,capture=False):
 root.mkdir(parents=True,exist_ok=False);env=dict(os.environ,CASE_ROOT=str(root),CASE_IID=iid,CASE_SEED=str(seed),CASE_SECONDS='90' if capture else '60',METHODS=json.dumps(methods),METHOD_WEIGHTS='{}',STAGE_MANIFEST=str(manifest),CHECKPOINT_ROOT=str(P/'training'),GATE_PATH=str(O/'training/gate.pt'))
 if capture:env['TRAIN_CAPTURE']='1'
 run('evaluate_case.py',('collect_' if capture else 'eval_')+iid+'_'+str(seed),env)
 assert json.loads((root/'analysis/AUDIT.json').read_text())['passed']
def main():
 import psutil
 p=psutil.Process();p.cpu_affinity([14,15]);fd=os.open(O/'claim.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd);write('process.json',dict(pid=p.pid,created=p.create_time(),ppid=p.ppid(),cmd=p.cmdline()))
 run('preflight.py','preflight');assert json.loads((O/'preflight/PASSED.json').read_text())['passed'];manifest=lock('collection')
 for split,d,tw in [('train',0,0),('heldout',1,2)]:
  for n in (50,150,250):
   write('progress.json',dict(phase='collect_real_states',split=split,n=n));case(f'v5p_train_{n}_d{d}_tw{tw}_0',20261103,['v2_baseline'],O/f'collection/{split}/{n}',manifest,True)
 coverage={}
 for split in ('train','heldout'):
  for n in (50,150,250):
   rows=json.loads((O/f'collection/{split}/{n}/training_states.json').read_text());coverage[f'{split}_{n}']=dict(total=len(rows),long_stall=sum(r['cross_stall']>=3 for r in rows),max_stall=max(r['cross_stall'] for r in rows))
 write('coverage.json',coverage)
 assert all(coverage[f'train_{n}']['long_stall']>0 for n in (50,150,250)),'Insufficient real long-stall training coverage'
 run('train_gate.py','training');write('COMPLETE.json',dict(passed=True,stage='state_conditioned_branch_and_heldout',training=json.loads((O/'training/COMPLETE.json').read_text())))
if __name__=='__main__':
 try:main()
 except BaseException:write('FAILURE.json',dict(error=traceback.format_exc()));raise

from pathlib import Path
import sys,os,json,time,subprocess,hashlib,traceback
O=Path(__file__).resolve().parent;P=Path('D:/workspace/fuxian/DPR/development/sa5_coverage_v2_20261104')
def write(name,x):
 p=O/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2),encoding='utf-8');os.replace(q,p)
def run(script,label,env=None):
 import psutil
 with (O/f'{label}.stdout.log').open('wb') as out,(O/f'{label}.stderr.log').open('wb') as err:
  child=subprocess.Popen([sys.executable,'-B','-X','utf8',str(O/script)],env=env,stdout=out,stderr=err);p=psutil.Process(child.pid);write('active_child.json',dict(pid=p.pid,created=p.create_time(),ppid=p.ppid(),cmd=p.cmdline(),stage=label));code=child.wait()
 if code:raise RuntimeError(f'{label} exited {code}; see retained stderr')
def manifest(stage):
 files=list(O.glob('*.py'))+[O/'PROTOCOL.md',P/'SOURCE_MANIFEST.json',P/'network.py']+list((P/'training').glob('direction_*.pt'))+list((P/'state_bank').glob('*/state_*.json'))
 if stage=='quick2':files+=list((O/'training').glob('*/direction_*.pt'))
 p=O/f'{stage}_SOURCE_MANIFEST.json';write(p.name,dict(files=[dict(windows_path=str(x),sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in files]));return p
def cases(stage,methods,roots):
 lock=manifest(stage)
 for i,iid in enumerate(['50-2-4','150-2-4','250-2-4']):
  write('progress.json',dict(phase=stage,iid=iid,complete_cases=i,total_cases=3,methods=methods,seconds_per_method=60))
  case=O/f'{stage}/{iid}/20261104';case.mkdir(parents=True,exist_ok=False)
  env=dict(os.environ,CASE_ROOT=str(case),CASE_IID=iid,CASE_SEED='20261104',CASE_SECONDS='60',METHODS=json.dumps(methods),METHOD_WEIGHTS=json.dumps(roots),STAGE_MANIFEST=str(lock),CHECKPOINT_ROOT=str(P/'training'))
  run('evaluate_case.py',f'{stage}_{iid}',env);assert json.loads((case/'analysis/AUDIT.json').read_text())['passed']
def main():
 import psutil
 p=psutil.Process();p.cpu_affinity([0,1]);fd=os.open(O/'claim.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd);write('process.json',dict(pid=p.pid,created=p.create_time(),ppid=p.ppid(),cmd=p.cmdline()))
 run('preflight.py','preflight');assert json.loads((O/'preflight/PASSED.json').read_text())['passed']
 cases('quick1',['v2_baseline','v2_stallq'],{})
 write('progress.json',dict(phase='matched_training',completed_searches=6));run('train_pair.py','training');assert json.loads((O/'training/COMPLETE.json').read_text())['passed']
 cases('quick2',['v2_baseline','v2_ppo_continued','v2_rloo_continued'],dict(v2_ppo_continued=str(O/'training/ppo'),v2_rloo_continued=str(O/'training/rloo')))
 run('report.py','analysis');write('COMPLETE.json',dict(passed=True,searches=15,quick1_searches=6,quick2_searches=9,report='REPORT.md'))
if __name__=='__main__':
 try:main()
 except BaseException:write('FAILURE.json',dict(error=traceback.format_exc()));raise

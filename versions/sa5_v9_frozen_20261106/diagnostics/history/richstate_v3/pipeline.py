from pathlib import Path
import sys,os,time,json,subprocess,hashlib,traceback
O=Path(__file__).resolve().parent;P=Path('D:/workspace/fuxian/DPR/development/sa5_coverage_v2_20261104')
def write(name,d):
 p=O/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2),encoding='utf-8');os.replace(t,p)
def run(script,label,env=None):
 with (O/f'{label}.stdout.log').open('wb') as out,(O/f'{label}.stderr.log').open('wb') as err:
  p=subprocess.Popen([sys.executable,'-B','-X','utf8',str(O/script)],env=env,stdout=out,stderr=err)
  import psutil
  pr=psutil.Process(p.pid);write('active_child.json',dict(pid=pr.pid,created=pr.create_time(),ppid=pr.ppid(),cmd=pr.cmdline(),phase=label))
  code=p.wait()
  if code:raise RuntimeError(f'{label} exited {code}; see retained stderr')
def main():
 import psutil
 psutil.Process().cpu_affinity([0,1,2,3,4,5]);fd=os.open(O/'claim.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd);pr=psutil.Process();write('process.json',dict(pid=pr.pid,created=pr.create_time(),ppid=pr.ppid(),cmd=pr.cmdline()))
 run('train.py','training');assert json.loads((O/'training/COMPLETE.json').read_text())['passed']
 # Freeze new implementation, source bank and all used checkpoint bytes.
 files=list(O.glob('*.py'))+[O/'PROTOCOL.md',P/'SOURCE_MANIFEST.json',P/'network.py']+list((O/'training').glob('*.pt'))+list((P/'training').glob('direction_*.pt'))+list((P/'state_bank').glob('*/state_*.json'))
 write('SOURCE_MANIFEST.json',dict(files=[dict(windows_path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]))
 for i,iid in enumerate(['50-2-4','150-2-4','250-2-4']):
  write('progress.json',dict(phase='four_method_development_evaluation',iid=iid,completed_cases=i,total_cases=3,methods=4,seconds_per_method=60))
  case=O/f'cases/{iid}/20261104';case.mkdir(parents=True,exist_ok=False);env=dict(os.environ,CASE_ROOT=str(case),CASE_IID=iid,CASE_SEED='20261104',CASE_SECONDS='60')
  run('evaluate_case.py','case_'+iid,env);assert json.loads((case/'analysis/AUDIT.json').read_text())['passed']
 run('report.py','analysis');write('COMPLETE.json',dict(passed=True,development_searches=12,training_budget=900,report='REPORT.md'))
if __name__=='__main__':
 try:main()
 except BaseException:write('FAILURE.json',dict(error=traceback.format_exc()));raise

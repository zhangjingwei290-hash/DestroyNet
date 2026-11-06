from pipeline import *
def finish():
 import psutil
 fd=os.open(O/'frozen_resume_claim.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd);pr=psutil.Process();write('process.json',dict(pid=pr.pid,created=pr.create_time(),ppid=pr.ppid(),cmd=pr.cmdline()))
 run('finish_frozen_validation.py','frozen_validation');assert json.loads((O/'training/COMPLETE.json').read_text())['passed']
 files=list(O.glob('*.py'))+[O/'PROTOCOL.md',P/'SOURCE_MANIFEST.json',P/'network.py']+list((O/'training').glob('*.pt'))+list((P/'training').glob('direction_*.pt'))+list((P/'state_bank').glob('*/state_*.json'))
 write('SOURCE_MANIFEST.json',dict(files=[dict(windows_path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]))
 for i,iid in enumerate(['50-2-4','150-2-4','250-2-4']):
  write('progress.json',dict(phase='four_method_development_evaluation',iid=iid,completed_cases=i,total_cases=3,methods=4,seconds_per_method=60))
  case=O/f'cases/{iid}/20261104';case.mkdir(parents=True,exist_ok=False);env=dict(os.environ,CASE_ROOT=str(case),CASE_IID=iid,CASE_SEED='20261104',CASE_SECONDS='60')
  run('evaluate_case.py','case_'+iid,env);assert json.loads((case/'analysis/AUDIT.json').read_text())['passed']
 run('report.py','analysis');write('COMPLETE.json',dict(passed=True,development_searches=12,optimization_ceiling_seconds=600,combined_training_validation_original_limit_passed=False,frozen_validation_extension=True,report='REPORT.md'))
if __name__=='__main__':
 try:finish()
 except BaseException:write('FAILURE.json',dict(error=traceback.format_exc()));raise

"""Verify packaged files, Python syntax and recorded numerical tables; no models loaded."""
import ast,csv,hashlib,json,statistics,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 manifest=json.loads((ROOT/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
 for p,d in manifest.items():assert sha(ROOT/p)==d,p
 copies=json.loads((ROOT/'ARCHIVE_MANIFEST.json').read_text(encoding='utf-8'))['files']
 for row in copies:assert sha(ROOT/row['path'])==row['sha256'],row['path']
 sources=list(ROOT.rglob('*.py'))
 for p in sources:ast.parse(p.read_text(encoding='utf-8-sig'))
 s=json.loads((ROOT/'versions/v5/results/summary.json').read_text(encoding='utf-8'))
 assert (s['raw_runs'],s['paired_blocks'],s['instances'])==(1800,900,45)
 assert sha(ROOT/'versions/v5/checkpoints/meta_complete.pt')=='1a3493e34e4c23f7eae7c465ba6c48a69d49884a01ec80d599eec6cd9c48187f'
 for version in ['v4','v4_3']:
  s=json.loads((ROOT/f'versions/{version}/results/summary.json').read_text())
  assert len(s['comparison'])==45 and len(s['search_seeds'])==5
  assert len(list((ROOT/f'versions/{version}/checkpoints').glob('direction_*.pt')))==16
 assert len(list((ROOT/'data/v5/instances').glob('*.json')))==270
 pilot=ROOT/'versions/v5_fix_pilot_20261030'
 complete=json.loads((pilot/'complete.json').read_text())
 assert (complete['total_runs'],complete['new_runs'],complete['reused_runs'],complete['methods'])==(135,81,54,5)
 assert json.loads((pilot/'step2_analysis/VERIFICATION.json').read_text())['passed']
 assert sha(pilot/'step2_training/meta_pilot.pt')=='ab8ae6bf87a327792ba4c5577d1246c98cdea06dfc37ffbe72b36786909f5d6b'
 for stage in (1,2):
  assert len(list((pilot/f'step{stage}_training/checkpoints').glob('direction_*.pt')))==16
 assert len(list((pilot/'data/inputs').glob('*.json')))==270
 assert sum(len(list((pilot/folder).glob('*_seed*.json'))) for folder in ('baseline_cache','step1_evaluation/metric_inputs','step2_evaluation/metric_inputs'))==135
 with (pilot/'step2_analysis/per_run_metrics.csv').open(encoding='utf-8-sig',newline='') as f:metrics=list(csv.DictReader(f))
 with (pilot/'step2_analysis/main_table.csv').open(encoding='utf-8-sig',newline='') as f:table=list(csv.DictReader(f))
 assert len(metrics)==135 and len(table)==45
 assert len({(r['iid'],r['seed'],r['method']) for r in metrics})==135
 for r in table:
  local=[x for x in metrics if x['iid']==r['iid'] and x['method']==r['method']];assert len(local)==3
  for key in ('hv','igd_plus'):
   values=[float(x[key]) for x in local]
   assert math.isclose(float(r[key+'_mean']),statistics.mean(values),rel_tol=1e-12,abs_tol=1e-12)
   assert math.isclose(float(r[key+'_sd']),statistics.stdev(values),rel_tol=1e-12,abs_tol=1e-12)
 contents=json.loads((pilot/'PACKAGE_CONTENTS.json').read_text())
 for entry in contents['files']:assert sha(pilot/entry['path'])==entry['sha256'],entry['path']
 print(json.dumps(dict(passed=True,manifest_files=len(manifest),original_byte_copies=len(copies),python_files_parsed=len(sources),neural_model_loaded=False,search_or_training_executed=False)))
if __name__=='__main__':main()

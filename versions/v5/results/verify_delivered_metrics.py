"""Recalculate per-seed metrics from delivered raw archives; no GPU/weight loading."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,csv,json
from pathlib import Path
import numpy as np,moocore
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parent

def head(path):
 data=b'';marker=b',"records":'
 with path.open('rb') as f:
  while marker not in data:
   b=f.read(65536)
   if not b:raise ValueError('Missing archive-prefix boundary')
   data+=b
 return json.loads(data.split(marker,1)[0]+b'}')

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'复核输出');a=p.parse_args();a.output.mkdir(exist_ok=True)
 with (ROOT/'main_table.csv').open(encoding='utf-8-sig') as f:expected=list(csv.DictReader(f))
 per_run=[];per_iid=[];seeds=range(20261024,20261044)
 for item in expected:
  iid=item['iid'];r=json.loads((ROOT/'共同经验参考前沿'/f'{iid}_reference.json').read_text());ideal=np.array(r['empirical_ideal']);nadir=np.array(r['empirical_nadir']);zero=np.zeros(5,dtype=bool);zero[r['zero_span_objectives']]=True;span=np.where(zero,1,nadir-ideal);pf=np.asarray(r['normalized_points'])
  row=dict(iid=iid)
  for method,label in [('heuristic','heuristic'),('v5_meta_frozen','v5')]:
   scores=[]
   for seed in seeds:
    archive=head(ROOT/'原始运行_JSON'/f'{iid}_seed{seed}_{method}.json')['archive'];points=np.array([x['objectives'] for x in archive]);points=np.where(zero,0.,(points-ideal)/span)
    out=dict(iid=iid,seed=seed,method=method,hv=float(moocore.hypervolume(points,ref=[1.1]*5)),igd_plus=float(moocore.igd_plus(points,ref=pf)))
    per_run.append(out);scores.append(out)
   for metric in ('hv','igd_plus'):
    row[f'{label}_{metric}_mean']=float(np.mean([s[metric] for s in scores]));row[f'{label}_{metric}_sd']=float(np.std([s[metric] for s in scores],ddof=1))
    for suffix in ('mean','sd'):
     key=f'{label}_{metric}_{suffix}';np.testing.assert_allclose(row[key],float(item[key]),atol=1e-10,rtol=0)
  per_iid.append(row)
  print('VERIFIED',iid,flush=True)
 tests={}
 for metric,sign in [('hv',1),('igd_plus',-1)]:
  d=np.array([r[f'v5_{metric}_mean']-r[f'heuristic_{metric}_mean'] for r in per_iid])*sign
  t=wilcoxon(d,alternative='two-sided',zero_method='wilcox',method='auto');tests[metric]=dict(W=float(t.statistic),raw_p=float(t.pvalue))
 saved=json.loads((ROOT/'summary.json').read_text())['results']
 for metric in tests:
  np.testing.assert_allclose(tests[metric]['raw_p'],saved[f'{metric if metric=="hv" else "igd"}_test']['p'],atol=1e-12)
  tests[metric]['original_four_endpoint_holm_p']=saved[f'{metric if metric=="hv" else "igd"}_test']['holm_p_four_overall_endpoints']
 for filename,rows in [('per_run_metrics.csv',per_run),('main_table_recomputed.csv',per_iid)]:
  with (a.output/filename).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (a.output/'verification.json').write_text(json.dumps(dict(passed=True,raw_runs=len(per_run),instances=len(per_iid),tests=tests,note='Holm values retained from original four-endpoint family; per-run metrics and raw signed-rank p independently recalculated.'),indent=2))
 print('ALL PER-SEED METRICS VERIFIED',flush=True)
if __name__=='__main__':main()

import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv,time
from pathlib import Path
import numpy as np,moocore,scipy,psutil
from scipy.stats import wilcoxon,rankdata
psutil.Process().cpu_affinity([12,13]);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;OLD=OUT
plan=json.loads((ROOT/'plan.json').read_text());seeds=list(range(20261024,20261044))
assert all((ROOT/'results'/f'{i}_seed{s}_{m}.json').exists() for i in plan['ids'] for s in seeds for m in plan['methods'])
def head(path):
 data=b'';marker=b',"records":'
 with path.open('rb') as f:
  while marker not in data:
   chunk=f.read(65536)
   if not chunk:raise ValueError(path)
   data+=chunk
 return json.loads(data.split(marker,1)[0]+b'}')
def test(d):
 d=np.asarray(d);nz=d[d!=0];rr=rankdata(abs(nz));t=wilcoxon(d,alternative='two-sided',zero_method='wilcox',method='auto')
 return dict(W=float(t.statistic),p=float(t.pvalue),wins_ties_losses=[int((d>0).sum()),int((d==0).sum()),int((d<0).sum())],rank_biserial=float((rr[nz>0].sum()-rr[nz<0].sum())/rr.sum()),median_advantage=float(np.median(d)))
with (OLD/'instances.csv').open(encoding='utf-8-sig',newline='') as f:single=list(csv.DictReader(f))
pooled=[]
for item in single:
 iid=item['iid'];ref=json.loads((OLD/f'{iid}_reference.json').read_text());ideal=np.array(ref['empirical_ideal']);nadir=np.array(ref['empirical_nadir']);zero=np.zeros(5,dtype=bool);zero[ref['zero_span_objectives']]=True;scale=np.where(zero,1.,nadir-ideal);pf=np.array(ref['normalized_points']);row=dict(iid=iid,n=int(item['n']))
 for m,label in [('heuristic','heuristic'),('v5_meta_frozen','v5')]:
  a=np.concatenate([np.array([p['objectives'] for p in head(ROOT/'results'/f'{iid}_seed{s}_{m}.json')['archive']]) for s in seeds]);a=np.unique(a,axis=0);a=a[moocore.is_nondominated(a)];a=np.where(zero,0.,(a-ideal)/scale)
  row[label+'_hv']=float(moocore.hypervolume(a,ref=[1.1]*5));row[label+'_igd_plus']=float(moocore.igd_plus(a,ref=pf))
 pooled.append(row)
summary=dict(created_unix=time.time(),search_seeds=seeds,instances=45,paired_runs=900,raw_runs=1800,scipy_version=scipy.__version__,scope='Complete 45-instance x20-search-seed cohort. One frozen training seed. Benchmark previously examined; interim looks were not corrected; related instance variants limit independence.',normalization='Per-instance empirical nondominated union of both methods x twenty final archives; common ideal/nadir, ref=(1.1)^5, no clipping; span/physical_scale<=1e-12 coordinates set zero for both methods retaining five dimensions.',groups={})
for label,n in [('all',None),('N50',50),('N150',150),('N250',250)]:
 ii=[x for x in single if n is None or int(x['n'])==n];pp=[x for x in pooled if n is None or x['n']==n];g={}
 for kind,rows,suffix in [('single_run_mean',ii,'_mean'),('pooled_archive',pp,'')]:
  q={k:float(np.mean([float(x[k+suffix]) for x in rows])) for k in ['heuristic_hv','v5_hv','heuristic_igd_plus','v5_igd_plus']}
  q['hv_gain_percent']=100*(q['v5_hv']/q['heuristic_hv']-1);q['igd_reduction_percent']=100*(1-q['v5_igd_plus']/q['heuristic_igd_plus'])
  q['hv_test']=test([float(x['v5_hv'+suffix])-float(x['heuristic_hv'+suffix]) for x in rows]);q['igd_test']=test([float(x['heuristic_igd_plus'+suffix])-float(x['v5_igd_plus'+suffix]) for x in rows]);g[kind]=q
 summary['groups'][label]=g
with (OUT/'pooled_instances.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(pooled[0]));w.writeheader();w.writerows(pooled)
g=summary['groups']['all'];keys=[(k,m) for k in ['single_run_mean','pooled_archive'] for m in ['hv_test','igd_test']];prev=0
for j,k in enumerate(sorted(keys,key=lambda k:g[k[0]][k[1]]['p'])):
 prev=max(prev,min(1,(4-j)*g[k[0]][k[1]]['p']));g[k[0]][k[1]]['holm_p_four_overall_endpoints']=prev
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

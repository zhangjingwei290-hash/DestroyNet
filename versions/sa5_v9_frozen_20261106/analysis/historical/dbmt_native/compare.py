from pathlib import Path
import os,json,csv,statistics
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np,moocore,psutil
psutil.Process().cpu_affinity([14,15])
R=Path('D:/workspace/fuxian/DPR/runs/experiments/sa5_v9_scale_preference_20261105');D=R.parent/'dbmt_ls9_3seed_fullr2_20261103';F=Path('D:/workspace/destroy-repair-V0.01-publication/versions/v5_step5_r2_frozen_20261102');O=Path(__file__).resolve().parent
plan=json.loads((R/'plan.json').read_text());v9=json.loads((R/'v9_completed.json').read_text());boxes=json.loads((F/'evaluation_source/boxes.json').read_text());metrics=[];means=[];geometries={}
for iid in plan['test_ids']:
 data={};budgets={}
 for p in sorted((D/'results/dbmt_original_ls').glob(f'{iid}_*.json')):
  d=json.loads(p.read_text());assert not d['neural'] and not d['preflight'] and d['method']=='dbmt_original_ls';audit=d['independent_physical_audit'];assert audit.get('passed',False),audit
  assert d['budget_seconds']==plan['budgets'][iid];data['dbmt',d['seed']]=np.asarray([a['objectives'] for a in d['archive']],float);budgets['dbmt',d['seed']]=d['budget_seconds']
 for r in [r for r in v9 if r['iid']==iid]:
  d=json.loads(Path(r['path']).read_text());assert d['physical_audit_passed'] and d['budget_seconds']==plan['budgets'][iid];data['v9',r['seed']]=np.asarray([a['objectives'] for a in d['archive']],float);budgets['v9',r['seed']]=d['budget_seconds']
 assert sum(m=='v9' for m,s in data)==3 and sum(m=='dbmt' for m,s in data) in (1,2)
 points=np.unique(np.concatenate(list(data.values())),axis=0);pf=points[moocore.is_nondominated(points)];ideal=pf.min(0);span=pf.max(0)-ideal;scale=np.asarray(boxes[iid]['scale']);zero=span/scale<=1e-12
 def norm(x):return np.where(zero,0.,(x-ideal)/np.where(zero,1.,span))
 ref=norm(pf);geometries[iid]=dict(ideal=ideal.tolist(),nadir=pf.max(0).tolist(),zero_coordinates=np.flatnonzero(zero).tolist(),seeds={m:sorted(s for mm,s in data if mm==m) for m in ('dbmt','v9')})
 for (m,s),arr in data.items():metrics.append(dict(iid=iid,n=int(iid.split('-')[0]),method=m,seed=s,hv=float(moocore.hypervolume(norm(arr),ref=[1.1]*5)),igd_plus=float(moocore.igd_plus(norm(arr),ref=ref)),budget=budgets[m,s]))
 for m in ('dbmt','v9'):
  rr=[r for r in metrics if r['iid']==iid and r['method']==m];means.append(dict(iid=iid,n=int(iid.split('-')[0]),method=m,seeds=len(rr),hv=statistics.mean(r['hv'] for r in rr),hv_sd=statistics.stdev(r['hv'] for r in rr) if len(rr)>1 else None,igd_plus=statistics.mean(r['igd_plus'] for r in rr)))
macro=[]
for scope in ['all',50,150,250]:
 z={m:[r for r in means if r['method']==m and (scope=='all' or r['n']==scope)] for m in ('dbmt','v9')};a=statistics.mean(r['hv'] for r in z['dbmt']);b=statistics.mean(r['hv'] for r in z['v9']);ai=statistics.mean(r['igd_plus'] for r in z['dbmt']);bi=statistics.mean(r['igd_plus'] for r in z['v9']);lookup={r['iid']:r['hv'] for r in z['dbmt']};macro.append(dict(scope=scope,dbmt_hv=a,v9_hv=b,hv_change_percent=(b/a-1)*100,dbmt_igd_plus=ai,v9_igd_plus=bi,igd_reduction_percent=(1-bi/ai)*100,iid_hv_wins=sum(r['hv']>lookup[r['iid']] for r in z['v9']),iid_hv_losses=sum(r['hv']<lookup[r['iid']] for r in z['v9'])))
for name,rr in [('per_run',metrics),('per_instance',means),('comparison',macro)]:
 with (O/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
summary=dict(new_searches=0,dbmt_runs=15,v9_runs=27,reference='Pairwise common per-IID empirical ND over all available archives; own-seed scoring, then IID equal mean; ref1.1^5,zero span relative physical scale1e-12',comparison=macro,limits=['DBMT native whole solver; excludes modified window baseline and all neural adapters','Different search seeds, DBMT2seed at N50/150 and1seed at N250, V9 3seed each','Native original DBMT rmp integer zero retained; describes actual source implementation','CPU/GPU allocation and historical loads differ, same wall budgets117-338s are not equal resource budgets','Development9IID; historical descriptive comparison only; no paired-seed significance claim'])
(O/'summary.json').write_text(json.dumps(summary,indent=2));(O/'reference_geometry.json').write_text(json.dumps(geometries,indent=2));(O/'AUDIT.json').write_text(json.dumps(dict(passed=True,metrics=len(metrics),native_sources15=True,source_physical_audits=True,same_total_budgets=True,new_searches=0),indent=2))
s='# V9 versus native DBMT historical comparison\n\n'+summary['reference']+'\n\n|N|DBMT HV|V9 HV|HV change|DBMT IGD+|V9 IGD+|\n|---|---:|---:|---:|---:|---:|\n'
for z in macro:s+=f"|{z['scope']}|{z['dbmt_hv']:.6f}|{z['v9_hv']:.6f}|{z['hv_change_percent']:+.2f}%|{z['dbmt_igd_plus']:.6f}|{z['v9_igd_plus']:.6f}|\n"
s+='\n'+'\n'.join('- '+x for x in summary['limits'])+'\n';(O/'REPORT.md').write_text(s,encoding='utf-8');print(json.dumps(summary),flush=True)

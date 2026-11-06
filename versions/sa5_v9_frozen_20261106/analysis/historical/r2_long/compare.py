from pathlib import Path
import os,sys,json,gzip,csv,statistics
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import psutil,numpy as np,moocore
psutil.Process().cpu_affinity([14,15])
R=Path('D:/workspace/fuxian/DPR/runs/experiments/sa5_v9_scale_preference_20261105');F=Path('D:/workspace/destroy-repair-V0.01-publication/versions/v5_step5_r2_frozen_20261102');O=Path(__file__).resolve().parent
plan=json.loads((R/'plan.json').read_text());runs=json.loads((R/'v9_completed.json').read_text());metrics=[];means=[];geometry={};checks=[]
for iid in plan['test_ids']:
 data={};metadata={}
 with gzip.open(F/f'results/compact_reports/step5_stage2_r2/{iid}.jsonl.gz','rt',encoding='utf-8') as f:
  for line in f:
   d=json.loads(line)['report'];seed=d['seed'];assert not d['technical_smoke'] and d['frozen_tensors_verified'];assert d['iid']==iid and d['budget_seconds']==plan['budgets'][iid]
   data['r2_long',seed]=np.asarray([a['objectives'] for a in d['archive']],float);scale=np.asarray(d['physical_scale'],float);metadata['r2_long',seed]=dict(seconds=d['search_seconds'],attempts=d['steps'],budget=d['budget_seconds'])
 for rec in [a for a in runs if a['iid']==iid]:
  d=json.loads(Path(rec['path']).read_text());assert d['physical_audit_passed'] and d['budget_seconds']==plan['budgets'][iid]
  seed=rec['seed'];data['v9_sa5',seed]=np.asarray([a['objectives'] for a in d['archive']],float);metadata['v9_sa5',seed]=dict(seconds=d['online_seconds'],attempts=len(d['records']),budget=d['budget_seconds'])
 oldseeds=sorted(s for m,s in data if m=='r2_long');newseeds=sorted(s for m,s in data if m=='v9_sa5');assert len(oldseeds)==20 and len(newseeds)==3 and not set(oldseeds)&set(newseeds)
 points=np.unique(np.concatenate(list(data.values())),axis=0);pf=points[moocore.is_nondominated(points)];ideal=pf.min(0);span=pf.max(0)-ideal;zero=span/scale<=1e-12
 def norm(x):return np.where(zero,0.,(np.asarray(x)-ideal)/np.where(zero,1.,span))
 ref=norm(pf);geometry[iid]=dict(ideal=ideal.tolist(),nadir=pf.max(0).tolist(),zero_coordinates=np.flatnonzero(zero).tolist(),old_seeds=oldseeds,new_seeds=newseeds,empirical_points=len(pf))
 for (m,s),arr in data.items():metrics.append(dict(iid=iid,n=int(iid.split('-')[0]),method=m,seed=s,hv=float(moocore.hypervolume(norm(arr),ref=[1.1]*5)),igd_plus=float(moocore.igd_plus(norm(arr),ref=ref)),**metadata[m,s]))
 for m in ['r2_long','v9_sa5']:
  local=[x for x in metrics if x['iid']==iid and x['method']==m];means.append(dict(iid=iid,n=int(iid.split('-')[0]),method=m,seeds=len(local),hv=statistics.mean(x['hv'] for x in local),hv_sd=statistics.stdev(x['hv'] for x in local),igd_plus=statistics.mean(x['igd_plus'] for x in local),igd_plus_sd=statistics.stdev(x['igd_plus'] for x in local),attempts=statistics.mean(x['attempts'] for x in local),seconds=statistics.mean(x['seconds'] for x in local)))
 checks.append(dict(iid=iid,same_total_budget=True,old_runs=20,new_runs=3,no_paired_seeds=True))
 print('Scored',iid,flush=True)
macro=[];comparison=[]
for scope in ['all',50,150,250]:
 chosen=[x for x in means if scope=='all' or x['n']==scope]
 for m in ['r2_long','v9_sa5']:
  local=[x for x in chosen if x['method']==m];macro.append(dict(scope=scope,method=m,iids=len(local),hv=statistics.mean(x['hv'] for x in local),igd_plus=statistics.mean(x['igd_plus'] for x in local),attempts=statistics.mean(x['attempts'] for x in local)))
 aa={x['method']:x for x in macro if x['scope']==scope};ds=[];igd=[]
 for iid in sorted({x['iid'] for x in chosen}):
  z={x['method']:x for x in chosen if x['iid']==iid};ds.append(z['v9_sa5']['hv']-z['r2_long']['hv']);igd.append(z['r2_long']['igd_plus']-z['v9_sa5']['igd_plus'])
 comparison.append(dict(scope=scope,hv_old=aa['r2_long']['hv'],hv_v9=aa['v9_sa5']['hv'],hv_relative_percent=(aa['v9_sa5']['hv']/aa['r2_long']['hv']-1)*100,igd_old=aa['r2_long']['igd_plus'],igd_v9=aa['v9_sa5']['igd_plus'],igd_reduction_percent=(1-aa['v9_sa5']['igd_plus']/aa['r2_long']['igd_plus'])*100,hv_wins=sum(x>0 for x in ds),hv_ties=sum(x==0 for x in ds),hv_losses=sum(x<0 for x in ds),igd_wins=sum(x>0 for x in igd),igd_losses=sum(x<0 for x in igd)))
for name,rr in [('per_run',metrics),('per_instance',means),('macro',macro),('comparison',comparison)]:
 with (O/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
summary=dict(new_searches=0,old_runs=180,new_runs=27,scoring='Each seed archive scored separately under two-method per-IID common empirical ND from old20+new3; HV ref1.1^5; zero relative physical scale threshold1e-12; IID equally weighted',comparison=comparison,limits=['Different non-overlapping search seeds and unequal sample sizes:20 old vs3 new; not paired-seed comparison','Old R2 long sessions use min(10,T/16,remaining); V9 uses five-step SA and parallel direction repairs','Model/controller/q mapping/acceptance/cache/scheduling also differ: full-system historical comparison, not isolated local-session-budget effect','Different historical CPU resource allocation/load epochs; same wall budget does not ensure same CPU core seconds','Nine IID are repeatedly inspected development instances'])
(O/'summary.json').write_text(json.dumps(summary,indent=2));(O/'reference_geometry.json').write_text(json.dumps(geometry,indent=2));(O/'AUDIT.json').write_text(json.dumps(dict(passed=True,metrics=len(metrics),budgets=checks,no_new_searches=True),indent=2))
s='# V9 versus original R2 long sessions\n\n'+summary['scoring']+'\n\n|N|R2 HV|V9 HV|HV change|R2 IGD+|V9 IGD+|\n|---|---:|---:|---:|---:|---:|\n'
for z in comparison:s+=f"|{z['scope']}|{z['hv_old']:.6f}|{z['hv_v9']:.6f}|{z['hv_relative_percent']:+.2f}%|{z['igd_old']:.6f}|{z['igd_v9']:.6f}|\n"
s+='\n'+ '\n'.join('- '+x for x in summary['limits'])+'\n';(O/'REPORT.md').write_text(s,encoding='utf-8');print(json.dumps(summary),flush=True)

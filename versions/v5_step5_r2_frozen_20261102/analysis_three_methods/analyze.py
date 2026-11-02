"""Posthoc three-method analysis only; no solver/model imports."""
from pathlib import Path
import json,time,statistics,itertools
import psutil,numpy as np,moocore
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from stats import geometry,signed_rank,wtl,csv_write
R=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
psutil.Process().cpu_affinity([14,15])
plan=json.loads((R/'plan.json').read_text());parent=Path(json.loads((R/'PARENT_R2.json').read_text())['archive'])
ordinary=Path('D:/workspace/fuxian/DPR/runs/experiments/v5_realworld45_frozen_20seed_20261026/results')
methods=['heuristic','alns_tuned','step5_stage2_r2'];seeds=plan['search_seeds'];metrics=['empirical_hv','igd_plus'];runs=[];means=[];geoms={};bindings=[]
for iid in plan['test_ids']:
 reports=[];initials={};budget=None;scale=None
 for method in methods:
  for seed in seeds:
   if method=='heuristic':p=ordinary/f'{iid}_seed{seed}_heuristic.json'
   elif method=='alns_tuned':p=Path(plan['baseline_root'])/'metric_inputs/results'/f'{iid}_seed{seed}_alns_tuned.json'
   elif iid in plan['core_ids'] and seed<=20261026:p=parent/'stage2_evaluation/metric_inputs'/f'{iid}_seed{seed}_step5_stage2_retrained.json'
   else:p=R/'stage2_evaluation/metric_inputs'/f'{iid}_seed{seed}_step5_stage2_r2.json'
   st=p.stat();d=json.loads(p.read_text(encoding='utf-8-sig'));assert d['iid']==iid and d['seed']==seed
   if seed not in initials:initials[seed]=d['initial_population']
   if budget is None:budget=d['budget_seconds'];scale=d['physical_scale'];offset=d['physical_offset']
   assert d['initial_population']==initials[seed] and d['budget_seconds']==budget and d['physical_scale']==scale and d['physical_offset']==offset,(iid,seed,method)
   assert (st.st_size,st.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
   bindings.append(dict(path=str(p),size=st.st_size,mtime_ns=st.st_mtime_ns))
   reports.append(dict(iid=iid,seed=seed,method=method,physical_scale=scale,archive=d['archive']))
   del d
 transform,pf,geoms[iid]=geometry(reports)
 for d in reports:
  F=transform(np.asarray([x['objectives'] for x in d['archive']]))
  runs.append(dict(iid=iid,n=int(iid.split('-')[0]),seed=d['seed'],method=d['method'],empirical_hv=float(moocore.hypervolume(F,ref=[1.1]*5)),igd_plus=float(moocore.igd_plus(F,ref=pf))))
 for method in methods:
  selected=[x for x in runs if x['iid']==iid and x['method']==method];row=dict(iid=iid,n=int(iid.split('-')[0]),method=method,seeds=20)
  for m in metrics:row[m+'_mean']=statistics.mean(x[m] for x in selected);row[m+'_sample_sd']=statistics.stdev(x[m] for x in selected)
  means.append(row)
 print('completed',iid,flush=True)
macro=[];comparisons=[];idx={(x['iid'],x['seed'],x['method']):x for x in runs}
for scope in ['overall',50,150,250]:
 ids=[i for i in plan['test_ids'] if scope=='overall' or int(i.split('-')[0])==scope]
 for method in methods:
  row=dict(n=scope,method=method,iids=len(ids),runs=len(ids)*20)
  for m in metrics:
   row[m]=statistics.mean(x[m+'_mean'] for x in means if x['iid'] in ids and x['method']==method)
   row[m+'_macro_seed_sd']=statistics.stdev(statistics.mean(idx[i,s,method][m] for i in ids) for s in seeds)
  macro.append(row)
 family=[]
 for baseline,target in itertools.combinations(methods,2):
  for m in metrics:
   sign=-1 if m=='igd_plus' else 1
   rd=[sign*(idx[i,s,target][m]-idx[i,s,baseline][m]) for i in ids for s in seeds]
   dd=[statistics.mean(sign*(idx[i,s,target][m]-idx[i,s,baseline][m]) for s in seeds) for i in ids]
   family.append(dict(n=scope,baseline=baseline,target=target,metric=m,iid_wtl=wtl(dd),run_wtl=wtl(rd),**signed_rank(dd)))
 prev=0.
 for j,x in enumerate(sorted(family,key=lambda x:x['p'])):prev=max(prev,min(1.,x['p']*(6-j)));x['p_holm6']=prev
 comparisons+=family
assert len(runs)==2700 and len(means)==135
for name,data in [('per_run_metrics',runs),('main_table',means),('macro',macro),('comparisons',comparisons)]:csv_write(OUT/(name+'.csv'),data)
summary=dict(methods=methods,seeds=seeds,unique_metrics=2700,mean_sd_rows=135,initial_budget_physical_scale_matched=True,reference='three methods x20seed per IID common empirical ND; own-seed archives scored independently; zero span/physicalscale<=1e-12 set0 preserving5D; HVref1.1^5,no clipping',statistics='two-sided exact conditional paired signed-rank on IID20seed means; supplementary posthoc Holm6=three unordered pairs xHV/IGD per scope; original Holm9 report unchanged',macro=macro,comparisons=comparisons,geometry=geoms)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'SOURCE_RECEIPT.json').write_text(json.dumps(bindings,indent=2))
lines=['# 普通ALNS、alns_tuned与r2：45实例×20seed补充比较','','普通ALNS为已有heuristic方法，仍含轮盘赌自适应；固定版本不等于非自适应算法。只读复用900×3结果，逐条核验IID/seed/预算/完整初始群/物理尺度相同。只重算指标，不执行训练或搜索。共同参考仅包含这三方法，因此不能直接沿用此前四/六方法参考下的绝对值。','','|规模|方法|HV↑|IGD+↓|','|---|---|---:|---:|']
for x in macro:lines.append(f"|{x['n']}|{x['method']}|{x['empirical_hv']:.8f}|{x['igd_plus']:.8f}|")
lines+=['','|规模|对照|方法|指标|IID胜平负|W|原始p|补充Holm6 p|','|---|---|---|---|---|---:|---:|---:|']
for x in comparisons:lines.append(f"|{x['n']}|{x['baseline']}|{x['target']}|{x['metric']}|{x['iid_wtl']}|{x['W']}|{x['p']:.7g}|{x['p_holm6']:.7g}|")
lines+=['','此为用户指定方法集合的事后补充分析，Holm6并未替代原计划Holm9检验家族；两套结论单独保留。45例已检查且12例用于整套模型选择，不能称盲测。旧方法来源主机负载/时期不同，IID/seed配对不等于同步同worker配对。宏观SD为每seed的IID均值之间样本SD；逐IIDSD跨20seed。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(macro=macro,overall=[x for x in comparisons if x['n']=='overall'])),flush=True)

"""Read-only checks of selected R2 hashes, compact reports and statistics; stdlib only."""
from pathlib import Path
import ast,csv,gzip,hashlib,itertools,json,math,statistics
ROOT=Path(__file__).resolve().parents[1]
V=ROOT/'versions/v5_step5_r2_frozen_20261102'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def csvread(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def exact(ds):
 pairs=sorted((abs(d),d>0) for d in ds if d!=0);n=len(pairs);ranks=[];i=0
 while i<n:
  j=i+1
  while j<n and pairs[j][0]==pairs[i][0]:j+=1
  ranks.extend([(i+1+j,pairs[t][1]) for t in range(i,j)]);i=j
 pos=sum(rank for rank,positive in ranks if positive);total=sum(rank for rank,_ in ranks);lo=min(pos,total-pos)
 counts={0:1}
 for rank,_ in ranks:
  nxt=counts.copy()
  for s,c in counts.items():nxt[s+rank]=nxt.get(s+rank,0)+c
  counts=nxt
 p=min(1.,2*sum(c for s,c in counts.items() if s<=lo)/2**n) if n else 1.
 return pos/2,(total-pos)/2,p

def main():
 declaration=read(V/'FREEZE_DECLARATION.json');contents=read(V/'PACKAGE_CONTENTS.json')
 assert declaration['selected_model']=='r2' and declaration['formal_primary_runs']==900
 for row in contents['files']:
  p=V/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
 for key,digest in declaration['weights_sha256'].items():assert sha(V/'checkpoints'/key)==digest
 assert len(list((V/'checkpoints').glob('*.pt')))==17
 for manifest,prefix in [('R2_ORIGINAL_SOURCE_MANIFEST.json','training_source'),('EFFECTIVE_EVALUATION_SOURCE_MANIFEST.json','evaluation_source')]:
  for rel,digest in read(V/'evidence'/manifest).items():assert sha(V/prefix/rel)==digest,(manifest,rel)
 fit=read(V/'data/FIT_MANIFEST.json');roles={k:{p.stem for p in (V/'data'/k).glob('*.json')} for k in ['fit','validation','test']}
 assert [len(roles[k]) for k in ['fit','validation','test']]==[180,45,45]
 assert not (roles['fit']&roles['validation'] or roles['fit']&roles['test'] or roles['validation']&roles['test'])
 for iid,digest in fit['input_sha256'].items():assert sha(V/'data/fit'/f'{iid}.json')==digest
 provenance=read(V/'results/REPORT_PROVENANCE.json');proof={(x['method_key'],x['iid'],x['seed']):x for x in provenance};assert len(proof)==len(provenance)==2700
 methods=['heuristic','alns_tuned','step5_stage2_r2'];direction_sha=[declaration['weights_sha256'][f'direction_{i:02d}.pt'] for i in range(16)]
 seen=set();initials={};physical={};compact_count=0
 for p in sorted((V/'results/compact_reports').rglob('*.jsonl.gz')):
  count=0
  with gzip.open(p,'rt',encoding='utf-8') as f:
   for line in f:
    x=json.loads(line);d=x['report'];key=(x['method_key'],d['iid'],d['seed']);assert key not in seen;seen.add(key);assert key in proof
    assert not set(d).intersection(['records','variation_events','point_events','worker_pid','worker_created','worker_affinity','resource_snapshot','execution'])
    initkey=(d['iid'],d['seed']);initials.setdefault(initkey,d['initial_population']);assert initials[initkey]==d['initial_population']
    assert len(d['initial_population']['routes'])==len(d['initial_population']['objectives'])==16
    bind=(d['budget_seconds'],d['physical_offset'],d['physical_scale']);physical.setdefault(d['iid'],bind);assert physical[d['iid']]==bind
    assert all(len(pt['objectives'])==5 and pt['routes'] for pt in d['archive'])
    if key[0]=='step5_stage2_r2':assert d['checkpoint_sha256']==direction_sha and d['frozen_tensors_verified']
    count+=1
  assert count==20;compact_count+=1
 assert len(seen)==2700 and compact_count==135 and all(sum(k[0]==m for k in seen)==900 for m in methods)
 s=V/'analysis_three_methods';rr=csvread(s/'per_run_metrics.csv');means=csvread(s/'main_table.csv');idx={(r['iid'],r['method'],int(r['seed'])):r for r in rr};mi={(r['iid'],r['method']):r for r in means}
 assert len(idx)==len(rr)==2700 and len(mi)==len(means)==135
 seeds=declaration['search_seeds'];iids=sorted(roles['test']);ks=['empirical_hv','igd_plus']
 for row in means:
  for k in ks:
   vs=[float(idx[row['iid'],row['method'],seed][k]) for seed in seeds]
   assert math.isclose(statistics.mean(vs),float(row[k+'_mean']),abs_tol=1e-12)
   assert math.isclose(statistics.stdev(vs),float(row[k+'_sample_sd']),abs_tol=1e-12)
 comparisons=csvread(s/'comparisons.csv')
 for c in comparisons:
  ids=[i for i in iids if c['n']=='overall' or i.split('-')[0]==c['n']];k=c['metric'];a=c['baseline'];b=c['target'];factor=1 if k=='empirical_hv' else -1
  ds=[statistics.mean(factor*(float(idx[i,b,seed][k])-float(idx[i,a,seed][k])) for seed in seeds) for i in ids]
  rp,rm,p=exact(ds);assert math.isclose(p,float(c['p']),abs_tol=1e-12) and min(rp,rm)==float(c['W'])
 for scope in {c['n'] for c in comparisons}:
  family=sorted([c for c in comparisons if c['n']==scope],key=lambda c:float(c['p']));last=0.
  for j,c in enumerate(family):last=max(last,min(1.,float(c['p'])*(6-j)));assert math.isclose(last,float(c['p_holm6']),abs_tol=1e-12)
 rows=csvread(s/'instance_paper_tables/Wilcoxon_逐实例_完整p值.csv');assert len(rows)==270
 for c in rows:
  k='empirical_hv' if c['metric']=='HV' else 'igd_plus';factor=1 if k=='empirical_hv' else -1
  ds=[factor*(float(idx[c['iid'],c['target_key'],seed][k])-float(idx[c['iid'],c['baseline_key'],seed][k])) for seed in seeds]
  rp,rm,p=exact(ds);assert rp==float(c['R_plus']) and rm==float(c['R_minus']) and math.isclose(p,float(c['p']),abs_tol=1e-12)
 for iid in iids:
  family=sorted([r for r in rows if r['iid']==iid],key=lambda c:float(c['p']));last=0.
  for j,c in enumerate(family):last=max(last,min(1.,float(c['p'])*(6-j)));assert math.isclose(last,float(c['p_holm6']),abs_tol=1e-12)
 training=read(V/'training/training_report.json');assert (training['meta_batches'],training['meta_merges'],training['optimized_transitions'])==(60,15,223330)
 assert training['full_fit_coverage'] and training['all_direction_targets_met']
 for f in V.rglob('*.py'):ast.parse(f.read_text(encoding='utf-8-sig'))
 large=[str(f.relative_to(ROOT)) for f in V.rglob('*') if f.is_file() and f.stat().st_size>=100*1024*1024];assert not large,large
 print(json.dumps(dict(passed=True,version=V.name,package_hashes_verified=len(contents['files']),r2_models=17,directions=16,source_locks_verified=2,input_roles=[180,45,45],compact_runs_verified=2700,initial_populations_paired=900,checkpoint_bindings_verified=900,per_run_metrics=2700,mean_sd_rows=135,aggregate_exact_tests=24,per_instance_exact_tests=270,all_holm_recomputed=True,original_raw_feasibility_audit='completed audit receipts retained; full search not rerun',model_loaded=False,training_or_search_executed=False),ensure_ascii=False))
if __name__=='__main__':main()

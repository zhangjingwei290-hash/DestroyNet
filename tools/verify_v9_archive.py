"""Check published V9 bytes and recorded statistics without neural execution."""
from pathlib import Path
import json,hashlib,ast,csv,gzip,statistics,math,itertools
ROOT=Path(__file__).resolve().parents[1];V=ROOT/'versions/sa5_v9_frozen_20261106'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-11),(a,b)
def csvrows(name):return list(csv.DictReader((V/'analysis/main'/name).open(encoding='utf-8-sig')))
def exact(diffs):
 d=[float(x) for x in diffs if x!=0];values=sorted(abs(x) for x in d);lookup={}
 for value in set(values):ids=[i+1 for i,x in enumerate(values) if x==value];lookup[value]=sum(ids)/len(ids)
 ranks=[lookup[abs(x)] for x in d];total=sum(ranks);obs=sum(r for r,x in zip(ranks,d) if x>0)
 return sum(abs(sum(a*b for a,b in zip(s,ranks))-total/2)>=abs(obs-total/2)-1e-12 for s in itertools.product((0,1),repeat=len(d)))/2**len(d)
def main():
 package=read(V/'PACKAGE_CONTENTS.json')
 for x in package['files']:assert sha(V/x['path'])==x['sha256'],x['path']
 copies=read(V/'evidence/COPY_PROVENANCE.json')
 for x in copies:assert sha(ROOT/x['path'])==x['sha256'],x['path']
 source=V/'source';py=list(source.rglob('*.py'))
 for p in py:ast.parse(p.read_text(encoding='utf-8-sig'))
 decl=read(V/'FREEZE_DECLARATION.json');assert len(decl['expert_files'])==16
 for x in decl['expert_files']:
  p=V/x['path'];assert sha(p)==x['sha256'];assert sha(p)==sha(ROOT/'versions/v5_step5_r2_frozen_20261102/checkpoints'/p.name)
 assert sha(V/decl['selected_controller'])==decl['selected_sha256']==sha(V/'checkpoints/controller/cycle_3.pt')
 stages={stage:len(list((V/f'results/{stage}').rglob('*.gz'))) for stage in ['test','validation','training','smoke','historical_dbmt']};assert stages==dict(test=81,validation=90,training=18,smoke=3,historical_dbmt=15),stages
 source_audits=list((V/'evidence/audits').rglob('*.json'));assert len(source_audits)==192 and all(read(p)['passed'] for p in source_audits)
 rows=csvrows('per_run.csv');means=csvrows('per_instance.csv');macro=csvrows('macro.csv');tests=csvrows('tests.csv');wtl=csvrows('win_tie_loss.csv');assert len(rows)==81 and len(means)==27 and len(tests)==16
 # Match compact archives to paired test metadata and recorded throughput provenance.
 provenance=read(V/'results/REPORT_PROVENANCE.json');counts={(x['method'],x['iid'],x['seed']):x for x in provenance if x['stage']=='test'}
 for row in rows:
  m=row['method'];iid=row['iid'];seed=int(row['seed']);p=V/f'results/test/{m}/{iid}_seed{seed}.json.gz'
  with gzip.open(p,'rt',encoding='utf-8') as f:d=json.load(f)
  assert d['iid']==iid and d['seed']==seed and d['physical_audit_passed'];close(d['budget_seconds'],row['budget']);close(row['batches'],counts[m,iid,seed]['batches']);close(row['attempts'],counts[m,iid,seed]['records'])
 for iid in {r['iid'] for r in rows}:
  for seed in {int(r['seed']) for r in rows if r['iid']==iid}:
   initial=[]
   for m in ['roulette','v8','v9']:
    with gzip.open(V/f'results/test/{m}/{iid}_seed{seed}.json.gz','rt',encoding='utf-8') as f:initial.append(json.load(f)['initial_population'])
   assert initial[0]==initial[1]==initial[2]
 for r in means:
  local=[x for x in rows if x['iid']==r['iid'] and x['method']==r['method']];assert len(local)==3
  for metric in ['hv','igd_plus']:
   vals=[float(x[metric]) for x in local];close(r[metric],statistics.mean(vals));close(r[metric+'_sd'],statistics.stdev(vals))
 for r in macro:
  local=[x for x in means if x['method']==r['method'] and (r['scope']=='all' or x['n']==r['scope'])]
  for key in ['hv','igd_plus']:close(r[key],statistics.mean(float(x[key]) for x in local))
 lookup={(x['iid'],x['method']):x for x in means};checks=[]
 for t in tests:
  ids=sorted({x['iid'] for x in means if t['scope']=='all' or x['n']==t['scope']});diffs=[float(lookup[i,'v9'][t['metric']])-float(lookup[i,t['control']][t['metric']]) for i in ids];p=exact(diffs);close(p,t['raw_p']);checks.append(dict(scope=t['scope'],control=t['control'],metric=t['metric'],p=p))
 for scope in set(t['scope'] for t in tests):
  local=sorted([t for t in checks if t['scope']==scope],key=lambda t:t['p']);prev=0
  for rank,t in enumerate(local):
   prev=max(prev,min(1,(len(local)-rank)*t['p']));original=next(x for x in tests if all(x[k]==t[k] for k in ['scope','control','metric']));close(original['holm4_p'],prev)
 records=read(V/'training/COLLECTION_COUNTS.json');history=read(V/'evidence/training_history.json');assert sum(x['sampled_rows'] for x in records)==sum(x['actions'] for x in history)==181360;assert len(history)==6 and all(len(x['epochs'])==2 and x['strata']==144 and x['minimum_stratum_count']>0 for x in history)
 result=dict(passed=True,version=V.name,package_files=len(package['files']),byte_copies=len(copies),python_source_files=len(py),expert_models=16,controller_files=8,compact_reports=stages,source_audits=len(source_audits),metrics=81,mean_sd_rows=27,exact_tests=16,holm4_recomputed=True,common_initial_blocks=27,training_rows=181360,neural_model_loaded=False,training_or_search_executed=False)
 (V/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()

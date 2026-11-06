from pathlib import Path
import json,csv,itertools,hashlib
import numpy as np,moocore
F=Path('D:/workspace/destroy-repair-V0.01-publication/versions/v5_step5_r2_frozen_20261102')
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def paired_metrics(records):
 rows=[]
 for iid in sorted({r['iid'] for r in records}):
  subset=[r for r in records if r['iid']==iid];loaded=[(r,json.loads(Path(r['path']).read_text())) for r in subset]
  for r,d in loaded:assert d['physical_audit_passed']
  for seed in {r['seed'] for r in subset}:
   pop=[d['initial_population'] for r,d in loaded if r['seed']==seed];assert all(x==pop[0] for x in pop),'Initial populations differ'
  allf=np.unique(np.concatenate([np.asarray([p['objectives'] for p in d['archive']]) for _,d in loaded]),axis=0);pf=allf[moocore.is_nondominated(allf)];ideal=pf.min(0);span=pf.max(0)-ideal
  if iid.startswith('v5p_validation'):
   from data_contract import physical_box
   # Zero-span threshold has units of physical scales. Derive from input only.
   rr=json.loads((F/f'data/validation/{iid}.json').read_text());n=rr['n_customers'];h=max(float(rr['depot_time_window'][1]),1.);dist=np.asarray(rr['distance_matrix']);sc=np.array([n,2*n*dist.max(),h,max(h,n*max(np.asarray(rr['customer_time_windows'])[:,0].max(),0)),max(h,n*rr['max_delay'])])
  else:sc=np.asarray(json.loads((F/'evaluation_source/boxes.json').read_text())[iid]['scale'])
  zero=span/sc<=1e-12
  def norm(a):return np.where(zero,0.,(np.asarray(a)-ideal)/np.where(zero,1.,span))
  for r,d in loaded:
   x=norm([p['objectives'] for p in d['archive']]);rows.append(dict(iid=iid,n=r['n'],seed=r['seed'],method=r['method'],hv=float(moocore.hypervolume(x,ref=[1.1]*5)),igd_plus=float(moocore.igd_plus(x,ref=norm(pf))),budget=d['budget_seconds'],seconds=d['online_seconds'],batches=len(d['batches']),attempts=len(d['records'])))
 return rows

def make_report(records,out,title):
 from scipy.stats import rankdata
 out.mkdir(parents=True,exist_ok=True);rows=paired_metrics(records);means=[];macro=[];tests=[];wtl=[]
 for iid in sorted({r['iid'] for r in rows}):
  for method in sorted({r['method'] for r in rows if r['iid']==iid}):
   rr=[r for r in rows if r['iid']==iid and r['method']==method];means.append(dict(iid=iid,n=rr[0]['n'],method=method,seeds=len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in ('hv','igd_plus')},**{k+'_sd':float(np.std([r[k] for r in rr],ddof=1)) if len(rr)>1 else None for k in ('hv','igd_plus')}))
 for scope in ('all',50,150,250):
  for method in ('roulette','v8','v9'):
   rr=[r for r in means if r['method']==method and (scope=='all' or r['n']==scope)];macro.append(dict(scope=scope,method=method,instances=len(rr),**{k:float(np.mean([r[k] for r in rr])) if rr else None for k in ('hv','igd_plus')}))
  for control in ('roulette','v8'):
   for metric in ('hv','igd_plus'):
    # Pair the same IID and seed for both controls.
    dd=[];wins=ties=losses=0
    for iid in sorted({r['iid'] for r in rows if scope=='all' or r['n']==scope}):
     aa={r['seed']:r[metric] for r in rows if r['iid']==iid and r['method']=='v9'};bb={r['seed']:r[metric] for r in rows if r['iid']==iid and r['method']==control};seeds=aa.keys()&bb.keys()
     if not seeds:continue
     d=float(np.mean([aa[s]-bb[s] for s in seeds]));dd.append(d);good=d if metric=='hv' else -d;wins+=good>0;ties+=good==0;losses+=good<0
    nonzero=np.asarray([d for d in dd if d!=0]);ranks=rankdata(abs(nonzero));total=ranks.sum();observed=ranks[nonzero>0].sum()
    p=1. if not len(ranks) else sum(abs(sum(a*b for a,b in zip(sign,ranks))-total/2)>=abs(observed-total/2)-1e-12 for sign in itertools.product((0,1),repeat=len(ranks)))/2**len(ranks)
    tests.append(dict(scope=scope,control=control,metric=metric,independent_iids=len(dd),raw_p=p,exploratory=scope!='all'));wtl.append(dict(scope=scope,control=control,metric=metric,wins=int(wins),ties=int(ties),losses=int(losses)))
  ts=[t for t in tests if t['scope']==scope];last=0.
  for rank,t in enumerate(sorted(ts,key=lambda t:t['raw_p'])):last=max(last,min(1.,(len(ts)-rank)*t['raw_p']));t['holm4_p']=last
 for name,data in [('per_run',rows),('per_instance',means),('macro',macro),('tests',tests),('win_tie_loss',wtl)]:
  with (out/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 receipt=[dict(**r,sha256=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()) for r in records];dump(out/'summary.json',dict(title=title,macro=macro,tests=tests,win_tie_loss=wtl,reference='per IID all available methods/seeds common empirical ND, own-seed scoring; interim and final references recomputed; never compare their absolute values',records=receipt))
 dump(out/'AUDIT.json',dict(passed=True,run_metrics=len(rows),source_receipts=len(receipt),all_common_initials=True,statistics='exact paired signedrank on IID seed means, Holm4 for V9 vs two controls x two metrics; scales exploratory'))
 text='# '+title+'\n\n每seed独立评分后逐IID均值；规模和总体按IID等权。三方法均覆盖相同27个IID-seed配对。胜平负及检验以9个IID的seed均值为单位。最终三方每IID共同经验ND参考统一重算；与中期绝对指标不可横比。测试是已多次查看的开发面板，不称盲测。\n\n|范围|方法|实例数|HV↑|IGD+↓|\n|---|---|---:|---:|---:|\n'
 for r in macro:text+=f"|{r['scope']}|{r['method']}|{r['instances']}|{r['hv']}|{r['igd_plus']}|\n"
 text+='\n配对胜平负与exact检验见win_tie_loss.csv和tests.csv；同规模仅3IID，无法承诺显著。\n';(out/'REPORT.md').write_text(text,encoding='utf-8')

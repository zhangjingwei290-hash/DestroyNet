import csv, hashlib, importlib.util, itertools, json, math, os
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import psutil
from scipy.stats import rankdata, wilcoxon

p=psutil.Process();p.cpu_affinity([14,15]);p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('analysis',root/'analyze.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
for values in [[1,2,3],[-1,2,3],[1,-1,2,-2],[0,1,-1,3,3],[-2,-2,-2],[0,0]]:
    v=[x for x in values if abs(x)>1e-12]
    if v:
        ranks=rankdata([abs(x) for x in v]);observed=min(sum(r for x,r in zip(v,ranks) if x>0),sum(r for x,r in zip(v,ranks) if x<0))
        brute=sum(min(sum(r for r,s in zip(ranks,signs) if s),sum(r for r,s in zip(ranks,signs) if not s))<=observed for signs in itertools.product([0,1],repeat=len(v)))/2**len(v)
        assert abs(a.signed_rank(values)['p']-brute)<1e-14
summary=json.loads((root/'summary.json').read_text())
rows=list(csv.DictReader((root/'per_run.csv').open(encoding='utf-8-sig')))
assert len(rows)==900 and len({(r['iid'],r['seed']) for r in rows})==900
receipts=json.loads((root/'source_receipt.json').read_text());assert sum(r['actions'] for r in receipts)==summary['total_actions']
comparisons=[]
for metric,description in summary['metrics'].items():
    values=[]
    for iid in sorted({r['iid'] for r in rows}):
        vals=[float(r[metric]) for r in rows if r['iid']==iid and r[metric]!='']
        if vals:values.append(math.fsum(vals)/len(vals))
    assert len(values)==description['instances']
    assert abs(math.fsum(values)/len(values)-description['mean'])<1e-12
    nonzero=[v for v in values if abs(v)>1e-12]
    if metric.endswith('_controlled') and len({abs(v) for v in nonzero})==len(nonzero):
        check=wilcoxon(nonzero,alternative='two-sided',method='exact')
        assert abs(check.pvalue-description['signed_rank']['p'])<1e-12
        comparisons.append(metric)
result=dict(passed=True,reports=900,actions=summary['total_actions'],aggregation_recomputed=True,rank_fixtures='exhaustive sign enumeration including ties and zeros',scipy_exact_crosschecks=comparisons,analysis_sha256=hashlib.sha256((root/'analyze.py').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((root/'PROTOCOL.json').read_bytes()).hexdigest())
(root/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))

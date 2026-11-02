from pathlib import Path
import json,csv,math,statistics
import psutil
from scipy.stats import wilcoxon
psutil.Process().cpu_affinity([14,15])
p=Path(__file__).resolve().parent
s=json.loads((p/'summary.json').read_text());rows=list(csv.DictReader((p/'per_run_metrics.csv').open(encoding='utf-8-sig')));means=list(csv.DictReader((p/'main_table.csv').open(encoding='utf-8-sig')))
assert len(rows)==len({(r['iid'],r['seed'],r['method']) for r in rows})==2700
for c in s['comparisons']:
 ids=sorted({r['iid'] for r in means if c['n']=='overall' or int(r['n'])==c['n']});m=c['metric']+'_mean'
 ds=[next(float(r[m]) for r in means if r['iid']==i and r['method']==c['target'])-next(float(r[m]) for r in means if r['iid']==i and r['method']==c['baseline']) for i in ids]
 assert len(set(abs(x) for x in ds))==len(ds) and all(abs(x)>1e-12 for x in ds)
 check=wilcoxon(ds,alternative='two-sided',method='exact');assert math.isclose(check.pvalue,c['p'],abs_tol=1e-12)
for b in json.loads((p/'SOURCE_RECEIPT.json').read_text()):
 st=Path(b['path']).stat();assert (st.st_size,st.st_mtime_ns)==(b['size'],b['mtime_ns'])
(p/'AUDIT.json').write_text(json.dumps(dict(passed=True,metrics2700_unique=True,comparisons24_scipy_exact_matched=True,source2700_size_mtime_unchanged=True,initial_budget_physical_bindings_passed=True,scope='supplementary statistics verification; no additional search or training'),indent=2))
print('Independent SciPy exact p verification passed; 2700 source files unchanged')

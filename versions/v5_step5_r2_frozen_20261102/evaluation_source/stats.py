"""Prespecified per-seed empirical-reference metrics and exact paired tests."""
import numpy as np,moocore,csv,statistics
from collections import Counter
def geometry(reports):
    points=np.unique(np.concatenate([np.asarray([p['objectives'] for p in r['archive']]) for r in reports]),axis=0)
    pf=points[moocore.is_nondominated(points)];ideal=pf.min(0);nadir=pf.max(0);span=nadir-ideal
    zero=span/np.asarray(reports[0]['physical_scale'])<=1e-12
    def transform(a):return np.where(zero,0.,(np.asarray(a)-ideal)/np.where(zero,1.,span))
    return transform,transform(pf),dict(ideal=ideal.tolist(),nadir=nadir.tolist(),zero_span_coordinates=np.flatnonzero(zero).tolist(),reference=[1.1]*5)
def signed_rank(values):
    d=[float(v) for v in values if abs(v)>1e-12]
    if not d:return dict(W=0.,p=1.,nonzero=0)
    ranks=[2*sum(abs(v)<abs(x) for v in d)+sum(abs(v)==abs(x) for v in d)+1 for x in d]
    total=sum(ranks);positive=sum(r for r,x in zip(ranks,d) if x>0);observed=min(positive,total-positive)
    counts=Counter({0:1})
    for rank in ranks:
        nxt=counts.copy()
        for value,count in counts.items():nxt[value+rank]+=count
        counts=nxt
    return dict(W=observed/2,p=sum(c for x,c in counts.items() if min(x,total-x)<=observed)/2**len(d),nonzero=len(d))
def wtl(values):return [sum(v>1e-12 for v in values),sum(abs(v)<=1e-12 for v in values),sum(v< -1e-12 for v in values)]
def mean(values):
    a=[v for v in values if v is not None];return float(statistics.mean(a)) if a else None
def csv_write(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

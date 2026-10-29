import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv,time
from pathlib import Path
import psutil,numpy as np,moocore
psutil.Process().cpu_affinity([12,13]);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;plan=json.loads((ROOT/'plan.json').read_text());seeds=list(range(20261024,20261044))
def head(path):
    # Raw report serializes archive before large records. Read only the prefix.
    data=b'';marker=b',"records":'
    with path.open('rb') as f:
        while marker not in data:
            chunk=f.read(65536)
            if not chunk:raise ValueError('Missing records boundary '+str(path))
            data+=chunk
    return json.loads(data.split(marker,1)[0]+b'}')
pairs=[];iids=[]
for iid in plan['ids']:
    data={(s,m):head(ROOT/'results'/f'{iid}_seed{s}_{m}.json') for s in seeds for m in plan['methods']}
    first=next(iter(data.values()));offset=np.array(first['physical_offset']);scale=np.array(first['physical_scale']);ref=first['normalized_reference'];arrays={}
    for key,r in data.items():
        assert r['physical_offset']==first['physical_offset'] and r['physical_scale']==first['physical_scale'] and r['normalized_reference']==ref
        a=(np.array([p['objectives'] for p in r['archive']])-offset)/scale
        assert np.isfinite(a).all();arrays[key]=a
        np.testing.assert_allclose(moocore.hypervolume(a,ref=ref),r['hv'],rtol=1e-10,atol=1e-12)
    union=np.unique(np.concatenate(list(arrays.values())),axis=0);pf=union[moocore.is_nondominated(union)]
    physical_pf=pf*scale+offset
    ideal=physical_pf.min(0);nadir=physical_pf.max(0);span=nadir-ideal
    zero=(span/scale<=1e-12)
    # Drop degenerate PF coordinates from distance/volume geometry, retaining
    # zero coordinate and 1.1 reference in five-dimensional HV for all methods.
    safe=np.where(zero,1.,span)
    arrays={k:np.where(zero,0.,(v*scale+offset-ideal)/safe) for k,v in arrays.items()}
    pf=np.where(zero,0.,(physical_pf-ideal)/safe);ref=np.full(5,1.1)
    (OUT/f'{iid}_reference.json').write_text(json.dumps(dict(normalized_points=pf.tolist(),empirical_ideal=ideal.tolist(),empirical_nadir=nadir.tolist(),zero_span_objectives=np.flatnonzero(zero).tolist(),source='empirical ND union of 40 final archives; zero-span criterion (nadir-ideal)/physical_scale<=1e-12; zero coordinates for both methods')))
    local=[]
    for s in seeds:
        h=data[s,'heuristic'];v=data[s,'v5_meta_frozen'];assert h['budget_seconds']==v['budget_seconds']
        row=dict(iid=iid,n=int(iid.split('-')[0]),seed=s,heuristic_hv=float(moocore.hypervolume(arrays[s,'heuristic'],ref=ref)),v5_hv=float(moocore.hypervolume(arrays[s,'v5_meta_frozen'],ref=ref)),heuristic_igd_plus=float(moocore.igd_plus(arrays[s,'heuristic'],ref=pf)),v5_igd_plus=float(moocore.igd_plus(arrays[s,'v5_meta_frozen'],ref=pf)))
        row['hv_delta']=row['v5_hv']-row['heuristic_hv'];row['hv_gain_percent']=100*row['hv_delta']/row['heuristic_hv'] if row['heuristic_hv'] else None
        row['igd_plus_delta']=row['v5_igd_plus']-row['heuristic_igd_plus'];local.append(row);pairs.append(row)
    item=dict(iid=iid,n=int(iid.split('-')[0]),zero_span_objectives=','.join(str(j+1) for j in np.flatnonzero(zero)))
    for key in ('heuristic_hv','v5_hv','heuristic_igd_plus','v5_igd_plus','hv_delta','igd_plus_delta'):
        a=np.array([r[key] for r in local]);item[key+'_mean']=float(a.mean());item[key+'_sd']=float(a.std(ddof=1))
    iids.append(item)
summary=dict(scope='full twenty-search-seed evaluation; one training seed; previously examined benchmark; interim looks were not corrected',search_seeds=seeds,iids=45,pairs=900,metric_library='moocore',normalization='per-IID empirical PF ideal/nadir from pooled first-twenty-seed final archives; minimization; HV ref=(1.1)^5; no clipping; degenerate PF coordinates fixed zero for both methods',igd_reference='per-IID empirical ND union of both methods × twenty seeds (40 archives); complete 20-seed reference',groups={})
for name,n in [('all',None),('N50',50),('N150',150),('N250',250)]:
    rr=[x for x in pairs if n is None or x['n']==n];ii=[x for x in iids if n is None or x['n']==n];g=dict(iids=len(ii),pairs=len(rr))
    for k in ('heuristic_hv','v5_hv','heuristic_igd_plus','v5_igd_plus'):g[k]=float(np.mean([x[k+'_mean'] for x in ii]))
    g['hv_relative_gain_ratio_of_macro_means_percent']=100*(g['v5_hv']/g['heuristic_hv']-1)
    g['mean_paired_hv_relative_gain_percent']=float(np.mean([x['hv_gain_percent'] for x in rr if x['hv_gain_percent'] is not None]))
    g['igd_plus_relative_reduction_ratio_of_macro_means_percent']=100*(1-g['v5_igd_plus']/g['heuristic_igd_plus'])
    for k,sign in [('hv_delta',1),('igd_plus_delta',-1)]:
        a=np.array([x[k] for x in rr])*sign;g[k+'_pair_wins_ties_losses']=[int((a>1e-12).sum()),int((abs(a)<=1e-12).sum()),int((a< -1e-12).sum())]
        a=np.array([x[k+'_mean'] for x in ii])*sign;g[k+'_iid_mean_wins_ties_losses']=[int((a>1e-12).sum()),int((abs(a)<=1e-12).sum()),int((a< -1e-12).sum())]
    summary['groups'][name]=g
for name,rows in [('pairs.csv',pairs),('instances.csv',iids)]:
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

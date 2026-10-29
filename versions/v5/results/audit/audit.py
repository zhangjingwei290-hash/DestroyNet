"""Independent final archive/initial-population audit without loading network weights."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import moocore
import psutil

OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
psutil.Process().cpu_affinity([14,15]);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def parts(p):
    data=b'';marker=b',"records":'
    with p.open('rb') as f:
        while marker not in data:
            chunk=f.read(65536)
            if not chunk:raise ValueError('Missing archive boundary '+str(p))
            data+=chunk
        prefix=json.loads(data.split(marker,1)[0]+b'}')
        f.seek(0,2);f.seek(max(0,f.tell()-131072));tail=f.read()
    marker=b',"method":';i=tail.rfind(marker)
    if i<0:raise ValueError('Missing final identity metadata '+str(p))
    metadata=json.loads(b'{'+tail[i+1:])
    return prefix,metadata


def audit_solution(api,inst,routes,objectives):
    sol=api.Solution([api.Route(list(row)) for row in routes])
    if not api.feasible(sol,inst):raise ValueError('Complete hard-constraint failure')
    values=api.evaluate(sol,inst).to_array()
    if not np.isfinite(values).all():raise ValueError('Nonfinite objective')
    np.testing.assert_allclose(values,objectives,atol=1e-7,rtol=0)


def main():
    began=time.time();plan=read(ROOT/'plan.json');cfg=read(ROOT/'eval_config.json')
    sources=read(ROOT/'source_hashes.json');inputs=read(ROOT/'input_hashes.json')
    for name,digest in sources.items():
        if sha(ROOT/name)!=digest:raise ValueError('Changed frozen source '+name)
    for iid,item in inputs.items():
        if sha(item['path'])!=item['sha256']:raise ValueError('Changed frozen input '+iid)
    if sha(ROOT/'checkpoints/meta_complete.pt')!=plan['checkpoint_sha256']:raise ValueError('Changed checkpoint')
    complete=read(ROOT/'results/complete.json')
    if not complete['completed'] or not complete['frozen_weights_verified']:raise ValueError('Missing runtime frozen completion gate')
    sys.path.insert(0,str(ROOT/'repo'))
    from solver.data import project_api
    from run import load_config
    api=project_api(load_config(ROOT/'eval_config.json').project_root)
    data_cfg=read(ROOT/'data/v5_config.json')
    expected={f'{i}_seed{s}_{m}.json' for i in plan['ids'] for s in plan['search_seeds'] for m in plan['methods']}
    found={p.name for p in (ROOT/'results').glob('*_seed*.json')}
    if found!=expected or len(found)!=1800:raise ValueError('Missing/unexpected/duplicate raw identities')
    stats=dict(started_unix=began,source_files=len(sources),input_files=len(inputs),raw_runs=0,pairs=0,
        archived_solutions=0,initial_solutions=0,completed_instances=0,checkpoint_sha256=plan['checkpoint_sha256'])
    for iid in plan['ids']:
        row=read(data_cfg['inputs'][iid]['path']);inst=api.instance(row)
        for seed in plan['search_seeds']:
            reports=[]
            for method in plan['methods']:
                h,m=parts(ROOT/'results'/f'{iid}_seed{seed}_{method}.json')
                if (h['iid'],h['seed'],m['method'],m['checkpoint_sha256'])!=(iid,seed,method,plan['checkpoint_sha256']):raise ValueError('Raw identity mismatch')
                if h['budget_seconds']!=cfg['solve_seconds_by_instance'][iid]:raise ValueError('Changed budget')
                if len(m['initial_population']['routes'])!=16 or len(m['initial_population']['objectives'])!=16:raise ValueError('Wrong initial population size')
                if not h['archive']:raise ValueError('Empty final archive')
                objectives=np.asarray([p['objectives'] for p in h['archive']])
                if not np.all(moocore.is_nondominated(objectives)) or len(np.unique(objectives,axis=0))!=len(objectives):raise ValueError('Final archive not unique/nondominated')
                for p in h['archive']:
                    audit_solution(api,inst,p['routes'],p['objectives']);stats['archived_solutions']+=1
                reports.append((h,m));stats['raw_runs']+=1
            h,m=reports[0];v,n=reports[1]
            for field in ('physical_offset','physical_scale','normalized_reference','budget_seconds'):
                if h[field]!=v[field]:raise ValueError('Pair mismatch '+field)
            if m['initial_population']!=n['initial_population']:raise ValueError('Pair initial routes/objectives differ')
            for routes,values in zip(m['initial_population']['routes'],m['initial_population']['objectives']):
                audit_solution(api,inst,routes,values);stats['initial_solutions']+=1
            stats['pairs']+=1
        stats['completed_instances']+=1
        (OUT/'audit_progress.json').write_text(json.dumps(stats,indent=2))
        print(json.dumps(dict(iid=iid,completed_instances=stats['completed_instances'],archive_solutions=stats['archived_solutions'])),flush=True)
    for name,count in [('main_table.csv',45),('paired_results.csv',900)]:
        with (ROOT/'results'/name).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        if len(rows)!=count:raise ValueError('Wrong formal table count '+name)
    formal=read(ROOT/'results/summary.json')
    if (formal['runs'],formal['pairs'],len(formal['instances']))!=(1800,900,45):raise ValueError('Wrong summary counts')
    if not (ROOT/'results/REPORT.md').is_file():raise ValueError('Missing original report')
    stats.update(passed=True,seconds=time.time()-began,finished_unix=time.time(),
        frozen_sources_unchanged=True,frozen_inputs_unchanged=True,checkpoint_unchanged=True,
        runtime_frozen_model_verified=complete['frozen_weights_verified'],archives_all_feasible=True,
        archived_objectives_independently_recomputed=True,initial_populations_paired=True,
        main_table_rows=45,paired_results_rows=900,scope='Full independent final archive and paired initial population audit. No neural weight load, no search or training.')
    (OUT/'AUDIT.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats),flush=True)


if __name__=='__main__':main()

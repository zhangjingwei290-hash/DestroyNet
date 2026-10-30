"""Independent final CSV, reference, physical and frozen-artifact verification."""
from pilot_common import ROOT,read,write,sha
import sys,time,csv,statistics,math,itertools,traceback,json
from pathlib import Path
import psutil,numpy as np,moocore

def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-11),(a,b)
def rows(name):
    with (ROOT/'step2_analysis'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def exact_signed_rank(delta):
    values=[d for d in delta if abs(d)>1e-12]
    if not values:return 0.,1.
    ranks=[]
    for d in values:
        lower=sum(abs(v)<abs(d) for v in values)
        equal=sum(abs(v)==abs(d) for v in values)
        ranks.append(2*lower+equal+1)
    total=sum(ranks);positive=sum(r for r,d in zip(ranks,values) if d>0);observed=min(positive,total-positive)
    distribution=[sum(r for r,sign in zip(ranks,signs) if sign) for signs in itertools.product((0,1),repeat=len(ranks))]
    return observed/2,sum(min(s,total-s)<=observed for s in distribution)/len(distribution)

def main():
    started=time.time();psutil.Process().cpu_affinity([14,15]);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    done=read(ROOT/'complete.json');assert done['total_runs']==135 and done['new_runs']==81 and done['reused_runs']==54
    assert read(ROOT/'BASELINE_READONLY_VERIFICATION.json')['passed']
    plan=read(ROOT/'plan.json');ids=plan['test_ids'];seeds=plan['search_seeds']
    methods=['alns_tuned','v5_meta_frozen','step1_adapted','step2_meta_only','step2_adapted']
    metrics=rows('per_run_metrics.csv');table=rows('main_table.csv');summary=read(ROOT/'step2_analysis/summary.json')
    assert len(metrics)==135 and len(table)==45
    lookup={(r['iid'],int(r['seed']),r['method']):r for r in metrics};assert len(lookup)==135
    grouped={}
    for r in table:
        iid,method=r['iid'],r['method'];group=[lookup[iid,s,method] for s in seeds]
        for key in ('hv','igd_plus'):
            values=[float(x[key]) for x in group];close(r[key+'_mean'],statistics.mean(values));close(r[key+'_sd'],statistics.stdev(values))
        grouped[iid,method]=r
    for r in summary['methods']:
        for key in ('hv','igd_plus'):close(r[key+'_mean'],statistics.mean(float(grouped[i,r['method']][key+'_mean']) for i in ids))
    independent_tests=[]
    for test in summary['tests']:
        key=test['metric'];sign=1 if key=='hv' else -1
        delta=[sign*(float(grouped[i,test['method']][key+'_mean'])-float(grouped[i,'alns_tuned'][key+'_mean'])) for i in ids]
        w,p=exact_signed_rank(delta);close(test['W'],w);close(test['p'],p);independent_tests.append(dict(method=test['method'],metric=key,p=p,holm_p=float(test['holm_p'])))
    previous=0.
    for index,test in enumerate(sorted(independent_tests,key=lambda r:r['p'])):
        previous=max(previous,min(1.,(len(independent_tests)-index)*test['p']));close(test['holm_p'],previous)
    source_files=0
    for name in ('STEP1_SOURCE_MANIFEST.json','STEP2_SOURCE_MANIFEST.json'):
        for rel,digest in read(ROOT/name).items():assert sha(ROOT/rel)==digest,rel;source_files+=1
    assert sha(ROOT/'checkpoints/meta_complete.pt')==plan['checkpoint_sha256']
    for entry in read(ROOT/'data/COPY_RECEIPT.json'):assert sha(ROOT/'data/inputs'/(entry['iid']+'.json'))==entry['sha256']
    for iid in ids:assert sha(ROOT/'inputs'/(iid+'.json'))==sha(ROOT/'data/inputs'/(iid+'.json'))
    checkpoints={}
    for stage in (1,2):
        report=read(ROOT/f'step{stage}_training/training_report.json');assert report['all_directions_updated'] and report['no_validation_or_realworld_training']
        assert all(report['direction_updates'][str(k)]>=1 for k in range(16))
        checkpoints[stage]=[sha(ROOT/f'step{stage}_training/checkpoints/direction_{k:02d}.pt') for k in range(16)]
    old=read(ROOT/'step1_analysis/VERIFICATION.json')['direction_checkpoint_sha256']
    for k,digest in enumerate(checkpoints[1]):assert old[f'direction_{k:02d}.pt']==digest
    meta_digest=sha(ROOT/'step2_training/meta_pilot.pt')
    resume=read(ROOT/'step2_evaluation/resume_receipt.json')
    for entry in resume['preserved_raw_inventory']:
        st=(ROOT/'step2_evaluation'/entry['name']).stat();assert (st.st_size,st.st_mtime_ns)==(entry['size'],entry['mtime_ns'])
    sys.path.insert(0,str(ROOT/'physical'))
    from src.core.protocol import Route,Solution
    from src.data_generation.sharded import _instance_from_row
    from src.evaluation.vrptw_eval import evaluate_solution,is_solution_feasible
    counts=dict(new_reports=0,archive_entries=0,initial_entries=0,direction_entries=0,unique_physical_solutions=0)
    zero={}
    for iid in ids:
        instance=_instance_from_row(read(ROOT/'inputs'/f'{iid}.json'));physical_cache={};reports={}
        def physical(routes,expected):
            key=tuple(tuple(r) for r in routes)
            if key not in physical_cache:
                sol=Solution([Route(list(r)) for r in routes]);assert is_solution_feasible(sol,instance)
                physical_cache[key]=evaluate_solution(sol,instance).to_array();counts['unique_physical_solutions']+=1
            np.testing.assert_allclose(physical_cache[key],expected,atol=1e-7,rtol=0)
        for seed in seeds:
            initial=None;directions=None
            for method in methods:
                folder=ROOT/'baseline_cache' if method in methods[:2] else ROOT/('step1_evaluation' if method=='step1_adapted' else 'step2_evaluation')/'metric_inputs'
                r=read(folder/f'{iid}_seed{seed}_{method}.json');reports[seed,method]=r
                assert r['budget_seconds']==read(ROOT/'config.json')['solve_seconds_by_instance'][iid]
                if initial is None:initial=r['initial_population'];directions=[d['weight'] for d in r['per_direction']]
                else:assert initial==r['initial_population'] and directions==[d['weight'] for d in r['per_direction']]
                if method in methods[:2]:continue
                counts['new_reports']+=1;assert r['checkpoint_source_sha256']==plan['checkpoint_sha256']
                raw=folder.parent/f'{iid}_seed{seed}_{method}.json';assert raw.stat().st_size==r['raw_bytes']
                if method in methods[3:]:assert r['evaluated_checkpoint_sha256']==([meta_digest] if method=='step2_meta_only' else checkpoints[2])
                for point in r['archive']:physical(point['routes'],point['objectives']);counts['archive_entries']+=1
                for routes,values in zip(initial['routes'],initial['objectives']):physical(routes,values);counts['initial_entries']+=1
                for direction in r['per_direction']:
                    values=np.asarray(direction['objectives']);weight=np.asarray(direction['weight']);normalized=(values-np.asarray(r['physical_offset']))/np.asarray(r['physical_scale'])
                    close(direction['scalar'],max(weight*normalized)+.05*sum(weight*normalized));counts['direction_entries']+=1
        points=np.unique(np.concatenate([np.asarray([p['objectives'] for p in r['archive']]) for r in reports.values()]),axis=0)
        pf=points[moocore.is_nondominated(points)];ideal=pf.min(0);nadir=pf.max(0);span=nadir-ideal
        iszero=span/np.asarray(next(iter(reports.values()))['physical_scale'])<=1e-12;zero[iid]=np.flatnonzero(iszero).tolist()
        transform=lambda x:np.where(iszero,0.,(x-ideal)/np.where(iszero,1.,span))
        saved=read(ROOT/'step2_analysis/references'/f'{iid}.json');np.testing.assert_array_equal(saved['ideal'],ideal);np.testing.assert_array_equal(saved['nadir'],nadir)
        for (seed,method),r in reports.items():
            a=transform(np.asarray([p['objectives'] for p in r['archive']]))
            close(lookup[iid,seed,method]['hv'],moocore.hypervolume(a,ref=[1.1]*5));close(lookup[iid,seed,method]['igd_plus'],moocore.igd_plus(a,ref=transform(pf)))
        print(json.dumps(dict(iid=iid,physical_counts=counts,phase='verified')),flush=True)
        write(ROOT/'step2_analysis/verification_progress.json',dict(iid=iid,counts=counts,checked_unix=time.time()))
    assert counts['new_reports']==81
    report=dict(passed=True,raw_runs=135,new_runs=81,reused_runs=54,instances=9,seeds=seeds,
        mean_sample_sd_and_exact_signed_rank_holm_verified=True,all_per_seed_hv_igd_recomputed=True,
        new_archives_feasible_objectives_recomputed=True,physical_counts=counts,paired_initials_inputs_budgets_weights_verified=True,
        frozen_source_files_checked=source_files,input_files_checked=270,baseline_readonly=read(ROOT/'BASELINE_READONLY_VERIFICATION.json'),
        step1_checkpoint_sha256=checkpoints[1],step2_checkpoint_sha256=checkpoints[2],meta_checkpoint_sha256=meta_digest,
        original34_reports_preserved=True,cross_interruption_pairs=resume['cross_interruption_pairs'],zero_span_coordinates=zero,
        checked_unix=time.time(),verification_seconds=time.time()-started,torch_or_checkpoint_objects_loaded=False)
    write(ROOT/'step2_analysis/VERIFICATION.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        (ROOT/'step2_analysis/verification_failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
